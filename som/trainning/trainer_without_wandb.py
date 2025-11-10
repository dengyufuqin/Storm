# training/train.py

import os
import yaml
import logging
import random
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.amp import autocast
from torch.amp import GradScaler
import numpy as np
from sklearn.metrics import precision_score, recall_score, average_precision_score

from data.dataload import MultiYCBVDataset as MultiDataset
from models.som_multiview import SOMMultiview
from .loss import CombinedLoss
from transformers import CLIPTokenizerFast, CLIPTextModel

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# --------- Distributed init (do not change) ---------
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl', init_method='env://')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda', local_rank)
# ----------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if target.dim() == 4 and target.shape[1] == 1:
        target = target[:, 0]
    # pred_bin = (pred >= threshold).float()
    pred_probs = torch.sigmoid(pred)
    pred_bin = (pred_probs >= threshold).float()
    target_bin = (target >= 0.5).float()
    inter = (pred_bin * target_bin).sum(dim=(1, 2))
    uni   = ((pred_bin + target_bin) >= 1).float().sum(dim=(1, 2))
    eps = 1e-6
    return ((inter + eps) / (uni + eps)).mean().item()


class Trainer:
    def __init__(self, cfg_path: str, resume_ckpt: str = None):
        # --- load config ---
        self.cfg = load_config(cfg_path)

        # --- global random seed & deterministic settings ---
        seed = self.cfg['train'].get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.seed = seed

        # --- distributed / device ---
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device

        # --- optional language model ---
        self.use_lang = self.cfg['model'].get('use_lang', False)
        if self.use_lang:
            self.clip_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_text_encoder = (
                CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
                .to(self.device)
                .eval()
            )

        # --- build everything ---
        self._setup_logging()
        self._build_dataset()
        self._build_model()
        self._build_loss()
        self._build_optimizer()

        self.scaler = GradScaler()
        self.best_val_ap = 0.0
        self.best_step = -1
        self.start_step = 0
        self.ema_loss = None

        if resume_ckpt:
            self.load_checkpoint(resume_ckpt)

    def load_checkpoint(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            if self.local_rank == 0:
                self.logger.warning(f"Checkpoint path {ckpt_path} does not exist. Starting from scratch.")
            return

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        # For backward compatibility with old checkpoints that only store model state_dict
        if 'model_state_dict' not in checkpoint:
            model_state_dict = checkpoint
            if self.local_rank == 0:
                self.logger.info("Loading model state from an old checkpoint format.")
        else:
            model_state_dict = checkpoint['model_state_dict']

        # Adjust for DDP model keys
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_load.load_state_dict(model_state_dict)

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'step' in checkpoint:
            self.start_step = checkpoint['step']
        if 'best_val_ap' in checkpoint:
            self.best_val_ap = checkpoint['best_val_ap']
        if 'best_step' in checkpoint:
            self.best_step = checkpoint['best_step']
        if 'ema_loss' in checkpoint:
            self.ema_loss = checkpoint['ema_loss']

        if self.local_rank == 0:
            self.logger.info(f"Resumed from checkpoint {ckpt_path} at step {self.start_step}")

    def save_checkpoint(self, path: str, step: int, ema_loss: float):
        if self.local_rank != 0:
            return
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        checkpoint = {
            'step': step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_ap': self.best_val_ap,
            'best_step': self.best_step,
            'ema_loss': ema_loss,
        }
        torch.save(checkpoint, path)
        tqdm.write(f"  → Saved checkpoint to {path}")

    def _setup_logging(self):
        save_dir = self.cfg['train'].get('save_dir', 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, 'train.log')
        # only file handler, no console StreamHandler
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
            handlers=[logging.FileHandler(log_file, mode='w')]
        )
        self.logger = logging.getLogger()
        if self.local_rank == 0:
            self.logger.info(f"Logging to file: {log_file}")

    def _build_dataset(self):
        dc = self.cfg['data']
        size = dc.get('image_size', 512)
        num_views = self.cfg['model']['hsfa']['num_views']
        include_datasets = dc.get('include_datasets', None)
        self.train_ds = MultiDataset(dc['root'], split='test', image_size=size, num_views=num_views, use_augment=True, include_datasets=include_datasets)

        # split train / val with fixed seed
        # n = len(full)
        # v_n = int(0.01 * n)
        # t_n = n - v_n
        # g = torch.Generator().manual_seed(self.seed)
        # self.train_ds, self.val_ds = random_split(full, [t_n, v_n], generator=g)

        self.val_ds = MultiDataset(dc['root'], split='test', image_size=size, num_views=num_views, include_datasets=include_datasets)
        self.test_ds = MultiDataset(dc['root'], split='test', image_size=size, num_views=num_views, include_datasets=include_datasets)

        # distributed samplers
        train_sampler = DistributedSampler(
            self.train_ds, num_replicas=self.world_size, rank=self.local_rank,
            shuffle=True, seed=self.seed, drop_last=True
        )
        val_sampler = DistributedSampler(
            self.val_ds, num_replicas=self.world_size, rank=self.local_rank,
            shuffle=False, seed=self.seed, drop_last=False
        )
        test_sampler = DistributedSampler(
            self.test_ds, num_replicas=self.world_size, rank=self.local_rank,
            shuffle=False, seed=self.seed, drop_last=False
        )

        tc       = self.cfg['train']
        bs       = tc.get('batch_size', 1)
        nw       = tc.get('num_workers', 4)
        bs_test  = tc.get('batch_size', 1)
        nw_test  = tc.get('num_workers', 4)
        per_bs      = bs // self.world_size
        per_bs_val  = bs // self.world_size
        per_bs_test = bs_test // self.world_size

        def worker_init_fn(worker_id):
            np.random.seed(self.seed + worker_id)
            random.seed(self.seed + worker_id)

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=per_bs,
            sampler=train_sampler,
            num_workers=nw,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=per_bs_val,
            sampler=val_sampler,
            num_workers=nw,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init_fn,
        )
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=per_bs_test,
            sampler=test_sampler,
            num_workers=nw_test,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init_fn,
        )

        if self.local_rank == 0:
            self.logger.info(f"Dataset sizes │ Train: {len(self.train_ds)} │ Val: {len(self.val_ds)} │ Test: {len(self.test_ds)})")

    def _build_model(self):
        mc = self.cfg['model']
        model = SOMMultiview(
            query_encoder_args = mc['encoder'],
            prompt_encoder_args = mc['prompt'],
            hsfa_config       = mc['hsfa'],
            decoder_args      = mc['decoder'],
            device            = self.device,
        ).to(self.device)

        self.model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,   
        )

    def _build_loss(self):
        lc = self.cfg.get('loss', {})
        self.criterion = CombinedLoss(
            weight_fbeta   = lc.get('weight_fbeta',    1.0),
            fbeta_beta     = lc.get('fbeta_beta',      0.5),
            weight_tversky = lc.get('weight_tversky',  1.0),
            tversky_alpha  = lc.get('tversky_alpha',   0.3),
            tversky_beta   = lc.get('tversky_beta',    0.7),
            weight_focal   = lc.get('weight_focal',    0.2),
            focal_alpha    = lc.get('focal_alpha',     0.1),
            focal_gamma    = lc.get('focal_gamma',     2.0),
            weight_dice    = lc.get('weight_dice',     0.0),
            weight_iou     = lc.get('weight_iou',      0.0),
        )

    def _build_optimizer(self):
        tc = self.cfg['train']
        lr = tc.get('optimizer', {}).get('lr', tc.get('lr', 1e-4))
        wd = tc.get('optimizer', {}).get('weight_decay', 0.0)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        
        # Scheduler with warmup
        scheduler_cfg = tc.get('scheduler', {})
        max_steps = tc['max_steps']
        eta_min = scheduler_cfg.get('eta_min', 1e-6)
        warmup_steps = scheduler_cfg.get('warmup_steps', 0)

        if self.local_rank == 0:
            self.logger.info(f"Optimizer: AdamW lr={lr} wd={wd}")

        if warmup_steps > 0 and warmup_steps < max_steps:
            warmup_scheduler = lr_scheduler.LinearLR(
                self.optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps
            )
            main_scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_steps - warmup_steps, eta_min=eta_min
            )
            self.scheduler = lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps]
            )
            if self.local_rank == 0:
                self.logger.info(
                    f"Scheduler: Linear warmup for {warmup_steps} steps, "
                    f"then CosineAnnealingLR to step {max_steps} with eta_min={eta_min}."
                )
        else:
            # Original Cosine Annealing LR Scheduler
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max_steps,
                eta_min=eta_min
            )
            if self.local_rank == 0:
                self.logger.info(f"Scheduler: CosineAnnealingLR T_max={max_steps}, eta_min={eta_min}")

    @torch.no_grad()
    def validate_epoch(self) -> float:
        self.model.eval()
        total_iou, cnt = 0.0, 0
        all_probs, all_labels = [], []
        is_main = (self.local_rank == 0)
        loader = (tqdm(self.val_loader, desc=" Validate", leave=False)
                  if is_main else self.val_loader)

        for batch in loader:
            q = batch['query_img'].to(self.device)
            v = batch['view_imgs'].to(self.device)
            m = batch['mask_gt'].to(self.device)

            pred, _ = self.model(q, v, multimask_output=False, lang_ctx=None)
            if pred.shape[-2:] != m.shape[-2:]:
                pred = F.interpolate(pred, size=m.shape[-2:], mode='bilinear', align_corners=False)

            biou = compute_iou(pred, m)
            total_iou += biou * q.size(0)
            cnt += q.size(0)

            probs = pred.sigmoid().cpu().flatten().numpy()
            labels = m.cpu().flatten().numpy().astype(int)
            all_probs.append(probs)
            all_labels.append(labels)

            if is_main:
                loader.set_postfix(val_IoU=total_iou/cnt)

        avg_iou = total_iou / cnt if cnt > 0 else 0.0
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        preds_bin = (all_probs >= 0.5).astype(int)

        precision = precision_score(all_labels, preds_bin, zero_division=0)
        recall    = recall_score(all_labels, preds_bin, zero_division=0)
        ap        = average_precision_score(all_labels, all_probs)

        if is_main:
            msg = f"Validate ─ IoU: {avg_iou:.4f} │ Precision: {precision:.4f} │ Recall: {recall:.4f} │ AP: {ap:.4f}"
            tqdm.write(msg)
            self.logger.info(msg)
        return precision, recall, ap

    @torch.no_grad()
    def test_epoch(self, ckpt: str) -> float:
        self.model.eval()
        is_main = (self.local_rank == 0)
        if is_main:
            self.logger.info(f"Loaded checkpoint {ckpt} for testing")

        total_iou, cnt = 0.0, 0
        all_probs, all_labels = [], []
        loader = (tqdm(self.test_loader, desc="  Test", leave=False)
                  if is_main else self.test_loader)

        for batch in loader:
            q = batch['query_img'].to(self.device)
            v = batch['view_imgs'].to(self.device)
            m = batch['mask_gt'].to(self.device)

            pred, _ = self.model(q, v, multimask_output=False, lang_ctx=None)
            if pred.shape[-2:] != m.shape[-2:]:
                pred = F.interpolate(pred, size=m.shape[-2:], mode='bilinear', align_corners=False)

            # IoU 统计
            biou = compute_iou(pred, m)
            total_iou += biou * q.size(0)
            cnt += q.size(0)

            # 收集概率与标签
            probs  = pred.sigmoid().cpu().flatten().numpy()
            labels = m.cpu().flatten().numpy().astype(int)
            all_probs.append(probs)
            all_labels.append(labels)

            if is_main:
                loader.set_postfix(test_IoU=total_iou/cnt)

        # 计算最终指标
        avg_iou    = total_iou / cnt if cnt > 0 else 0.0
        all_probs  = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        preds_bin  = (all_probs >= 0.5).astype(int)

        precision = precision_score(all_labels, preds_bin, zero_division=0)
        recall    = recall_score(all_labels, preds_bin, zero_division=0)
        ap        = average_precision_score(all_labels, all_probs)

        if is_main:
            msg = (f"Test complete ─ IoU: {avg_iou:.4f} │ "
                   f"Precision: {precision:.4f} │ Recall: {recall:.4f} │ AP: {ap:.4f}")
            tqdm.write(msg)
            self.logger.info(msg)

        # 如果需要也可以返回 precision
        return avg_iou

    def run(self):
        tc = self.cfg['train']
        max_steps     = tc['max_steps']
        val_interval  = tc.get('val_interval', 1000)
        test_interval = tc.get('test_interval', 5000)
        ema_alpha     = tc.get('ema_alpha', 0.98)
        save_dir      = tc.get('save_dir', 'checkpoints')
        is_main       = (self.local_rank == 0)

        train_iter = iter(self.train_loader)
        step = self.start_step
        ema_loss = self.ema_loss

        if is_main:
            pbar = tqdm(total=max_steps, desc="Train", leave=True, initial=step)

        while step < max_steps:
            self.model.train()
            try:
                batch = next(train_iter)
            except StopIteration:
                self.train_loader.sampler.set_epoch(step)
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            q, v, m = batch['query_img'], batch['view_imgs'], batch['mask_gt']
            q, v, m = q.to(self.device), v.to(self.device), m.to(self.device)

            lang_ctx = None
            if self.use_lang:
                tokens = self.clip_tokenizer(
                    batch['description'], padding=True, return_tensors='pt'
                ).to(self.device)
                with torch.no_grad():
                    lang_ctx = self.clip_text_encoder(**tokens).pooler_output

            with autocast(device_type='cuda'):
                pred, iou_preds = self.model(q, v, multimask_output=False, lang_ctx=lang_ctx)
                if pred.shape[-2:] != m.shape[-2:]:
                    pred = F.interpolate(pred, size=m.shape[-2:], mode='bilinear', align_corners=False)
                loss = self.criterion(pred, m)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)
            
            # Clip gradients
            grad_clip_cfg = self.cfg['train'].get('grad_clip', {})
            if grad_clip_cfg.get('enabled', False):
                max_norm = grad_clip_cfg.get('max_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            loss_val = loss.item()
            if ema_loss is None:
                ema_loss = loss_val
            else:
                ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * loss_val

            step += 1
            if is_main:
                pbar.update(1)
                pbar.set_postfix({"ema_loss": f"{ema_loss:.4f}"})
            if is_main and step % 100 == 0:
                self.logger.info(f"[Step {step}] Loss: {loss_val:.4f} (EMA: {ema_loss:.4f})")
                
            if step % val_interval == 0 or step == max_steps:
                if is_main:
                    self.logger.info(f"[Step {step}] running validation…")
                if is_main and step > 0:
                    last = os.path.join(save_dir, "last.pth")
                    self.save_checkpoint(last, step, ema_loss)
                
                val_prec, val_recall, val_ap = self.validate_epoch()
                if val_ap > self.best_val_ap:
                    self.best_val_ap = val_ap
                    self.best_step = step
                    if is_main:
                        dst = os.path.join(save_dir, "best.pth")
                        self.save_checkpoint(dst, step, ema_loss)
                        tqdm.write(
                            f"  → New best AP={val_ap:.4f} at step {step}, saved to {dst}"
                        )
                    ema_loss = None

            if is_main and step > 0 and step % test_interval == 0:
                tqdm.write(f"[Step {step}] running test…")
                self.test()

        if is_main:
            pbar.close()
            self.logger.info(
                f"Training complete. Best AP={self.best_val_ap:.4f} at step {self.best_step}"
            )

    def test(self, ckpt: str = None):
        sd = self.cfg['train'].get('save_dir', 'checkpoints')
        ckpt = ckpt or os.path.join(sd, "best.pth")
        
        if not os.path.exists(ckpt):
            if self.local_rank == 0:
                self.logger.error(f"Checkpoint {ckpt} not found for testing.")
            return

        checkpoint = torch.load(ckpt, map_location=self.device)
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
        
        # Handle both new and old checkpoint formats
        if 'model_state_dict' in checkpoint:
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
        else:
            model_to_load.load_state_dict(checkpoint)

        return self.test_epoch(ckpt)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to train.yaml')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--ckpt', default=None, help='Checkpoint for test mode')
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    trainer = Trainer(args.config, resume_ckpt=args.resume)
    if args.mode == 'train':
        trainer.run()
    else:
        trainer.test(args.ckpt)