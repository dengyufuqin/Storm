import os
import yaml
import logging
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

import numpy as np
from sklearn.metrics import precision_score, recall_score, average_precision_score

from data.dataload import MultiYCBVDataset as MultiDataset
from models.som_multiview import SOMMultiview
from .loss import CombinedLoss
from transformers import CLIPTokenizerFast, CLIPTextModel


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    # Reduce to 2D mask if channel dim =1
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if target.dim() == 4 and target.shape[1] == 1:
        target = target[:, 0]
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= 0.5).float()
    inter = (pred_bin * target_bin).sum(dim=(1, 2))
    uni   = ((pred_bin + target_bin) >= 1).float().sum(dim=(1, 2))
    eps = 1e-6
    return ((inter + eps) / (uni + eps)).mean().item()


class Trainer:
    def __init__(self, cfg_path: str):
        self.cfg = load_config(cfg_path)
        self.use_lang = self.cfg['model'].get('use_lang', False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_logging()
        self._build_dataset()
        self._build_model()
        self._build_loss()
        self._build_optimizer()
        self.scaler = GradScaler()
        self.best_val_precision = 0.0
        self.best_epoch = -1

        if self.use_lang:
            self.clip_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")\
                .to(self.device).eval()

    def _setup_logging(self):
        save_dir = self.cfg['train'].get('save_dir', 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, 'train.log')
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, mode='w')
            ]
        )
        self.logger = logging.getLogger()
        self.logger.info(f"Logging to console and {log_file}")

    def _build_dataset(self):
        dc = self.cfg['data']
        size = dc.get('image_size', 512)
        num_views = self.cfg['model']['hsfa']['num_views']
        full = MultiDataset(dc['root'], image_size=size, num_views=num_views)

        # split train/val
        n = len(full)
        v_n = int(0.1 * n)
        t_n = n - v_n
        g = torch.Generator().manual_seed(42)
        self.train_ds, self.val_ds = random_split(full, [t_n, v_n], generator=g)
        self.test_ds = MultiDataset(dc['root'], image_size=size, num_views=num_views)

        tc = self.cfg['train']
        bs = tc.get('batch_size', 1)
        nw = tc.get('num_workers', 4)
        bs_test = tc.get('batch_size', 1)
        nw_test = tc.get('num_workers', 4)

        self.train_loader = DataLoader(self.train_ds, batch_size=bs, shuffle=True,
                                       num_workers=nw, pin_memory=True)
        self.val_loader   = DataLoader(self.val_ds,   batch_size=bs, shuffle=False,
                                       num_workers=nw, pin_memory=True)
        self.test_loader  = DataLoader(self.test_ds,  batch_size=bs_test, shuffle=False,
                                       num_workers=nw_test, pin_memory=True)

        self.logger.info(
            f"Dataset sizes │ Train: {t_n} │ Val: {v_n} │ Test: {len(self.test_ds)} (total {n})"
        )

    def _build_model(self):
        mc = self.cfg['model']
        self.model = SOMMultiview(
            query_encoder_args = mc['encoder'],
            prompt_encoder_args = mc['prompt'],
            hsfa_config = mc['hsfa'],
            decoder_args = mc['decoder'],
            device = self.device,
        ).to(self.device)
        self.logger.info(f"Built SOMMultiview model on {self.device}")

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
        self.logger.info("Initialized CombinedLoss")

    def _build_optimizer(self):
        tc = self.cfg['train']
        lr = tc.get('optimizer', {}).get('lr', tc.get('lr', 1e-4))
        wd = tc.get('optimizer', {}).get('weight_decay', 0.0)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.logger.info(f"Optimizer: AdamW lr={lr} wd={wd}")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"[Epoch {epoch+1}] Train", leave=False)
        for batch in pbar:
            q = batch['query_img'].to(self.device)
            v = batch['view_imgs'].to(self.device)
            m = batch['mask_gt'].to(self.device)

            # prepare language context if enabled
            lang_ctx = None
            if self.use_lang:
                texts = batch['description']
                tokens = self.clip_tokenizer(texts, padding=True, return_tensors='pt')\
                    .to(self.device)
                with torch.no_grad():
                    txt_out = self.clip_text_encoder(**tokens)
                lang_ctx = txt_out.pooler_output

            with autocast():
                pred, _ = self.model(q, v, multimask_output=False, lang_ctx=lang_ctx)
                if pred.shape[-2:] != m.shape[-2:]:
                    pred = F.interpolate(pred, size=m.shape[-2:],
                                         mode='bilinear', align_corners=False)
                loss = self.criterion(pred, m)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(avg_loss=total_loss/(pbar.n+1))

        avg = total_loss / len(self.train_loader)
        self.logger.info(f"Epoch {epoch+1} ─ Train loss: {avg:.4f}")
        return avg

    @torch.no_grad()
    def validate_epoch(self) -> float:
        self.model.eval()
        total_iou, cnt = 0.0, 0
        all_probs, all_labels = [], []

        pbar = tqdm(self.val_loader, desc=" Validate", leave=False)
        for batch in pbar:
            q = batch['query_img'].to(self.device)
            v = batch['view_imgs'].to(self.device)
            m = batch['mask_gt'].to(self.device)

            # reuse last lang_ctx if needed
            lang_ctx = None

            pred, _ = self.model(q, v, multimask_output=False, lang_ctx=lang_ctx)
            if pred.shape[-2:] != m.shape[-2:]:
                pred = F.interpolate(pred, size=m.shape[-2:],
                                     mode='bilinear', align_corners=False)

            biou = compute_iou(pred, m)
            total_iou += biou * q.size(0)
            cnt += q.size(0)

            probs = pred.sigmoid().cpu().flatten().numpy()
            labels = m.cpu().flatten().numpy().astype(int)
            all_probs.append(probs)
            all_labels.append(labels)

            pbar.set_postfix(val_IoU=total_iou/cnt)

        avg_iou = total_iou / cnt if cnt > 0 else 0.0
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        preds_bin  = (all_probs >= 0.5).astype(int)

        precision = precision_score(all_labels, preds_bin, zero_division=0)
        recall    = recall_score(all_labels, preds_bin, zero_division=0)
        ap        = average_precision_score(all_labels, all_probs)

        self.logger.info(
            f"Validate ─ IoU: {avg_iou:.4f} │ Precision: {precision:.4f} "
            f"│ Recall: {recall:.4f} │ AP: {ap:.4f}"
        )
        return precision

    @torch.no_grad()
    def test_epoch(self, ckpt: str) -> float:
        state = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.logger.info(f"Loaded checkpoint {ckpt} for testing")

        total_iou, cnt = 0.0, 0
        pbar = tqdm(self.test_loader, desc="  Test", leave=False)
        for batch in pbar:
            q = batch['query_img'].to(self.device)
            v = batch['view_imgs'].to(self.device)
            m = batch['mask_gt'].to(self.device)

            pred, _ = self.model(q, v, multimask_output=False, lang_ctx=None)
            if pred.shape[-2:] != m.shape[-2:]:
                pred = F.interpolate(pred, size=m.shape[-2:],
                                     mode='bilinear', align_corners=False)

            biou = compute_iou(pred, m)
            total_iou += biou * q.size(0)
            cnt += q.size(0)
            pbar.set_postfix(test_IoU=total_iou/cnt)

        avg_iou = total_iou / cnt if cnt > 0 else 0.0
        self.logger.info(f"Test complete ─ Avg IoU: {avg_iou:.4f}")
        return avg_iou

    def run(self):
        epochs = self.cfg['train'].get('epochs', 300)
        sd = self.cfg['train'].get('save_dir', 'checkpoints')
        for ep in range(epochs):
            self.train_epoch(ep)
            val_prec = self.validate_epoch()
            if val_prec > self.best_val_precision:
                self.best_val_precision = val_prec
                self.best_epoch = ep + 1
                dst = os.path.join(sd, "best.pth")
                torch.save(self.model.state_dict(), dst)
                self.logger.info(f"New best Precision={val_prec:.4f} at epoch {ep+1}, saved to {dst}")
        self.logger.info(f"Training complete. Best Precision={self.best_val_precision:.4f} at epoch {self.best_epoch}")

    def test(self, ckpt: str = None):
        sd = self.cfg['train'].get('save_dir', 'checkpoints')
        ckpt = ckpt or os.path.join(sd, "best.pth")
        return self.test_epoch(ckpt)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to train.yaml')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--ckpt', default=None, help='Checkpoint for test mode')
    args = parser.parse_args()

    trainer = Trainer(args.config)
    if args.mode == 'train':
        trainer.run()
    else:
        trainer.test(args.ckpt)
