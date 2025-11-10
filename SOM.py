#!/usr/bin/env python3
import os
import yaml
import numpy as np
from glob import glob
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

from som.models.som_multiview import SOMMultiview
from collections import OrderedDict

class SOMinfer:
    def __init__(self, config_path = "som/configs/train.yaml", ckpt_path = "som/checkpoints/best.pth", device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        mc = self.cfg["model"]
        self.model = SOMMultiview(
            query_encoder_args = mc['encoder'],
            prompt_encoder_args = mc['prompt'],
            hsfa_config = mc['hsfa'],
            decoder_args = mc['decoder'],
            device = self.device
        ).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)

        # --- FIX: Check if the checkpoint is a dictionary and extract model state ---
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            # Handle old checkpoints that only contain the model's state_dict
            state_dict = ckpt
        # --------------------------------------------------------------------------

        # remove 'module.' prefix if it exists (from DDP training)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

        self.img_size  = self.cfg["data"].get("image_size", self.cfg["model"].get("img_size", 512))
        self.num_views = self.cfg["model"]["hsfa"]["num_views"]

        # 与训练时相同的 RGB 预处理
        dino2_mean = [0.485, 0.456, 0.406]
        dino2_std  = [0.229, 0.224, 0.225]
        self.rgb_transform = Compose([
            Resize(self.img_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(self.img_size),
            ToTensor(),
            Normalize(mean=dino2_mean, std=dino2_std),
        ])

    def _preprocess(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path).convert("RGB")
        return self.rgb_transform(img).to(self.device)

    def _load_views(self, view_paths):
        views = []
        exts = ("jpg","png","jpeg")
        for p in view_paths:
            if os.path.isdir(p):
                files = []
                for e in exts:
                    files.extend(glob(os.path.join(p, f"*.{e}")))
                for fp in sorted(files):
                    views.append(self._preprocess(fp))
            else:
                views.append(self._preprocess(p))
        if len(views) < self.num_views:
            raise ValueError(f"需要 {self.num_views} 张视图，但只找到 {len(views)} 张")
        views = torch.stack(views[:self.num_views], dim=0)
        return views.unsqueeze(0)

    def predict_mask(self, query_path: str, view_paths, threshold: float = 0.5):
        q = self._preprocess(query_path)
        v = self._load_views(view_paths)

        with torch.no_grad():
            logits, _ = self.model(q.unsqueeze(0), v, multimask_output=False)
            logit = logits[0,0]
            up = F.interpolate(
                logit.unsqueeze(0).unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)
            mask_square = (up.sigmoid() >= threshold).cpu().numpy().astype(np.uint8)

        orig = Image.open(query_path)
        mask_pil = Image.fromarray(mask_square * 255, mode="L").resize(orig.size, resample=Image.NEAREST)
        return mask_pil

    def predict_masks_batch(self, query_paths: list[str], view_paths_list: list, threshold: float = 0.5):
        if not query_paths:
            return []

        q_batch = torch.stack([self._preprocess(p) for p in query_paths])
        
        v_batch_list = []
        for view_paths in view_paths_list:
            # _load_views returns (1, num_views, C, H, W), so we squeeze it.
            v_batch_list.append(self._load_views(view_paths).squeeze(0))
        v_batch = torch.stack(v_batch_list)

        with torch.no_grad():
            logits, _ = self.model(q_batch, v_batch, multimask_output=False)
            # logits shape: (batch_size, 1, H, W)
            
            # Upsample all logits in a batch
            upscaled_logits = F.interpolate(
                logits,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False
            )
            # upscaled_logits shape: (batch_size, 1, img_size, img_size)
            
            mask_squares = (upscaled_logits.sigmoid() >= threshold).cpu().numpy().astype(np.uint8)
            # mask_squares shape: (batch_size, 1, img_size, img_size)

        # Post-process masks and resize to original image dimensions
        pred_masks_pil = []
        for i, q_path in enumerate(query_paths):
            mask_square = mask_squares[i, 0] # Get the mask for the i-th image
            orig_size = Image.open(q_path).size
            mask_pil = Image.fromarray(mask_square * 255, mode="L").resize(orig_size, resample=Image.NEAREST)
            pred_masks_pil.append(mask_pil)
            
        return pred_masks_pil

    @staticmethod
    def compute_centroid_point(mask: np.ndarray):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        cy, cx = ys.mean(), xs.mean()
        iy, ix = int(round(cy)), int(round(cx))
        # 若质心在 mask 外，选最近的 mask 像素
        if not mask[iy, ix]:
            coords = np.column_stack((ys, xs))
            d2 = (coords[:,0] - cy)**2 + (coords[:,1] - cx)**2
            idx = int(d2.argmin())
            iy, ix = int(coords[idx,0]), int(coords[idx,1])
        return iy, ix

    def predict(self, query_path: str, view_paths, threshold: float = 0.5):
        mask_pil = self.predict_mask(query_path, view_paths, threshold)
        mask_arr = np.array(mask_pil) > 0
        pt = self.compute_centroid_point(mask_arr)
        return mask_pil, pt

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="som/configs/train.yaml")
    parser.add_argument("--ckpt", default="som/checkpoints/best.pth")
    parser.add_argument("--query", required=True, help="Path to the query image")
    parser.add_argument("--views", nargs='+', required=True, help="Paths to the view images")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    infer = SOMinfer(args.config, args.ckpt)
    mask, pt = infer.predict(args.query, args.views, threshold=args.threshold)

    os.makedirs("output", exist_ok=True)
    mask_fp = os.path.join("output", os.path.basename(args.query).replace(".png","_mask.png"))
    mask.save(mask_fp)
    print(f"Saved mask to {mask_fp}")
    if pt:
        print("Prompt point (y, x):", pt)
    else:
        print("No valid mask pixels found, no prompt point generated.")
