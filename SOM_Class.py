import os
import random
import numpy as np
import torch
from PIL import Image

from SOM import SOMinfer
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SOM:
    def __init__(
        self,
        som_thresh: float = 0.5,
        sam_thresh: float = 0.5,
        sam_model: str = "facebook/sam2-hiera-tiny",
        device: str = 'cuda'
    ):
        self.device = device
        self.torch_device = torch.device(device)
        self.cmt_thresh = som_thresh
        self.sam_thresh = sam_thresh

        self.cmt = SOMinfer(device=device)

        self.predictor = SAM2ImagePredictor.from_pretrained(
            sam_model,
            device=device,
            torch_dtype=torch.bfloat16
        )

    def get_mask(self, image_path: str, view_paths: list[str]) -> np.ndarray:
        orig = Image.open(image_path).convert('RGB')
        img_np = np.array(orig)
        H, W = img_np.shape[:2]

        coarse_mask, pt = self.cmt.predict(
            query_path=image_path,
            view_paths=view_paths,
            threshold=self.cmt_thresh
        )

        if pt is None or coarse_mask is None:
            x = random.randint(0, W - 1)
            y = random.randint(0, H - 1)
            bw, bh = W // 2, H // 2
            x0 = random.randint(0, W - bw)
            y0 = random.randint(0, H - bh)
            x1, y1 = x0 + bw, y0 + bh
        else:
            y, x = pt
            mask_np = (coarse_mask.cpu().numpy() if torch.is_tensor(coarse_mask)
                       else np.array(coarse_mask))
            mask_bool = mask_np > self.cmt_thresh
            ys, xs = np.where(mask_bool)
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            pad = int(0.1 * max(y1 - y0, x1 - x0))
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(W, x1 + pad), min(H, y1 + pad)

        with torch.inference_mode(), torch.autocast(self.torch_device.type, dtype=torch.bfloat16):
            self.predictor.set_image(img_np)
            masks, _, _ = self.predictor.predict(
                point_coords=torch.tensor([[[x, y]]], device=self.torch_device),
                point_labels=torch.tensor([[1]], device=self.torch_device),
                box=torch.tensor([[[x0, y0, x1, y1]]], device=self.torch_device),
                multimask_output=False
            )
        mask = masks[0]
        mask_arr = mask.cpu().numpy() if torch.is_tensor(mask) else mask
        bin_mask = (mask_arr > self.sam_thresh).astype(np.uint8) * 255
        return bin_mask


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SIC: CMT + SAM2 细化分割工具"
    )
    parser.add_argument("--image", required=True,)
    parser.add_argument("--views", nargs='+', required=True,)
    parser.add_argument("--cmt_config", type=str,
                        default="som/configs/train.yaml",
                        help="CMTInfer 配置文件路径")
    parser.add_argument("--cmt_ckpt", type=str,
                        default="som/checkpoints/best.pth",
                        help="CMTInfer 权重文件路径")
    parser.add_argument("--som_thresh", type=float, default=0.5,
                        help="CMTInfer 掩码二值化阈值")
    parser.add_argument("--sam_thresh", type=float, default=0.5,
                        help="SAM2 掩码二值化阈值")
    parser.add_argument("--device", type=str, default="cuda",
                        help="运行设备：cuda 或 cpu")
    parser.add_argument("--output", type=str, default="./output",
                        help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    sic = SOM(
        som_thresh=args.som_thresh,
        sam_thresh=args.sam_thresh,
        device=args.device
    )
    mask = sic.get_mask(args.image, args.views)
    out_name = os.path.basename(args.image).replace('.png', '_mask.png')
    out_path = os.path.join(args.output, out_name)
    Image.fromarray(mask).save(out_path)
    print(f"Saved refined mask to: {out_path}")