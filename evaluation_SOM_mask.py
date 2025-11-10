import os
import time
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from som.data.dataload import MultiYCBVDataset, find_with_ext
from SOM import SOMinfer

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_bin = (pred_mask > 127).astype(bool)
    gt_bin   = (gt_mask   > 127).astype(bool)
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union

def calculate_ap(iou: float) -> float:
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    precisions = [(1.0 if iou >= t else 0.0) for t in iou_thresholds]
    return np.mean(precisions)

def main():
    parser = argparse.ArgumentParser(description="Evaluate SOMinfer model performance")
    parser.add_argument("--data_root",   type=str, required=True, help="Root directory containing BOP datasets")
    parser.add_argument("--output_root", type=str, default="./output_masks", help="Root directory to save predicted masks")
    parser.add_argument("--config", default="som/configs/train.yaml", help="Path to model configuration file")
    parser.add_argument("--ckpt", default="som/checkpoints/best.pth", help="Path to model checkpoint file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for mask binarization")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on: 'cuda' or 'cpu'")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    log_path = os.path.join(args.output_root, "evaluation_som_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    print(f"Loading model: {args.ckpt}")
    segmentor = SOMinfer(config_path=args.config, ckpt_path=args.ckpt, device=args.device)
    
    image_size = segmentor.img_size
    print(f"Loading dataset: {args.data_root}")
    dataset = MultiYCBVDataset(root=args.data_root, split="test", image_size=image_size)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    all_ious = []
    all_times = []
    per_ds_metrics = {}  

    print(f"Starting evaluation for {len(dataset)} samples...")
    for idx, batch in enumerate(loader):
        sample = dataset.samples[idx]
        
        rel_rgb = os.path.relpath(sample['rgb_dir'], args.data_root)
        ds_name = rel_rgb.split(os.sep, 1)[0]

        if ds_name not in per_ds_metrics:
            per_ds_metrics[ds_name] = {"ious": [], "times": [], "aps": []}
        
        try:
            image_path = find_with_ext(sample['rgb_dir'], sample['base'])
        except FileNotFoundError:
            print(f"Warning: Image not found {sample['rgb_dir']}/{sample['base']}, skipping this sample.")
            continue

        view_paths = [sample['render_base']]

        start_time = time.time()
        pred_mask_pil = segmentor.predict_mask(image_path, view_paths, threshold=args.threshold)
        elapsed = time.time() - start_time
        
        ds_output_dir = os.path.join(args.output_root, ds_name)
        os.makedirs(ds_output_dir, exist_ok=True)
        mask_filename = f"{sample['base']}_{sample['inst_idx']:06d}_pred.png"
        mask_save_path = os.path.join(ds_output_dir, mask_filename)
        pred_mask_pil.save(mask_save_path)

        gt_mask_tensor = batch['mask_gt'][0]
        gt_mask_np = (gt_mask_tensor.squeeze().numpy() * 255).astype(np.uint8)

        pred_mask_np = np.array(pred_mask_pil.resize((image_size, image_size), resample=Image.NEAREST))

        iou = compute_iou(pred_mask_np, gt_mask_np)
        ap = calculate_ap(iou) # Per-object AP_O

        all_ious.append(iou)
        all_times.append(elapsed)
        
        per_ds_metrics[ds_name]["ious"].append(iou)
        per_ds_metrics[ds_name]["times"].append(elapsed)
        per_ds_metrics[ds_name]["aps"].append(ap)

        log_msg = f"[{idx+1}/{len(dataset)}] {ds_name}/{sample['base']}_{sample['inst_idx']:06d} | IoU: {iou:.4f} | AP_O: {ap:.4f} | Time: {elapsed:.3f}s\n"
        print(log_msg, end="")
        log_file.write(log_msg)
    
    # Calculate AP_D for each dataset
    ds_aps = []
    for ds_name, metrics in per_ds_metrics.items():
        if metrics["aps"]:
            ds_ap = np.mean(metrics["aps"])
            ds_aps.append(ds_ap)

    # Calculate global metrics
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    mean_time = np.mean(all_times) if all_times else 0.0
    # Core AP (AP_C) is the average of all dataset AP_Ds
    ap_c = np.mean(ds_aps) if ds_aps else 0.0

    summary = (
        "\n===== Global Evaluation Results =====\n"
        f"Mean IoU: {mean_iou:.4f}\n"
        f"Core AP (AP_C): {ap_c:.4f}\n"
        f"Mean Inference Time: {mean_time:.3f}s\n"
    )
    print(summary)
    log_file.write(summary + "\n")

    # Summarize metrics for each sub-dataset
    ds_summary_header = "\n===== Per-Dataset Evaluation Results =====\n"
    print(ds_summary_header.strip())
    log_file.write(ds_summary_header)
    for ds_name, metrics in sorted(per_ds_metrics.items()):
        if not metrics["ious"]: continue
        ds_iou = np.mean(metrics["ious"])
        ds_ap = np.mean(metrics["aps"]) # AP_D
        ds_time = np.mean(metrics["times"])
        line = f"[{ds_name}] Mean IoU: {ds_iou:.4f}, Dataset AP (AP_D): {ds_ap:.4f}, Mean Inference Time: {ds_time:.3f}s\n"
        print(line, end="")
        log_file.write(line)

    log_file.close()
    print(f"\nEvaluation complete. Log saved to: {log_path}")

if __name__ == "__main__":
    main()