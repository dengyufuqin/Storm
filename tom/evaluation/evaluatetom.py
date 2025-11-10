# evaluation.py

import os
import argparse
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataload import PairDataset
from model.modelDino import DINOv2SelfAttentionModel
from torchvision import transforms

def get_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def load_model(checkpoint_path: str, device: torch.device) -> DINOv2SelfAttentionModel:
    model = DINOv2SelfAttentionModel(num_layers=1).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def evaluate(model: DINOv2SelfAttentionModel,
             loader: DataLoader,
             device: torch.device,
             threshold: float = 0.5):
    correct = 0
    total = 0
    inference_time = 0.0

    with torch.no_grad():
        loop = tqdm(loader, desc="Evaluating", leave=False)
        for imgs1, imgs2, labels in loop:
            imgs1 = imgs1.to(device)
            imgs2 = imgs2.to(device)
            labels = labels.to(device).float()

            
            t0 = time.time()
            logits = model(imgs1, imgs2)
            probs  = torch.sigmoid(logits)
            t1 = time.time()
            inference_time += (t1 - t0)

            preds  = (probs >= threshold).float()

            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            
            current_acc = correct / total * 100
            loop.set_postfix(acc=f"{current_acc:.2f}%")
            logging.info(f"Step {total}: Accuracy={current_acc:.2f}%, InferenceTimeSoFar={inference_time:.3f}s")

    accuracy = correct / total if total > 0 else 0.0
    avg_time = inference_time / total if total > 0 else 0.0
    return accuracy, avg_time

def main():
    parser = argparse.ArgumentParser(description="Evaluate Tom Model on Pair Dataset")
    parser.add_argument("--checkpoint", type=str,
                        default="",
                        help="Path to the saved model .pth file")
    parser.add_argument("--pairs-jsonl", type=str,
                        default="",
                        help="Path to pairs.jsonl file")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of DataLoader workers")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold on sigmoid(logit) to decide label")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Size to resize images (must match training)")
    args = parser.parse_args()


    
    log_path = ""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )

   
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.isfile(args.pairs_jsonl):
        raise FileNotFoundError(f"Pairs JSONL not found: {args.pairs_jsonl}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    transform = get_transform(args.img_size)
    dataset = PairDataset(args.pairs_jsonl, transform=transform)
    loader  = DataLoader(dataset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.num_workers,
                         pin_memory=True)

   
    model = load_model(args.checkpoint, device)

   
    accuracy, avg_time = evaluate(model, loader, device, threshold=args.threshold)
    logging.info("\nEvaluation complete.")
    logging.info(f"  → Average Accuracy      : {accuracy * 100:.2f}%")
    logging.info(f"  → Avg Inference Time    : {avg_time * 1000:.2f} ms per sample")

if __name__ == "__main__":
    main()
