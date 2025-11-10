import os
import torch
import logging
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import time


def get_default_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


class SimilarityModel:
    def __init__(self,
                 model_name: str = "dinov2_vits14",
                 img_size: int = 224,
                 threshold: float = 0.2,
                 device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.threshold = threshold
        self.transform = get_default_transform(img_size)
        self.model = self._load_backbone(model_name)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def _load_backbone(self, model_name: str):
        model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
        model.eval().to(self.device)
        logging.info(f"DINOv2 model {model_name} loaded on {self.device}")
        return model

    def _load_image(self, img_path: str) -> torch.Tensor:
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)  # [1, 3, H, W]

    def _cosine_similarity(self, f1: torch.Tensor, f2: torch.Tensor) -> float:
        f1 = F.normalize(f1, dim=-1)
        f2 = F.normalize(f2, dim=-1)
        return (f1 * f2).sum(dim=-1).item()

    def predict(self, img1_path: str, img2_path: str) -> float:
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)

        with torch.no_grad():
            feat1 = self.model(img1)  # [1, D]
            feat2 = self.model(img2)

        score = self._cosine_similarity(feat1, feat2)
        logging.info(f"Similarity: {score:.4f}")
        return score
    
    def predict_from_tensors(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        #print the time taken to move tensors to device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        with torch.no_grad():
            feat1 = self.model(img1)
            feat2 = self.model(img2)
        cos = self._cosine_similarity(feat1, feat2)
        
        # print the time in ms taken for prediction
        return cos

    def predict_from_arrays(self, img1: np.ndarray, img2: np.ndarray) -> float:
        pil1 = Image.fromarray(img1)
        pil2 = Image.fromarray(img2)

        tensor1 = self.transform(pil1).unsqueeze(0).to(self.device)
        tensor2 = self.transform(pil2).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat1 = self.model(tensor1)
            feat2 = self.model(tensor2)

        score = self._cosine_similarity(feat1, feat2)
        return score

    def is_same(self, img1_path: str, img2_path: str) -> bool:
        score = self.predict(img1_path, img2_path)
        return score >= self.threshold

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test feature similarity')
    parser.add_argument('--img1', type=str, default='',
                        help='Path to the first image')
    parser.add_argument('--img2', type=str, default='',
                        help='Path to the second image')
    args = parser.parse_args()

    Sim = SimilarityModel()
    score = Sim.predict(args.img1, args.img2)
    print(f"Cosine similarity: {score:.4f}")
    if Sim.is_same(args.img1, args.img2):
        print("Images are the same")
    else:
        print("Images are different")