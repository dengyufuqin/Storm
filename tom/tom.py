import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import logging
import numpy as np

from .model.modelDino import DINOv2SelfAttentionModel


def get_default_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class TOM:

    def __init__(
        self,
        checkpoint_path: str = "./tom/checkpoints/tom1_epoch10.pth",
        img_size: int = 224,
        threshold: float = 0.9,
        device: torch.device = None,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.checkpoint_path = checkpoint_path
        self.img_size = img_size
        self.threshold = threshold
        self.transform = get_default_transform(img_size)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        self.model = self._load_model()

    def _load_model(self) -> DINOv2SelfAttentionModel:
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        model = DINOv2SelfAttentionModel(num_layers=1)
        state = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        logging.info(f"Model loaded from {self.checkpoint_path} on {self.device}")
        return model

    def _load_image(self, img_path: str) -> torch.Tensor:
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert('RGB')
        return self.transform(img).unsqueeze(0).to(self.device)

    def predict(self, img1_path: str, img2_path: str) -> float:
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)

        with torch.no_grad():
            start = time.time()
            logits = self.model(img1, img2)
            prob = torch.sigmoid(logits).item()
            elapsed = time.time() - start
        logging.info(f"Predicted={prob:.4f} in {elapsed*1000:.2f} ms")
        return prob
    def predict_from_arrays(self, img1: np.ndarray, img2: np.ndarray) -> float:
        pil1 = Image.fromarray(img1)
        pil2 = Image.fromarray(img2)

        tensor1 = self.transform(pil1).unsqueeze(0).to(self.device)
        tensor2 = self.transform(pil2).unsqueeze(0).to(self.device)

        with torch.no_grad():
            #start = time.time()
            logits = self.model(tensor1, tensor2)
            prob   = torch.sigmoid(logits).item()
            #elapsed = time.time() - start

        #logging.info(f"Predicted={prob:.4f} in {elapsed*1000:.2f} ms")
        return prob

    def predict_batch(self, pairs: list) -> list:
        results = []
        imgs1, imgs2 = [], []
        for p in pairs:
            imgs1.append(self._load_image(p[0]))
            imgs2.append(self._load_image(p[1]))
        
        batch1 = torch.cat(imgs1, dim=0)
        batch2 = torch.cat(imgs2, dim=0)

        with torch.no_grad():
            start = time.time()
            logits = self.model(batch1, batch2)
            probs = torch.sigmoid(logits).squeeze().tolist()
            elapsed = time.time() - start
        logging.info(f"Batch predicted {len(pairs)} pairs in {elapsed*1000:.2f} ms")
        return probs

    def is_same(self, img1_path: str, img2_path: str) -> bool:
        prob = self.predict(img1_path, img2_path)
        return prob >= self.threshold
