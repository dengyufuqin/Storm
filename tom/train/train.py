# train.py

import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataload import PairDataset
from model.modelDino import DINOv2SelfAttentionModel

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    processed_samples = 0

    loop = tqdm(loader, desc=f"Epoch [{epoch}/{num_epochs}]", leave=False)
    for imgs1, imgs2, labels in loop:
        imgs1 = imgs1.to(device)
        imgs2 = imgs2.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(imgs1, imgs2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = imgs1.size(0)
        running_loss += loss.item() * batch_size
        processed_samples += batch_size

        avg_loss_so_far = running_loss / processed_samples
        loop.set_postfix(loss=f"{avg_loss_so_far:.4f}")

    return running_loss / len(loader.dataset)

def main():
    jsonl_path   = "*******"
    batch_size   = 64
    lr           = 1e-4
    num_epochs   = 10
    num_workers  = 4
    device       = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    dataset = PairDataset(jsonl_path, transform=None)
    loader  = DataLoader(dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=num_workers,
                         pin_memory=True)

    model     = DINOv2SelfAttentionModel(num_layers = 2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs)
        print(f"Epoch {epoch}/{num_epochs} — Avg Loss: {avg_loss:.4f}")

        if epoch % 5 == 0 or epoch == num_epochs:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(),
                       f"checkpoints/tom2_epoch{epoch}.pth")

    print("Training finished. Models saved in ./checkpoints/")

if __name__ == "__main__":
    main()
