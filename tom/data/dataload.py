import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class PairDataset(Dataset):
    def __init__(self, jsonl_file, transform=None):
        """
        Args:
            jsonl_file (str): Path to the JSONL file containing img1, img2, label per line.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.pairs = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.pairs.append((item['img1'], item['img2'], item['label']))

        # 默认转换：Resize->ToTensor->Normalize
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, label


def main():
    jsonl_path = '' 
    dataset = PairDataset(jsonl_path)
    print(f"Dataset size: {len(dataset)} samples")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    for batch_idx, (img1_batch, img2_batch, labels) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print(" img1 batch shape:", img1_batch.shape)
        print(" img2 batch shape:", img2_batch.shape)
        print(" labels shape:", labels.shape)
        assert img1_batch.ndim == 4 and img2_batch.ndim == 4, "Image batch tensor should have 4 dimensions"
        assert labels.ndim == 1, "Labels tensor should have 1 dimension"
        print("Basic test passed!\n")
        break  

if __name__ == '__main__':
    main()
