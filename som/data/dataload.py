import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode,
    ColorJitter, RandomAffine, GaussianBlur, RandomErasing, RandomHorizontalFlip,
)
import torchvision.transforms.functional as F
import glob


class JointGeometricTransform:
    def __init__(self, degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), p_flip=0.5, shear=(25, 25)):
        self.affine = RandomAffine(degrees=degrees, translate=translate, shear=shear, scale=scale)
        self.p_flip = p_flip

    def __call__(self, sample):
        query_img, mask = sample['query_img'], sample['mask']

        # Random horizontal flip
        if torch.rand(1) < self.p_flip:
            query_img = F.hflip(query_img)
            mask = F.hflip(mask)

        # Random affine
        affine_params = self.affine.get_params(
            self.affine.degrees, self.affine.translate,
            self.affine.scale, self.affine.shear, query_img.size
        )
        query_img = F.affine(query_img, *affine_params, interpolation=InterpolationMode.BICUBIC)
        mask = F.affine(mask, *affine_params, interpolation=InterpolationMode.NEAREST)

        sample['query_img'] = query_img
        sample['mask'] = mask
        return sample

class ApplyToQuery:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, sample):
        sample['query_img'] = self.transform(sample['query_img'])
        return sample

class ApplyToMask:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, sample):
        sample['mask'] = self.transform(sample['mask'])
        return sample

class ToTensorAndNormalize:
    def __init__(self, mean, std):
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=mean, std=std)

    def __call__(self, sample):
        sample['query_img'] = self.normalize(self.to_tensor(sample['query_img']))
        sample['mask'] = self.to_tensor(sample['mask'])
        return sample

class RandomEraseBoth:
    def __init__(self, p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'):
        self.eraser = RandomErasing(p=p, scale=scale, ratio=ratio, value=value)

    def __call__(self, sample):
        if torch.rand(1) < self.eraser.p:
            fill_value = self.eraser.value
            if fill_value == 'random':
                fill_value = None
            
            erase_params = self.eraser.get_params(
                sample['query_img'], self.eraser.scale, self.eraser.ratio, value=fill_value
            )
            sample['query_img'] = F.erase(sample['query_img'], *erase_params)
            sample['mask'] = F.erase(sample['mask'], *erase_params[:-1], v=0)
        return sample


HB_PRESENT = {
    3:  [4, 9, 19, 29],
    5:  [3, 12, 15, 22, 23],
    13: [1, 4, 8, 10, 17, 18, 32, 33],
}

def find_with_ext(directory: str, base_name: str, exts=("png", "jpg", "jpeg")) -> str:
    for ext in exts:
        path = os.path.join(directory, f"{base_name}.{ext}")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"File {base_name} not found in {directory} with exts {exts}")


class MultiYCBVDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_size: int = 512,
        num_views: int = 16,
        use_augment: bool = False,
        include_datasets: list[str] = ["lmo", "hb", "tless", "tudl", "ycbv"] ,
    ):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.num_views = num_views
        self.samples = []
        self.use_augment = use_augment
        self.include_datasets = include_datasets if include_datasets is not None else []
        
        splits_to_load = [self.split]
        
        # transforms for DINOv2
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        # Base transform (resize/crop, applied after augmentation)
        base_transform = Compose([
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
        ])
        mask_base_transform = Compose([
            Resize(image_size, interpolation=InterpolationMode.NEAREST),
            CenterCrop(image_size),
        ])

        # Augmentation transforms (applied only on training)
        self.augment_transform = None
        if self.use_augment and self.split == 'train':
            photometric_augment = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            blur_augment = GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))
            
            self.augment_transform = Compose([
                ApplyToQuery(photometric_augment),
                ApplyToQuery(lambda x: blur_augment(x) if torch.rand(1) < 0.5 else x),
                JointGeometricTransform(degrees=180, translate=(0, 0), scale=(0.95, 1.05), shear=(25, 25)),
            ])

        # Base and final transforms
        self.base_transform = Compose([
            ApplyToQuery(base_transform),
            ApplyToMask(mask_base_transform),
        ])

        self.to_tensor_transform = ToTensorAndNormalize(mean=mean, std=std)
        
        self.occlusion_transform = None
        if self.use_augment and self.split == 'train':
            self.occlusion_transform = RandomEraseBoth(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random')


        # Transform to convert to tensor and normalize
        self.final_view_transform = Compose([
            base_transform,
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

        # collect dataset folders
        for split_to_load in splits_to_load:
            for ds in sorted(os.listdir(self.root)):
                ds_path = os.path.join(self.root, ds)
                if not os.path.isdir(ds_path):
                    continue
                if self.include_datasets and ds not in self.include_datasets:
                    continue
                # load descriptions mapping
                desc_path = os.path.join(ds_path, 'descriptions.json')
                if not os.path.exists(desc_path):
                    desc_path = os.path.join(ds_path, 'description.json')
                if os.path.exists(desc_path):
                    with open(desc_path, 'r') as f:
                        desc_map = json.load(f)
                else:
                    desc_map = {}

                if split_to_load == 'train':
                    source_dir_name = 'train'
                else:
                    source_dir_name = 'test_bop19'

                source_dir = os.path.join(ds_path, source_dir_name)
                render_dir = os.path.join(ds_path, 'render')
                if not os.path.isdir(source_dir):
                    continue

                if 'hb' in ds.lower() and split_to_load == 'test':
                    source_root = source_dir
                    render_dir  = os.path.join(ds_path, 'render')
                    if not os.path.isdir(source_root) or not os.path.isdir(render_dir):
                        continue
                
                    for scene in sorted(os.listdir(source_root)):
                        scene_id = int(scene)
                        video_dir = os.path.join(source_root, scene)
                        rgb_dir   = os.path.join(video_dir, 'rgb')
                        mask_dir  = os.path.join(video_dir, 'mask_visib')
                        if not os.path.isdir(rgb_dir) or not os.path.isdir(mask_dir):
                            continue

                        if scene_id not in HB_PRESENT:
                            print(f"[WARN] HB scene {scene_id} not in HB_PRESENT, skip or custom handle.")
                            continue
                        obj_order = HB_PRESENT[scene_id]

                        for img_file in sorted(os.listdir(rgb_dir)):
                            base, _ = os.path.splitext(img_file)  # e.g. "000123"
                            mask_files = glob.glob(f"{mask_dir}/{base}_*.png")
                            mask_files.sort()
                            
                            for inst_idx, mask_path in enumerate(mask_files):
                                obj_id = obj_order[inst_idx]
                                desc_key    = f"obj_{obj_id:06d}"
                                description = desc_map.get(desc_key, "")
                                render_base = os.path.join(render_dir, f"obj_{obj_id:06d}", 'nature')
                                self.samples.append({
                                    'rgb_dir': rgb_dir,
                                    'mask_dir': mask_dir,
                                    'render_base': render_base,
                                    'base': base,
                                    'inst_idx': inst_idx,
                                    'description': description,
                                    'obj_id': obj_id,
                                    'mask_path': mask_path,
                                })
                else:
                    # iterate scenes
                    for scene in sorted(os.listdir(source_dir)):
                        scene_path = os.path.join(source_dir, scene)
                        gt_file = os.path.join(scene_path, 'scene_gt.json')
                        if not os.path.exists(gt_file):
                            continue
                        with open(gt_file, 'r') as f:
                            scene_gt = json.load(f)

                        rgb_dir  = os.path.join(scene_path, 'rgb')
                        mask_dir = os.path.join(scene_path, 'mask_visib')

                        for img_key, entries in sorted(scene_gt.items(), key=lambda x: int(x[0])):
                            base = f"{int(img_key):06d}"
                            for inst_idx, entry in enumerate(entries):
                                obj_id = entry['obj_id']
                                desc_key = f"obj_{obj_id:06d}"
                                description = desc_map.get(desc_key, "")
                                render_base = os.path.join(
                                    render_dir, f"obj_{obj_id:06d}", 'nature'
                                )
                                self.samples.append({
                                    'rgb_dir': rgb_dir,
                                    'mask_dir': mask_dir,
                                    'render_base': render_base,
                                    'base': base,
                                    'inst_idx': inst_idx,
                                    'description': description,
                                })

    def __len__(self):
        return len(self.samples)    

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        # load query
        img_path = find_with_ext(s['rgb_dir'], s['base'])
        query_img = Image.open(img_path).convert('RGB')

        # load mask
        mask_name = f"{s['base']}_{s['inst_idx']:06d}"
        mask_path = find_with_ext(s['mask_dir'], mask_name)
        mask = Image.open(mask_path).convert('L')

        sample = {'query_img': query_img, 'mask': mask}

        # Apply augmentations if enabled for training
        if self.augment_transform:
            sample = self.augment_transform(sample)

        # Apply base transforms (resize/crop)
        sample = self.base_transform(sample)

        # Convert to tensor and normalize
        sample = self.to_tensor_transform(sample)
        query_img = sample['query_img']
        mask_gt = sample['mask']

        # Apply occlusion augmentation after converting to tensor
        if self.occlusion_transform:
            sample = self.occlusion_transform(sample)
            query_img = sample['query_img']
            mask_gt = sample['mask']


        # load views (no augmentation)
        views = []
        for v in range(self.num_views):
            view_name = f"angle_{v:03d}"
            vp = find_with_ext(s['render_base'], view_name)
            vi = Image.open(vp).convert('RGB')
            # Apply the full original transform to views
            view_tensor = self.final_view_transform(vi)
            views.append(view_tensor)
        view_imgs = torch.stack(views, dim=0)

        return {
            'query_img': query_img,
            'mask_gt':   mask_gt,
            'view_imgs': view_imgs,
            'description': s['description'],
            'image_id': f"{os.path.basename(os.path.dirname(s['rgb_dir']))}/{s['base']}",
        }


# Test loader
if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Test MultiYCBVDataset loader")
    parser.add_argument("--root", type=str, default="./datasets/bop_datasets", help="数据根目录，包含多个子数据集文件夹")
    parser.add_argument("--split", type=str, default="test", choices=['train', 'test'], help="加载 'train' 或 'test' 数据集")
    parser.add_argument("--image_size", type=int, default=256, help="输入图像大小")
    parser.add_argument("--batch_size", type=int, default=2, help="测试时 batch size")
    parser.add_argument("--use_augment", action='store_true', help="是否使用数据增强")
    args = parser.parse_args()

    dataset = MultiYCBVDataset(
        root=args.root, 
        split=args.split, 
        image_size=args.image_size,
        use_augment=args.use_augment,
        include_datasets=['ycbv'] # Example: only load ycbv
    )
    print(f"Loaded MultiYCBV dataset (split: {args.split}, augment: {args.use_augment}), total samples: {len(dataset)}")

    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        batch = next(iter(loader))
        print("Batch keys:", batch.keys())
        print(" query_img:", batch['query_img'].shape)
        print(" mask_gt:  ", batch['mask_gt'].shape)
        print(" view_imgs:", batch['view_imgs'].shape)
        print(" description:", batch['description'][0])  # example description

        # To visualize the augmentation, you can save the images
        from torchvision.utils import save_image
        print("Saving augmented sample to `sample_aug.png` and `sample_mask.png`")
        
        save_image(batch['query_img'][0], 'sample_aug.png', normalize=True)
        save_image(batch['mask_gt'][0], 'sample_mask.png')
        save_image(batch['view_imgs'][0], 'sample_view.png', normalize=True)

    else:
        print("Dataset is empty. Check data paths and split configuration.")
