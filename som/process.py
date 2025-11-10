import os
import json
import shutil

# Paths
to_process = {
    'train': os.path.join('.', 'dataset', 'ycbv', 'train_pbr'),
    'test':  os.path.join('.', 'dataset', 'ycbv', 'test_bop', 'test'),
}
base_out = os.path.join('.', 'dataset', 'ycbv')
exts = ['.png', '.jpg']

def find_with_ext(directory, basename, exts):
    """Return path to file with one of the given extensions."""
    for e in exts:
        path = os.path.join(directory, basename + e)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No file found for {basename} in {directory} with exts {exts}")

for split, source_dir in to_process.items():
    train_dir = os.path.join(base_out, split)
    img_out = os.path.join(train_dir, 'img')
    mask_out = os.path.join(train_dir, 'mask')
    ann_path = os.path.join(train_dir, 'annotations.json')

    # Create output directories
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    annotations = []
    global_idx = 0

    # Iterate through each scene folder
    for scene in sorted(os.listdir(source_dir)):
        scene_path = os.path.join(source_dir, scene)
        if not os.path.isdir(scene_path):
            continue

        # Load scene ground truth
        gt_file = os.path.join(scene_path, 'scene_gt.json')
        with open(gt_file, 'r') as f:
            scene_gt = json.load(f)

        rgb_dir  = os.path.join(scene_path, 'rgb')
        mask_dir = os.path.join(scene_path, 'mask_visib')

        # For each image index in this scene
        for img_key in sorted(scene_gt, key=lambda x: int(x)):
            img_idx = int(img_key)
            basename = f"{img_idx:06d}"
            rgb_src = find_with_ext(rgb_dir, basename, exts)

            # For each object instance in this image
            for inst_idx, entry in enumerate(scene_gt[img_key]):
                obj_id = entry['obj_id']
                mask_base = f"{basename}_{inst_idx:06d}"
                mask_src = find_with_ext(mask_dir, mask_base, exts)

                # Determine extension and destination filenames
                _, img_ext  = os.path.splitext(rgb_src)
                _, mask_ext = os.path.splitext(mask_src)
                dst_img  = os.path.join(img_out,  f"{global_idx:06d}{img_ext}")
                dst_mask = os.path.join(mask_out, f"{global_idx:06d}{mask_ext}")

                # Copy files
                shutil.copyfile(rgb_src,  dst_img)
                shutil.copyfile(mask_src, dst_mask)

                # Record annotation (index, obj_id)
                annotations.append([global_idx, obj_id])
                global_idx += 1

    # Save annotations
    with open(ann_path, 'w') as f:
        json.dump(annotations, f, indent=4)

    print(f"[{split}] Processed {global_idx} instances. Annotations saved to {ann_path}")
