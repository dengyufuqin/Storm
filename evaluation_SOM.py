#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.distance import pdist
import nvdiffrast.torch as dr

from FoundationPose.Utils import set_seed, set_logging_format, code_dir
from FoundationPose.datareader import BOP_LIST, get_bop_video_dirs, get_bop_reader
from FoundationPose.estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from pathlib import Path
# from cnos.cnos_mask import CNOSMask


def save_bop_scene_pred(preds: dict, out_scene_dir: Path):
    """
    Save scene predictions in BOP format.

    preds: dict[str or int, list of dict]{"obj_id": int, "cam_R_m2c": list, "cam_t_m2c": list}
    out_scene_dir: Path to folder where scene_pred.json will be written.
    """
    out_scene_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_scene_dir / "scene_pred.json"

    # 只保留非空的预测列表
    preds_filtered = {img_id: lst for img_id, lst in preds.items() if len(lst) > 0}

    with open(out_path, "w") as f:
        json.dump(preds_filtered, f, indent=2)
    print(f"Saved predictions to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Save FoundationPose predictions in BOP Challenge format"
    )
    parser.add_argument('--bop_dir', type=str,
                        required=True,
                        help='BOP dataset root directory')
    parser.add_argument('--mask_source', choices=['gt','som', 'cnos'], default='som',
                        help='Mask source: gt or som')
    parser.add_argument('--cnos_json', type=str, default="cnos/datasets/bop23_challenge/results/cnos_exps/FastSAM_template_pyrender0_aggavg_5_lmo.json",
                        metavar='JSON',
                    help='Path to CNOS inference JSON (required if mask_source=cnos)')
    parser.add_argument('--mask_type', choices=['mask_visib','mask'], default='mask_visib',
                        help='GT mask type, valid when mask_source=gt')
    parser.add_argument('--config', type=str, default="som/configs/train.yaml",
                        help='SOM config, valid when mask_source=som')
    parser.add_argument('--ckpt', type=str, default="som/checkpoints/best.pth",
                        help='SOM checkpoint, valid when mask_source=som')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='SOM confidence threshold, valid when mask_source=som')
    parser.add_argument('--sam_thresh', type=float, default=0.5,
                        help='SAM mask threshold, valid when mask_source=som')
    parser.add_argument('--est_refine_iter', type=int, default=5,
                        help='Number of refinement iterations')
    parser.add_argument('--debug_dir', type=str,
                        default=f'{code_dir}/../output/',
                        help='Output directory for predictions')
    args = parser.parse_args()

    if args.mask_source == 'som':
        from SOM import SOMinfer
        som = SOMinfer(config_path=args.config, ckpt_path=args.ckpt,
                        device='cuda')
    elif args.mask_source == 'cnos':
        if args.cnos_json is None:
            parser.error("When --mask_source=cnos, --cnos_json must be provided")
        # cm = CNOSMask(args.cnos_json)

    os.environ['BOP_DIR'] = args.bop_dir
    set_logging_format(logging.INFO)
    set_seed(0)
    logging.info("Starting to save predictions in BOP format")

    # Clear and create debug_dir
    os.system(f"rm -rf {args.debug_dir}/* && mkdir -p {args.debug_dir}")

    for ds in BOP_LIST:
        for video_dir in get_bop_video_dirs(ds):
            reader = get_bop_reader(video_dir, zfar=np.inf)
            inst_ids = reader.get_instance_ids_in_image(0)

            # Collect predictions for this scene
            preds = {}

            for ob_id in inst_ids:
                mesh = reader.get_gt_mesh(ob_id)
                sample_pts, _ = trimesh.sample.sample_surface(mesh, 2000)
                verts = sample_pts if sample_pts is not None else mesh.vertices

                scorer = ScorePredictor()
                refiner = PoseRefinePredictor()
                glctx = dr.RasterizeCudaContext()
                est = FoundationPose(model_pts=verts,
                                     model_normals=mesh.vertex_normals,
                                     symmetry_tfs=getattr(reader,'symmetry_tfs',{}).get(ob_id),
                                     mesh=mesh, scorer=scorer,
                                     refiner=refiner, glctx=glctx,
                                     debug=0, debug_dir=args.debug_dir)

                for i, imgp in enumerate(reader.color_files):
                    mask = None  # Initialize mask to None
                    if args.mask_source == 'gt':
                        mask = reader.get_mask(i, ob_id, type=args.mask_type)
                    elif args.mask_source == 'som':
                        imgp = reader.color_files[i]
                        rd = os.path.join(
                            args.bop_dir,
                            ds,
                            'render',
                            f'obj_{int(ob_id):06d}',
                            'nature'
                        )
                        views = [rd] if os.path.isdir(rd) else []
                        bm = som.predict_mask(imgp, views)
                        mask = np.array(bm).astype(bool)

                    elif args.mask_source == 'cnos':
                        scene_id = int(Path(video_dir).name)
                        image_id = int(Path(imgp).stem)
                        try:
                            mask_arr = cm.get_mask(scene_id, image_id, ob_id)
                            mask = mask_arr.astype(bool)
                        except KeyError:
                            logging.warning(
                                f"No CNOS mask for (scene={scene_id}, image={image_id}, obj={ob_id}), using full-image mask"
                            )
                            # fallback to full-image mask
                            color = reader.get_color(i)
                            h, w = color.shape[:2]
                            mask = np.ones((h, w), dtype=bool)

                    if mask is None or not mask.any():
                        continue
                    color = reader.get_color(i)
                    depth = reader.get_depth(i)
                    K = reader.get_K(i)

                    pose_est = est.register(K=K, rgb=color, depth=depth,
                                             ob_mask=mask.astype(bool), ob_id=ob_id,
                                             iteration=args.est_refine_iter)
                    # Store prediction
                    img_key = str(int(Path(imgp).stem))
                    t_m2c_mm = (pose_est[:3, 3] * 1000.0).reshape(-1)
                    preds.setdefault(img_key, []).append({
                        "cam_R_m2c": pose_est[:3, :3].reshape(-1).tolist(),
                        "cam_t_m2c": t_m2c_mm.tolist(),
                        "obj_id": int(ob_id)
                    })

            # Save predictions for this scene
            scene_id = Path(video_dir).name
            out_scene = Path(args.debug_dir) / ds / scene_id
            save_bop_scene_pred(preds, out_scene)

if __name__ == '__main__':
    main()
