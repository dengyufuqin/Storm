import os
import json
import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.distance import cdist


def compute_diameter(mesh: trimesh.Trimesh) -> float:
    """
    Compute model diameter as maximum pairwise vertex distance.
    """
    verts = mesh.vertices
    if verts.shape[0] > 2000:
        idx = np.random.choice(verts.shape[0], 2000, replace=False)
        verts = verts[idx]
    dists = cdist(verts, verts)
    return float(np.max(dists))


def compute_add(vertices: np.ndarray, pose_est: np.ndarray, pose_gt: np.ndarray) -> float:
    """
    Compute ADD: mean Euclidean distance between corresponding model points.
    """
    pts_est = (pose_est[:3, :3] @ vertices.T + pose_est[:3, 3:4]).T
    pts_gt  = (pose_gt[:3, :3] @ vertices.T + pose_gt[:3, 3:4]).T
    return float(np.mean(np.linalg.norm(pts_est - pts_gt, axis=1)))


def compute_adds(vertices: np.ndarray, pose_est: np.ndarray, pose_gt: np.ndarray) -> float:
    """
    Compute ADD-S: mean nearest neighbor distance between model points.
    """
    pts_est = (pose_est[:3, :3] @ vertices.T + pose_est[:3, 3:4]).T
    pts_gt  = (pose_gt[:3, :3] @ vertices.T + pose_gt[:3, 3:4]).T
    dist_matrix = cdist(pts_est, pts_gt)
    return float(np.mean(np.min(dist_matrix, axis=1)))


def load_json(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def unwrap_pred(pred):
    """
    If pred is a list with one dict or a dict with one nested dict, unwrap one level.
    """
    if isinstance(pred, list) and len(pred) == 1 and isinstance(pred[0], dict):
        return pred[0]
    if isinstance(pred, dict) and len(pred) == 1:
        only_val = next(iter(pred.values()))
        if isinstance(only_val, dict):
            return only_val
    return pred

def compute_auc(recalls: np.ndarray, thresholds: np.ndarray) -> float:
    """
    Compute normalized AUC over thresholds: area under recall–threshold curve,
    normalized by the maximum threshold.
    """
    # Calculate the area under the recalls curve over the interval from 0 to thresholds[-1]
    # and divide by thresholds[-1]
    return float(np.trapz(recalls, thresholds) / thresholds[-1])
    
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ADD and ADD-S recall for BOP predictions"
    )
    parser.add_argument('--bop_dir', type=str, default="CMTdemo/dataset/data_test_all",
                        help='Path to BOP dataset root')
    parser.add_argument('--pred_dir', type=str, default="output/gt_fp_tracking_lost/",
                        help='Path to predictions root (matching BOP datasets)')
    parser.add_argument('--max_ratio', type=float, default=0.1,
                        help='Relative error threshold for recall (e.g. 0.1)')
    args = parser.parse_args()

    bop_dir = Path(args.bop_dir)
    pred_dir = Path(args.pred_dir)

    datasets = [d.name for d in pred_dir.iterdir() if d.is_dir()]
    stats = {}

    for ds in datasets:
        stats[ds] = {}
        ds_gt = bop_dir / ds
        ds_pred = pred_dir / ds

        for scene in sorted(ds_pred.iterdir()):
            if not scene.is_dir():
                continue
            gt_path = ds_gt / 'test' / scene.name / 'scene_gt.json'
            pred_path = scene / 'scene_pred.json'
            if not gt_path.exists() or not pred_path.exists():
                continue

            scene_gt = load_json(gt_path)
            raw_pred = load_json(pred_path)
            scene_pred = unwrap_pred(raw_pred)

            for img_id, pred_list in scene_pred.items():
                gt_map = {}
                for inst in scene_gt.get(img_id, []):
                    gt_map.setdefault(inst['obj_id'], []).append(inst)

                for inst_pred in pred_list:
                    obj_id = inst_pred['obj_id']
                    if obj_id not in gt_map:
                        continue

                    est = np.eye(4)
                    est[:3, :3] = np.array(inst_pred['cam_R_m2c']).reshape(3,3)
                    est[:3, 3] = np.array(inst_pred['cam_t_m2c'])

                    gt_inst = gt_map[obj_id][0]
                    gt = np.eye(4)
                    gt[:3, :3] = np.array(gt_inst['cam_R_m2c']).reshape(3,3)
                    gt[:3, 3] = np.array(gt_inst['cam_t_m2c'])

                    if obj_id not in stats[ds]:
                        mesh_path = ds_gt / 'models' / f'obj_{int(obj_id):06d}.ply'
                        mesh = trimesh.load(mesh_path, process=False)
                        diam = compute_diameter(mesh)
                        # -- Use all vertices to compute ADD/ADD-S --
                        verts = mesh.vertices
                        stats[ds][obj_id] = {'add': [], 'adds': [], 'diam': diam, 'verts': verts}

                    verts = stats[ds][obj_id]['verts']
                    stats[ds][obj_id]['add'].append(compute_add(verts, est, gt))
                    stats[ds][obj_id]['adds'].append(compute_adds(verts, est, gt))

    print('| ds | obj_id | ADD_rec | ADD-S_rec |   AR   | samples |')
    print('|---|---|---|---|---|---|')
    for ds, objs in stats.items():
        all_add = []
        all_adds = []
        all_ar = []

        for obj_id, data in sorted(objs.items(), key=lambda x: int(x[0])):
            errs_add  = np.array(data['add'])
            errs_adds = np.array(data['adds'])
            n = errs_add.size
            if n == 0:
                continue
            diam = data['diam']

            # Single-threshold recall (max_ratio × diameter)
            thr = args.max_ratio * diam
            recall_add  = float((errs_add  <= thr).mean())
            recall_adds = float((errs_adds <= thr).mean())

            # AR = AUC over [0, max_ratio]
            t_vec = np.linspace(0, args.max_ratio, 100)
            rec1 = [(errs_add  / diam <= t).mean() for t in t_vec]
            rec2 = [(errs_adds / diam <= t).mean() for t in t_vec]
            auc_add  = compute_auc(np.array(rec1), t_vec)
            auc_adds = compute_auc(np.array(rec2), t_vec)
            # AR as average of ADD and ADD-S AUC
            ar = (auc_add + auc_adds) / 2

            print(f'| {ds} | {obj_id} | {recall_add:.4f} | {recall_adds:.4f} | {ar:.4f} |   {n}    |')

            all_add.append(recall_add)
            all_adds.append(recall_adds)
            all_ar.append(ar)

        # Dataset average
        if all_add:
            mean_add  = float(np.mean(all_add))
            mean_adds = float(np.mean(all_adds))
            mean_ar   = float(np.mean(all_ar))
            print(f'| {ds} | mean   | {mean_add:.4f} | {mean_adds:.4f} | {mean_ar:.4f} |   -    |')

if __name__ == '__main__':
    main()
