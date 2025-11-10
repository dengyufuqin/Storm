import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import math
import numpy as np
import trimesh
import pyrender
import cv2

def look_at(camera_position, target, up=np.array([0, 0, 1])):
    forward = target - camera_position
    forward /= np.linalg.norm(forward)
    if abs(np.dot(forward, up)) > 0.999:
        up = np.array([0, 1, 0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    R = np.array([
        [ right[0],  true_up[0], -forward[0]],
        [ right[1],  true_up[1], -forward[1]],
        [ right[2],  true_up[2], -forward[2]]
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = camera_position
    return T

N = 16  
golden_angle = math.pi * (3.0 - math.sqrt(5.0))  

cwd = "****************"  # Replace with your actual path

for ds in os.listdir(cwd):
    ds_dir = os.path.join(cwd, ds)
    if not os.path.isdir(ds_dir):
        continue

    models_dir = os.path.join(ds_dir, "model")
    if not os.path.isdir(models_dir):
        continue

    renders_dir = os.path.join(ds_dir, "render")
    os.makedirs(renders_dir, exist_ok=True)

    for fname in os.listdir(models_dir):
        if not fname.lower().endswith('.ply'):
            continue
        base = os.path.splitext(fname)[0]  # e.g. obj_000001
        ply_path = os.path.join(models_dir, fname)

        model_out  = os.path.join(renders_dir, base)
        nature_dir = os.path.join(model_out, 'nature')
        mask_dir   = os.path.join(model_out, 'mask')
        for d in (nature_dir, mask_dir):
            os.makedirs(d, exist_ok=True)

        print(f"[{ds}] load model: {ply_path}")
        scene_data = trimesh.load(ply_path, force='scene')
        meshes = (list(scene_data.geometry.values())
                  if isinstance(scene_data, trimesh.Scene)
                  else [scene_data])

        verts = np.vstack([m.vertices for m in meshes])
        center = verts.mean(axis=0)
        diag = np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))
        cam_dist = diag * 1.5 if diag > 0 else 10.0

        normal_meshes = [pyrender.Mesh.from_trimesh(m, smooth=True)
                         for m in meshes]
        mask_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1,1,1,1],
            metallicFactor=0,
            roughnessFactor=1,
            emissiveFactor=[1,1,1]
        )
        mask_meshes = [pyrender.Mesh.from_trimesh(m, material=mask_mat, smooth=True)
                       for m in meshes]

        scene      = pyrender.Scene(bg_color=[0.5,0.5,0.5,1.0])
        scene_mask = pyrender.Scene(bg_color=[0,0,0,1.0])
        for m in normal_meshes:
            scene.add(m)
        for m in mask_meshes:
            scene_mask.add(m)

        camera = pyrender.PerspectiveCamera(yfov=np.radians(45.0))
        cam_node      = scene.add(camera,      pose=np.eye(4))
        cam_node_mask = scene_mask.add(camera, pose=np.eye(4))

        scene.ambient_light      = [0.4,0.4,0.4,1.0]
        scene_mask.ambient_light = [0.4,0.4,0.4,1.0]
        sun = pyrender.DirectionalLight(color=[1,1,1], intensity=500.0)
        sun_pose = look_at(center + np.array([cam_dist, cam_dist, cam_dist]), center)
        scene.add(sun,      pose=sun_pose)
        scene_mask.add(sun, pose=sun_pose)

        renderer = pyrender.OffscreenRenderer(640, 480)

        for k in range(N):
            z = 1.0 - 2.0 * (k + 0.5) / N
            r = math.sqrt(max(0.0, 1.0 - z*z))
            phi = golden_angle * k
            cam_pos = center + cam_dist * np.array([r * math.cos(phi),
                                                    r * math.sin(phi),
                                                    z])
            cam_pose = look_at(cam_pos, center)
            scene.set_pose(cam_node,      cam_pose)
            scene_mask.set_pose(cam_node_mask, cam_pose)

            key_light = pyrender.PointLight(color=[1,1,1], intensity=600.0)
            key_node  = scene.add(key_light, pose=cam_pose)

            color, _    = renderer.render(scene)
            mask_img, _ = renderer.render(scene_mask)

            gray    = np.mean(mask_img, axis=2)
            mask_bin= (gray > 1e-6).astype(np.uint8) * 255

            cv2.imwrite(os.path.join(nature_dir, f"angle_{k:03d}.png"),
                        cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(mask_dir,   f"angle_{k:03d}.png"),
                        mask_bin)

            scene.remove_node(key_node)
            print(f"[{ds}] {base} finished: {k+1}/{N}")

        renderer.delete()
        print(f"[{ds}] finished {base} totally {N} pictures")
