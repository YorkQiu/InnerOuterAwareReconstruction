import os
import numpy as np
import transforms3d
import pickle
import cv2
import argparse
from IOAR.tsdf_fusion import *
import open3d as o3d
import time
import glob
import os.path as osp


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=float, default=3.)
    parser.add_argument("--voxel_size", type=float, default=0.01)
    parser.add_argument("--pred_voxel_size", type=float, default=0.04)
    parser.add_argument("--margin", type=int, default=3)
    args = parser.parse_args()
    return args


def save_tsdf_info(vol_bnds, save_dir, voxel_size):
    vol_origin = vol_bnds[:, 0].copy(order="C").astype(np.float32)
    vol_dim = (np.round((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size).copy(order="C").astype(int))

    tsdf_info = {
        "vol_origin": vol_origin,
        "vol_dim": vol_dim,
        "voxel_size": voxel_size
    }
    with open(osp.join(save_dir, "tsdf_info.pkl"), 'wb') as f:
        pickle.dump(tsdf_info, f)


def process_single_scene(args, data_root, save_root, scene_name):
    save_path = os.path.join(save_root, scene_name, "tsdf2mesh.ply")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    depth_all = {}
    cam_pose_all = {}
    rgb_all = {}

    cam_intr = np.array([[525.0, 0.000, 319.5, 0.000],
                         [0.000, 525.0, 239.5, 0.000],
                         [0.000, 0.000, 1.000, 0.000],
                         [0.000, 0.000, 0.000, 1.000],
                         ])[:3, :3]

    rgb_paths = sorted(glob.glob(os.path.join(
        data_root, scene_name, "rgb", "*.png")))
    depth_paths = sorted(glob.glob(os.path.join(
        data_root, scene_name, "depth", "*.png")))
    pose_paths = sorted(glob.glob(os.path.join(
        data_root, scene_name, "extrinsic", "*.txt")))
    assert len(rgb_paths) == len(depth_paths)
    assert len(depth_paths) == len(pose_paths)

    n_imgs = len(rgb_paths)

    for id, (img_path, depth_path, pose_path) in enumerate(zip(rgb_paths, depth_paths, pose_paths)):
        if id % 100 == 0:
            print("read frame {}/{}".format(str(id), str(n_imgs)))
        rgb_img = cv2.imread(img_path, -1).astype(np.float32)
        depth_img = cv2.imread(depth_path, -1).astype(np.float32)
        depth_img /= 5000.
        depth_img[depth_img > args.max_depth] = 0

        cam_pose = np.loadtxt(pose_path)
        depth_all.update({id: depth_img})
        cam_pose_all.update({id: cam_pose})
        rgb_all.update({id: rgb_img})

    vol_bnds = np.zeros((3, 2))
    n_imgs = len(depth_all.keys())
    if n_imgs > 200:
        ind = np.linspace(0, n_imgs - 1, 200).astype(np.int32)
        image_id = np.array(list(depth_all.keys()))[ind]
    else:
        image_id = depth_all.keys()
    for id in image_id:
        depth_im = depth_all[id]
        cam_pose = cam_pose_all[id]

        # Compute camera view frustum and extend convex hull
        view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(
            vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(
            vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    print("vol_bnds", vol_bnds)
    save_tsdf_info(vol_bnds, osp.dirname(save_path), args.pred_voxel_size)

    print("Initializing voxel volume...")
    # tsdf_vol_list = []
    # for l in range(args.num_layers):
    # dbug
    tsdf = TSDFVolume(vol_bnds, voxel_size=args.voxel_size,
                      margin=args.margin, use_gpu=True)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for id in depth_all.keys():
        if id % 100 == 0:
            print("Fusing frame {}/{}.".format(str(id), str(n_imgs)))
        depth_im = depth_all[id]
        cam_pose = cam_pose_all[id]
        rgb_img = rgb_all[id]

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf.integrate(rgb_img, depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    print("Saveing mesh to {}".format(save_path))
    verts, faces, _, _ = tsdf.get_mesh()
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
    )
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(save_path, mesh)


if __name__ == "__main__":
    names = ["rgbd_dataset_freiburg1_room",
             "rgbd_dataset_freiburg1_plant",
             "rgbd_dataset_freiburg1_teddy",
             "rgbd_dataset_freiburg2_large_no_loop",
             "rgbd_dataset_freiburg2_dishes",
             "rgbd_dataset_freiburg2_desk",
             "rgbd_dataset_freiburg1_desk",
             "rgbd_dataset_freiburg3_cabinet",
             "rgbd_dataset_freiburg3_long_office_household",
             "rgbd_dataset_freiburg3_nostructure_notexture_far",
             "rgbd_dataset_freiburg3_nostructure_texture_far",
             "rgbd_dataset_freiburg3_structure_notexture_far",
             "rgbd_dataset_freiburg3_structure_texture_far"]
    args = get_parser()
    assert len(names) == 13
    for name in names:
        data_root = "/home/yukun/data/processed_TUM_RGBD_IOAR"
        save_root = "/home/yukun/data/processed_TUM_RGBD_IOAR"

        process_single_scene(args, data_root, save_root, name)
