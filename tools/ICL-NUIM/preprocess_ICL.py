import copy
import os
import os.path as osp
import shutil
from glob import glob

import numpy as np
import transforms3d
from tqdm import tqdm, trange
from tqdm.contrib import tzip


def read_gt_trajectory(path):
    trajectory_data = np.loadtxt(path, dtype=np.float32)
    timestamp = trajectory_data[:, 0].astype(np.int32)
    # numpy array of shape (N, 7), with columns of:
    # tx, ty, tz, qx, qy, qz, qw
    trajectory_data = trajectory_data[:, 1:]
    trajectory_data = dict(zip(
        timestamp.tolist(),
        map(lambda x: np.array(x), trajectory_data.tolist()),
    ))
    return trajectory_data


def trajectory2pose(trajectory):
    extrinsics = {}
    for i, v in trajectory.items():
        # each row is a tuple of (tx, ty, tz, qx, qy, qz, qw)
        quat = v[-4:][[3, 0, 1, 2]]  # (qw, qx, qy, qz)
        rotation = transforms3d.quaternions.quat2mat(quat)
        # fix "rotation" and "translation" to a new coordinate
        # where z-axis points towards the ceil
        rot_mat = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ])
        rotation = rot_mat @ rotation
        translation = rot_mat @ v[:3]

        extrinsic = np.zeros((4, 4), dtype=np.float32)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = translation
        extrinsic[3, 3] = 1.0

        extrinsics[i] = extrinsic
    return extrinsics


if __name__ == "__main__":
    data_root = "/home/yukun/data/ICL_V2"
    save_root = "/home/yukun/data/processed_ICL_IOAR"
    scene_names = [
        "living_room_traj0",
        "living_room_traj1",
        "living_room_traj2",
        "living_room_traj3",
        "office_room_traj0",
        "office_room_traj1",
        "office_room_traj2",
        "office_room_traj3",
    ]
    scene_names = [f'{n}_frei_png' for n in scene_names]

    for scene in tqdm(scene_names):
        print(f"\nprocessing scene:{scene}")

        gt_pose_path = glob(osp.join(data_root, scene, '*.gt.freiburg'))
        assert len(gt_pose_path) == 1
        gt_pose_path = gt_pose_path[0]

        traj_data = read_gt_trajectory(gt_pose_path)
        traj_data = trajectory2pose(traj_data)

        rgb_paths = glob(osp.join(data_root, scene, 'rgb', '*.png'))
        rgb_indices = [
            int(osp.splitext(osp.split(p)[1])[0])
            for p in rgb_paths
        ]
        rgb_data = dict(zip(rgb_indices, rgb_paths))

        depth_paths = glob(osp.join(data_root, scene, 'depth', '*.png'))
        depth_indices = [
            int(osp.splitext(osp.split(p)[1])[0])
            for p in depth_paths
        ]
        depth_data = dict(zip(depth_indices, depth_paths))

        os.makedirs(osp.join(save_root, scene, "rgb"), exist_ok=True)
        os.makedirs(osp.join(save_root, scene, "depth"), exist_ok=True)
        os.makedirs(osp.join(save_root, scene, "extrinsic"), exist_ok=True)

        min_index = min(
            min(traj_data.keys()),
            min(rgb_data.keys()),
            min(depth_data.keys()),
        )
        max_index = max(
            max(traj_data.keys()),
            max(rgb_data.keys()),
            max(depth_data.keys()),
        )
        print(f'{min_index=}')
        print(f'{max_index=}')
        for i in trange(min_index, max_index):
            if i in rgb_data.keys() and i in depth_data.keys() and i in traj_data.keys():
                rgb_save_path = osp.join(save_root, scene, 'rgb')
                depth_save_path = osp.join(save_root, scene, 'depth')
                extrinsic_save_path = osp.join(
                    save_root, scene, 'extrinsic', f'{i}.txt')

                shutil.copy(rgb_data[i], rgb_save_path)
                shutil.copy(depth_data[i], depth_save_path)
                np.savetxt(extrinsic_save_path, traj_data[i])
            else:
                print(f'\nNot found: {i}')
