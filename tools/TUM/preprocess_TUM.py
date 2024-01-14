import os
import numpy as np
import transforms3d
import copy
import shutil
from tqdm.contrib import tzip

def read_gt_trajectory(path):
    trajectory_data = {}
    trajectory_data['timestamp'] = []
    trajectory_data['tx'] = []
    trajectory_data['ty'] = []
    trajectory_data['tz'] = []
    trajectory_data['qx'] = []
    trajectory_data['qy'] = []
    trajectory_data['qz'] = []
    trajectory_data['qw'] = []
    with open(path, 'r') as f:
        # skip the information in first three lines.
        f.readline()
        f.readline()
        f.readline()
        lines = f.readlines()
        for line in lines:
            timestamp, tx, ty, tz, qx, qy, qz, qw = line.split(" ")
            trajectory_data["timestamp"].append(float(timestamp))
            trajectory_data["tx"].append(float(tx))
            trajectory_data["ty"].append(float(ty))
            trajectory_data["tz"].append(float(tz))
            trajectory_data["qx"].append(float(qx))
            trajectory_data["qy"].append(float(qy))
            trajectory_data["qz"].append(float(qz))
            trajectory_data["qw"].append(float(qw))
        f.close()
    return trajectory_data


def read_depth(root_path, path):
    depth_data = {}
    depth_data['timestamp'] = []
    depth_data['depth_path'] = []
    with open(path, 'r') as f:
        # skip the information in first three lines.
        f.readline()
        f.readline()
        f.readline()
        lines = f.readlines()
        for line in lines:
            timestamp, tmp_path = line.split(" ")
            depth_data["timestamp"].append(float(timestamp))
            if os.sep == "\\":
                tmp_path.replace("/", os.sep)
            depth_data["depth_path"].append(os.path.join(root_path, tmp_path.strip()))
        f.close()
    return depth_data


def read_rgb(root_path, path):
    rbg_data = {}
    rbg_data['timestamp'] = []
    rbg_data['rgb_path'] = []
    with open(path, 'r') as f:
        # skip the information in first three lines.
        f.readline()
        f.readline()
        f.readline()
        lines = f.readlines()
        for line in lines:
            timestamp, tmp_path = line.split(" ")
            rbg_data["timestamp"].append(float(timestamp))
            if os.sep == "\\":
                tmp_path.replace("/", os.sep)
            rbg_data["rgb_path"].append(os.path.join(root_path, tmp_path.strip()))
        f.close()
    return rbg_data


def trajectory2pose(trajectory):
    num = len(trajectory["timestamp"])
    trajectory['extrinsic'] = []
    trajectory['rotations'] = []
    trajectory['translations'] = []
    for i in range(num):
        quat = np.array([trajectory['qw'][i], trajectory['qx'][i], trajectory['qy'][i], trajectory['qz'][i]])
        trajectory['rotations'].append(transforms3d.quaternions.quat2mat(quat))
        trajectory['translations'].append(np.array([trajectory['tx'][i], trajectory['ty'][i], trajectory['tz'][i]]))
        extrinsic = np.zeros((4, 4))
        extrinsic[:3, :3] = trajectory['rotations'][i]
        extrinsic[:3, 3] = trajectory['translations'][i]
        extrinsic[3, 3] = 1
        trajectory['extrinsic'].append(extrinsic)
    return trajectory

def find_match_depth_extrinsic(first_data, second_data, threshold=0.02):
    first_time = copy.deepcopy(first_data["timestamp"])
    scond_time = copy.deepcopy(second_data["timestamp"])
    potential_matches = [(abs(a-b), a, b, i, j) 
                         for i, a in enumerate(first_time) 
                         for j, b in enumerate(scond_time)
                         if abs(a-b) < threshold]
    # matches = []
    match_idx_first = []
    match_idx_second = []
    for diff, a, b, i, j in potential_matches:
        if a in first_time and b in scond_time:
            first_time.remove(a)
            scond_time.remove(b)
            # matches.append((a, b))
            match_idx_first.append(i)
            match_idx_second.append(j)
    ret_data = {}
    ret_data["depth_timestamp"] = np.array(first_data["timestamp"])[match_idx_first]
    ret_data["depth_path"] = np.array(first_data["depth_path"])[match_idx_first]
    ret_data["extrinsic_timestamp"] = np.array(second_data["timestamp"])[match_idx_second]
    ret_data["extrinsic"] = np.array(second_data["extrinsic"])[match_idx_second]
    return ret_data


def find_match_rgb_depth(first_data, second_data, threshold=0.02):
    first_time = list(copy.deepcopy(first_data["timestamp"]))
    scond_time = list(copy.deepcopy(second_data["depth_timestamp"]))
    potential_matches = [(abs(a-b), a, b, i, j) 
                         for i, a in enumerate(first_time) 
                         for j, b in enumerate(scond_time)
                         if abs(a-b) < threshold]
    # matches = []
    match_idx_first = []
    match_idx_second = []
    for diff, a, b, i, j in potential_matches:
        if a in first_time and b in scond_time:
            first_time.remove(a)
            scond_time.remove(b)
            # matches.append((a, b))
            match_idx_first.append(i)
            match_idx_second.append(j)
    ret_data = {}
    ret_data["rgb_timestamp"] = np.array(first_data["timestamp"])[match_idx_first]
    ret_data["rgb_path"] = np.array(first_data["rgb_path"])[match_idx_first]
    ret_data["extrinsic_timestamp"] = second_data["extrinsic_timestamp"][match_idx_second]
    ret_data["extrinsic"] = second_data["extrinsic"][match_idx_second]
    ret_data["depth_timestamp"] = second_data["depth_timestamp"][match_idx_second]
    ret_data["depth_path"] = second_data["depth_path"][match_idx_second]
    return ret_data



if __name__ == "__main__":
    data_root = "/home/yukun/data/TUM_RGBD"
    save_root = "/home/yukun/data/processed_TUM_RGBD_IOAR"
    scene_names = ["rgbd_dataset_freiburg2_large_no_loop",
            "rgbd_dataset_freiburg2_dishes",
            "rgbd_dataset_freiburg2_desk",
            "rgbd_dataset_freiburg1_desk",
            "rgbd_dataset_freiburg1_room",
            "rgbd_dataset_freiburg1_plant",
            "rgbd_dataset_freiburg1_teddy",
            "rgbd_dataset_freiburg3_cabinet", 
            "rgbd_dataset_freiburg3_long_office_household",
            "rgbd_dataset_freiburg3_nostructure_notexture_far",
            "rgbd_dataset_freiburg3_nostructure_texture_far",
            "rgbd_dataset_freiburg3_structure_notexture_far",
            "rgbd_dataset_freiburg3_structure_texture_far"]
    for scene in scene_names:
        print("processing scene:{}".format(scene))
        gt_path = os.path.join(data_root, scene, "groundtruth.txt")
        traj_data = read_gt_trajectory(gt_path)
        traj_data = trajectory2pose(traj_data)
        depth_path = os.path.join(data_root, scene, "depth.txt")
        depth_data = read_depth(os.path.join(data_root, scene), depth_path)
        rgb_path = os.path.join(data_root, scene, "rgb.txt")
        rgb_data = read_rgb(os.path.join(data_root, scene), rgb_path)
        depth_traj_data = find_match_depth_extrinsic(depth_data, traj_data)
        rgb_depth_traj_data = find_match_rgb_depth(rgb_data, depth_traj_data)
        rgb_paths = rgb_depth_traj_data["rgb_path"]
        depth_paths = rgb_depth_traj_data["depth_path"]
        extrinsics = rgb_depth_traj_data["extrinsic"]
        assert len(rgb_paths) == len(depth_paths)
        assert len(depth_paths) == len(extrinsics)
        os.makedirs(os.path.join(save_root, scene, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(save_root, scene, "depth"), exist_ok=True)
        os.makedirs(os.path.join(save_root, scene, "extrinsic"), exist_ok=True)
        for i, (rgb_path, depth_path, extrinsic) in enumerate(tzip(rgb_paths, depth_paths, extrinsics)):
            rgb_save_path = os.path.join(save_root, scene, "rgb", str(i).zfill(4)+".png")
            depth_save_path = os.path.join(save_root, scene, "depth", str(i).zfill(4)+".png")
            extrinsic_save_path = os.path.join(save_root, scene, "extrinsic", str(i).zfill(4)+".txt")
            shutil.copy(rgb_path, rgb_save_path)
            shutil.copy(depth_path, depth_save_path)
            np.savetxt(extrinsic_save_path, extrinsic)
