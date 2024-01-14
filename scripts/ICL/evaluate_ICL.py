# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  Originating Author: Zak Murez (zak.murez.com)

#  Modified by Noah Stier

from IOAR import tsdf_fusion
from IOAR import data, eval_utils, utils
import tqdm
import trimesh
import pyrender
import numpy as np
import open3d as o3d
import imageio
import argparse
import glob
import json
import os
import yaml
# from inference_3CE_ICL import ICLSet
import pickle
from torch.utils.data import Dataset
import numpy as np
import cv2

os.environ["PYOPENGL_PLATFORM"] = "egl"


class ICLSet(Dataset):

    def __init__(self, data_root: str, mesh_path: str) -> None:
        super().__init__()
        self.scenes = list(filter(lambda x: os.path.isdir(os.path.join(data_root, x)), os.listdir(data_root)))
        self.data = []
        for scene in tqdm.tqdm(self.scenes, desc="Loading data", leave=False):
            depth_files = self._load_depth(os.path.join(data_root, scene))
            rgb_files = self._load_rgb(os.path.join(data_root, scene))
            poses = self._load_extrinsic(os.path.join(data_root, scene))
            tsdf_info = self._load_tsdf_info(os.path.join(data_root, scene))
            self.data.append({
                'name': scene,
                'depth_files': depth_files,
                'rgb_files': rgb_files,
                'extrinsic': poses,
                'intrinsic': np.array([
                    [481.20, 0.0000, 319.50, 0.0000],
                    [0.0000, -480.00, 239.50, 0.0000],
                    [0.0000, 0.0000, 1.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 1.0000],
                ])[:3, :3],
                'gt_mesh': os.path.join(mesh_path, scene, 'tsdf2mesh.ply'),
                'tsdf_info': tsdf_info
            })

    def __getitem__(self, index):
        return self.data[index]

    def _load_depth(self, scene_path):
        depth_dir = os.path.join(scene_path, 'depth')
        data = os.listdir(depth_dir)
        data.sort(key=lambda x: int(os.path.splitext(x)[0]))
        data = list(map(lambda x: os.path.join(depth_dir, x), data))
        return data

    def _load_rgb(self, scene_path):
        rgb_dir = os.path.join(scene_path, 'rgb')
        data = os.listdir(rgb_dir)
        data.sort(key=lambda x: int(os.path.splitext(x)[0]))
        data = list(map(lambda x: os.path.join(rgb_dir, x), data))
        return data

    def _load_extrinsic(self, scene_path):
        extrinsic_dir = os.path.join(scene_path, 'extrinsic')
        files = os.listdir(extrinsic_dir)
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        files = list(map(lambda x: os.path.join(extrinsic_dir, x), files))
        data = np.empty((len(files), 4, 4), dtype=np.float32)
        for i, f in enumerate(files):
            data[i] = np.loadtxt(f, dtype=np.float32)
        return data

    def _load_tsdf_info(self, scene_path):
        pkl_fname = os.path.join(scene_path, "tsdf_info.pkl")
        with open(pkl_fname, "rb") as tsdf_pkl:
            tsdf_info = pickle.load(tsdf_pkl)
        return tsdf_info

    def __len__(self):
        return len(self.scenes)


class Renderer:
    """OpenGL mesh renderer

    Used to render depthmaps from a mesh for 2d evaluation
    """

    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(
            cx=intrinsics[0, 2],
            cy=intrinsics[1, 2],
            fx=intrinsics[0, 0],
            fy=intrinsics[1, 1],
        )
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene, self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--icl-dir", type=str)
    parser.add_argument("--mesh-dir", type=str)
    args = parser.parse_args()

    # each item contains: name, rgb_files, depth_files, extrinsics, intrinsic, gt_mesh_file
    ICL_test_set = {d['name']: d for d in ICLSet(args.icl_dir, args.mesh_dir)}

    # read pred_plyfiles from result_dir
    pred_plyfiles = sorted(
        glob.glob(os.path.join(args.results_dir, "*", "prediction.ply"))
    )
    for pred_plyfile in tqdm.tqdm(pred_plyfiles, desc='Pred Files'):
        savefile = os.path.join(os.path.dirname(pred_plyfile), "metrics.json")

        if os.path.exists(savefile):
            continue

        scene_name = os.path.basename(os.path.dirname(pred_plyfile))
        icl_data = ICL_test_set[scene_name]

        # load pred mesh and gt mesh
        pred_mesh_o3d = o3d.io.read_triangle_mesh(pred_plyfile)
        pred_mesh_o3d.compute_vertex_normals()
        gt_mesh_o3d = o3d.io.read_triangle_mesh(icl_data['gt_mesh'])
        gt_mesh_o3d.compute_vertex_normals()
        # load tsdf data
        tsdf_info = icl_data['tsdf_info']
        origin = tsdf_info['vol_origin']
        voxel_size = tsdf_info['voxel_size']
        vol_dim = tsdf_info['vol_dim']

        tsdf_volume = tsdf_fusion.TSDFVolume(
            np.c_[origin, origin + np.array(vol_dim) * voxel_size],
            voxel_size,
            use_gpu=True,
            margin=3,
        )
        pred_mesh = trimesh.Trimesh(
            vertices=np.asarray(pred_mesh_o3d.vertices),
            faces=np.asarray(pred_mesh_o3d.triangles),
        )
        gt_mesh = trimesh.Trimesh(
            vertices=np.asarray(gt_mesh_o3d.vertices),
            faces=np.asarray(gt_mesh_o3d.triangles),
        )
        renderer = Renderer()
        pred_mesh_opengl = renderer.mesh_opengl(pred_mesh)
        gt_mesh_opengl = renderer.mesh_opengl(gt_mesh)

        rgb_imgfiles = icl_data['rgb_files']
        depth_imgfiles = icl_data['depth_files']
        pose = icl_data['extrinsic']
        intr = icl_data['intrinsic']

        depth_metrics = {}

        for i in tqdm.trange(len(pose), desc=f"Poses of {scene_name}", leave=False):
            depth_gt = imageio.imread(
                depth_imgfiles[i]).astype(np.float32) / 5000.0
            imheight, imwidth = depth_gt.shape

            _, depth_pred = renderer(
                imheight, imwidth, intr, pose[i], pred_mesh_opengl)
            for k, v in eval_utils.eval_depth(depth_pred, depth_gt).items():
                if k not in depth_metrics:
                    depth_metrics[k] = []
                else:
                    depth_metrics[k].append(v)

            _, gt_mask = renderer(imheight, imwidth, intr,
                                  pose[i], gt_mesh_opengl)
            depth_without_mask = depth_pred.copy()
            depth_pred[gt_mask == 0] = 0

            rgb_img = imageio.imread(rgb_imgfiles[i])
            tsdf_volume.integrate(rgb_img, depth_pred, intr, pose[i])

        tsdf_vol, color_vol, weight_vol = tsdf_volume.get_volume()
        tsdf_vol = -tsdf_vol
        tsdf_vol[weight_vol == 0] = np.nan
        pred_mesh = utils.to_mesh(
            -tsdf_vol, voxel_size=0.04, level=0, origin=origin
        )

        # utils.visualize([pred_mesh, gt_mesh])

        depth_metrics = {k: np.nanmean(v) for k, v in depth_metrics.items()}
        mesh_metrics = eval_utils.eval_mesh(pred_mesh, gt_mesh)
        metrics = {**depth_metrics, **mesh_metrics}
        metrics = {k: float(v) for k, v in metrics.items()}
        renderer.delete()

        # write metrics and trimmed mesh
        o3d.io.write_triangle_mesh(
            os.path.join(os.path.dirname(pred_plyfile),
                         "trimmed_mesh.ply"), pred_mesh
        )
        with open(savefile, "w") as f:
            json.dump(metrics, f)
