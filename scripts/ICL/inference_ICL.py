import argparse
import json
import os
import pickle
import yaml

import imageio
import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import skimage.measure
import torch
import torchsparse
import tqdm
from torch.utils.data import DataLoader, Dataset

from IOAR import data, utils

from IOAR import lightningmodel


class ICLSet(Dataset):

    def __init__(self, data_root: str) -> None:
        super().__init__()
        self.scenes = list(filter(lambda x: os.path.isdir(os.path.join(data_root, x)), os.listdir(data_root)))
        self.data = []
        for scene in tqdm.tqdm(self.scenes, desc="Loading data", leave=False):
            depth_files = self._load_depth(os.path.join(data_root, scene))
            rgb_files = self._load_rgb(os.path.join(data_root, scene))
            poses = self._load_extrinsic(os.path.join(data_root, scene))
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

    def __len__(self):
        return len(self.scenes)


def load_model(ckpt_file, use_proj_occ, config):
    model = lightningmodel.LightningModel.load_from_checkpoint(
        ckpt_file, config=config)
    print("load IOAR.")
    model.vortx.use_proj_occ = use_proj_occ
    model = model.cuda()
    model = model.eval()
    model.requires_grad_(False)
    return model


# load rgb_files path, depth_files path, and poses, intr from info.json
def load_scene(info_file):
    # load info.json info to help load imgs, depths, intr, pose of this scene.
    with open(info_file, "r") as f:
        info = json.load(f)

    rgb_imgfiles = [frame["filename_color"] for frame in info["frames"]]
    depth_imgfiles = [frame["filename_depth"] for frame in info["frames"]]
    pose = np.empty((len(info["frames"]), 4, 4), dtype=np.float32)
    for i, frame in enumerate(info["frames"]):
        pose[i] = frame["pose"]
    intr = np.array(info["intrinsics"], dtype=np.float32)
    return rgb_imgfiles, depth_imgfiles, pose, intr


def get_scene_bounds(pose, intr, imheight, imwidth, frustum_depth):
    # four frustum pts in img
    frust_pts_img = np.array(
        [
            [0, 0],
            [imwidth, 0],
            [imwidth, imheight],
            [0, imheight],
        ]
    )
    # change to cam coord
    # 4 x 3
    frust_pts_cam = (
        np.linalg.inv(intr) @ np.c_[frust_pts_img,
                                    np.ones(len(frust_pts_img))].T
    ).T * frustum_depth
    # change to world coord
    # n_imgs x 4 x 3
    frust_pts_world = (
        pose @ np.c_[frust_pts_cam, np.ones(len(frust_pts_cam))].T
    ).transpose(0, 2, 1)[..., :3]

    # get bound
    minbound = np.min(frust_pts_world, axis=(0, 1))
    maxbound = np.max(frust_pts_world, axis=(0, 1))
    return minbound, maxbound


def get_tiles(minbound, maxbound, cropsize_voxels_fine, voxel_size_fine):
    # change cropsize to meter.
    cropsize_m = cropsize_voxels_fine * voxel_size_fine

    # ensure voxel_size can divide to coarse
    assert np.all(cropsize_voxels_fine % 4 == 0)
    cropsize_voxels_coarse = cropsize_voxels_fine // 4
    voxel_size_coarse = voxel_size_fine * 4

    # how many tiles to crop
    ncrops = np.ceil((maxbound - minbound) / cropsize_m).astype(int)
    # get the origins of tiles
    x = np.arange(ncrops[0], dtype=np.int32) * cropsize_voxels_coarse[0]
    y = np.arange(ncrops[1], dtype=np.int32) * cropsize_voxels_coarse[1]
    z = np.arange(ncrops[2], dtype=np.int32) * cropsize_voxels_coarse[2]
    yy, xx, zz = np.meshgrid(y, x, z)
    tile_origin_inds = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

    # get the base voxel inds (for each tile)
    x = np.arange(0, cropsize_voxels_coarse[0], dtype=np.int32)
    y = np.arange(0, cropsize_voxels_coarse[1], dtype=np.int32)
    z = np.arange(0, cropsize_voxels_coarse[2], dtype=np.int32)
    yy, xx, zz = np.meshgrid(y, x, z)
    base_voxel_inds = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

    tiles = []
    for origin_ind in tile_origin_inds:
        origin = origin_ind * voxel_size_coarse + minbound
        tile = {
            "origin_ind": origin_ind,
            "origin": origin.astype(np.float32),
            "maxbound_ind": origin_ind + cropsize_voxels_coarse,
            "voxel_inds": torch.from_numpy(base_voxel_inds + origin_ind),
            "voxel_coords": torch.from_numpy(
                base_voxel_inds * voxel_size_coarse + origin
            ).float(),
            "voxel_features": torch.empty(
                (len(base_voxel_inds), 0), dtype=torch.float32
            ),
            "voxel_features_occ": torch.empty(
                (len(base_voxel_inds), 0), dtype=torch.float32
            ),
            "voxel_logits": torch.empty((len(base_voxel_inds), 0), dtype=torch.float32),
            "voxel_logits_occ": torch.empty((len(base_voxel_inds), 0), dtype=torch.float32),
        }
        tiles.append(tile)
    return tiles


def frame_selection(tiles, pose, intr, imheight, imwidth, n_imgs, rmin_deg, tmin):
    # filter frames based on the rotation and translation.
    sparsified_frame_inds = np.array(
        utils.remove_redundant(pose, rmin_deg, tmin))

    if len(sparsified_frame_inds) < n_imgs:
        # after redundant frame removal we can end up with too few frames--
        # add some back in
        avail_inds = list(set(np.arange(len(pose))) -
                          set(sparsified_frame_inds))
        n_needed = n_imgs - len(sparsified_frame_inds)
        extra_inds = np.random.choice(avail_inds, size=n_needed, replace=False)
        selected_frame_inds = np.concatenate(
            (sparsified_frame_inds, extra_inds))
    else:
        selected_frame_inds = sparsified_frame_inds

    for i, tile in enumerate(tiles):
        if len(selected_frame_inds) > n_imgs:
            sample_pts = tile["voxel_coords"].numpy()
            cur_frame_inds, score = utils.frame_selection(
                pose[selected_frame_inds],
                intr,
                imwidth,
                imheight,
                sample_pts,
                tmin,
                rmin_deg,
                n_imgs,
            )
            tile["frame_inds"] = selected_frame_inds[cur_frame_inds]
        else:
            tile["frame_inds"] = selected_frame_inds
    return tiles


def get_img_feats(vortx, imheight, imwidth, proj_mats, rgb_imgfiles, cam_positions):
    imsize = np.array([imheight, imwidth])
    dims = {
        "coarse": imsize // 16,
        "medium": imsize // 8,
        "fine": imsize // 4,
    }
    feats_2d = {
        "coarse": torch.empty(
            (1, len(rgb_imgfiles), 80, *dims["coarse"]), dtype=torch.float16
        ),
        "medium": torch.empty(
            (1, len(rgb_imgfiles), 40, *dims["medium"]), dtype=torch.float16
        ),
        "fine": torch.empty(
            (1, len(rgb_imgfiles), 24, *dims["fine"]), dtype=torch.float16
        ),
    }
    cam_positions = torch.from_numpy(cam_positions).cuda()[None]
    for i in range(len(rgb_imgfiles)):
        rgb_img = data.load_rgb_imgs([rgb_imgfiles[i]], imheight, imwidth)
        rgb_img = torch.from_numpy(rgb_img).cuda()[None]
        cur_proj_mats = {k: v[:, i, None] for k, v in proj_mats.items()}
        cur_feats_2d = model.vortx.get_img_feats(
            rgb_img, cur_proj_mats, cam_positions[:, i, None]
        )
        for resname in feats_2d:
            feats_2d[resname][0, i] = cur_feats_2d[resname][0, 0].cpu()
    return feats_2d


# yukun: main different from original version.
def inference(model, scene_data, outfile, n_imgs, cropsize, config):
    # load rgb, depth, pose, intr based on info file
    rgb_imgfiles = scene_data['rgb_files']
    depth_imgfiles = scene_data['depth_files']
    pose = scene_data['extrinsic']
    intr = scene_data['intrinsic']
    test_img = imageio.imread(rgb_imgfiles[0])
    imheight, imwidth, _ = test_img.shape

    # get scene bound in world coord with the help of intr and pose
    scene_minbound, scene_maxbound = get_scene_bounds(
        pose, intr, imheight, imwidth, frustum_depth=4)
    print("scene_minbound", scene_minbound)
    print("scene_maxbound", scene_maxbound)

    pose_w2c = np.linalg.inv(pose)
    # split the whole scene to several tiles
    tiles = get_tiles(
        scene_minbound,
        scene_maxbound,
        cropsize_voxels_fine=np.array(cropsize),
        voxel_size_fine=0.04,
    )

    # pre-select views for each tile, same strategy as training phase.
    tiles = frame_selection(
        tiles, pose, intr, imheight, imwidth, n_imgs=n_imgs, rmin_deg=15, tmin=0.1
    )

    # drop the frames that weren't selected for any tile, re-index the selected frame indicies
    selected_frame_inds = np.unique(
        np.concatenate([tile["frame_inds"] for tile in tiles])
    )
    all_frame_inds = np.arange(len(pose))
    # re-index from 0 to len(selected_frame_inds)
    frame_reindex = np.full(len(all_frame_inds), 100_000)
    frame_reindex[selected_frame_inds] = np.arange(len(selected_frame_inds))
    for tile in tiles:
        tile["frame_inds"] = frame_reindex[tile["frame_inds"]]
    pose_w2c = pose_w2c[selected_frame_inds]
    pose = pose[selected_frame_inds]
    rgb_imgfiles = np.array(rgb_imgfiles)[selected_frame_inds]

    factors = np.array([1 / 16, 1 / 8, 1 / 4])
    proj_mats = data.get_proj_mats(intr, pose_w2c, factors)
    proj_mats = {k: torch.from_numpy(v)[None].cuda()
                 for k, v in proj_mats.items()}
    # extract img_feats based on the trained model
    img_feats = get_img_feats(
        model,
        imheight,
        imwidth,
        proj_mats,
        rgb_imgfiles,
        cam_positions=pose[:, :3, 3],
    )
    for resname, res in model.vortx.resolutions.items():

        # populate feature volume independently for each tile
        for tile in tiles:
            voxel_coords = tile["voxel_coords"].cuda()
            voxel_batch_inds = torch.zeros(
                len(voxel_coords), dtype=torch.int64, device="cuda"
            )

            cur_img_feats = img_feats[resname][:, tile["frame_inds"]].cuda()
            cur_proj_mats = proj_mats[resname][:, tile["frame_inds"]]

            featheight, featwidth = img_feats[resname].shape[-2:]
            bp_uv, bp_depth, bp_mask = model.vortx.project_voxels(
                voxel_coords,
                voxel_batch_inds,
                cur_proj_mats.transpose(0, 1),
                featheight,
                featwidth,
            )
            bp_data = {
                "voxel_batch_inds": voxel_batch_inds,
                "bp_uv": bp_uv,
                "bp_depth": bp_depth,
                "bp_mask": bp_mask,
            }

            bp_feats, proj_occ_logits = model.vortx.back_project_features(
                bp_data,
                cur_img_feats.transpose(0, 1),
                model.vortx.mv_fusion[resname],
            )

            bp_feats = model.vortx.layer_norms[resname](bp_feats)

            tile["voxel_features"] = torch.cat(
                (tile["voxel_features"], bp_feats.cpu(), tile["voxel_logits"]),
                dim=-1,
            )

            tile["voxel_features_occ"] = torch.cat(
                (tile["voxel_features_occ"],
                 bp_feats.cpu(), tile["voxel_logits_occ"]),
                dim=-1,
            )

        # combine all tiles into one sparse tensor & run convolution
        voxel_inds = torch.cat([tile["voxel_inds"] for tile in tiles], dim=0)
        voxel_batch_inds = torch.zeros((len(voxel_inds), 1), dtype=torch.int32)
        voxel_features = torchsparse.SparseTensor(
            torch.cat([tile["voxel_features"]
                      for tile in tiles], dim=0).cuda(),
            torch.cat([voxel_inds, voxel_batch_inds], dim=-1).cuda(),
        )
        voxel_features_occ = torchsparse.SparseTensor(
            torch.cat([tile["voxel_features_occ"]
                      for tile in tiles], dim=0).cuda(),
            torch.cat([voxel_inds, voxel_batch_inds], dim=-1).cuda(),
        )

        voxel_features = model.vortx.cnns3d[resname](voxel_features)
        voxel_features_occ = model.vortx.cnns3d_occ[resname](
            voxel_features_occ)
        voxel_logits = model.vortx.output_layers[resname](voxel_features)
        voxel_logits_occ = model.vortx.output_layers_occ[resname](
            voxel_features_occ)

        if resname in ["coarse", "medium"]:
            # sparsify & upsample
            # occupancy = voxel_logits.F.squeeze(1) > 0
            occupancy = voxel_logits_occ.F.softmax(-1).argmax(-1) == 1
            if not torch.any(occupancy):
                raise Exception("um")
            voxel_features = model.vortx.upsampler.upsample_feats(
                voxel_features.F[occupancy]
            )
            voxel_features_occ = model.vortx.upsampler.upsample_feats(
                voxel_features_occ.F[occupancy]
            )
            voxel_inds = model.vortx.upsampler.upsample_inds(
                voxel_logits_occ.C[occupancy])
            voxel_logits = model.vortx.upsampler.upsample_feats(
                voxel_logits.F[occupancy]
            )
            voxel_logits_occ = model.vortx.upsampler.upsample_feats(
                voxel_logits_occ.F[occupancy]
            )
            voxel_features = voxel_features.cpu()
            voxel_features_occ = voxel_features_occ.cpu()
            voxel_inds = voxel_inds.cpu()
            voxel_logits = voxel_logits.cpu()
            voxel_logits_occ = voxel_logits_occ.cpu()

            # split back up into tiles
            for tile in tiles:
                tile["origin_ind"] *= 2
                tile["maxbound_ind"] *= 2

                tile_voxel_mask = (
                    (voxel_inds[:, 0] >= tile["origin_ind"][0])
                    & (voxel_inds[:, 1] >= tile["origin_ind"][1])
                    & (voxel_inds[:, 2] >= tile["origin_ind"][2])
                    & (voxel_inds[:, 0] < tile["maxbound_ind"][0])
                    & (voxel_inds[:, 1] < tile["maxbound_ind"][1])
                    & (voxel_inds[:, 2] < tile["maxbound_ind"][2])
                )

                tile["voxel_inds"] = voxel_inds[tile_voxel_mask, :3]
                tile["voxel_features"] = voxel_features[tile_voxel_mask]
                tile["voxel_features_occ"] = voxel_features_occ[tile_voxel_mask]
                tile["voxel_logits"] = voxel_logits[tile_voxel_mask]
                tile["voxel_logits_occ"] = voxel_logits_occ[tile_voxel_mask]
                tile["voxel_coords"] = tile["voxel_inds"] * (res / 2) \
                    + scene_minbound.astype(np.float32)

    # convert logits to tsdf vol
    tsdf_vol = utils.to_vol(
        voxel_logits.C[:, :3].cpu().numpy(),
        1.05 * torch.tanh(voxel_logits.F).squeeze(-1).cpu().numpy(),
    )
    mesh = utils.to_mesh(
        -tsdf_vol,
        voxel_size=0.04,
        origin=scene_minbound,
        level=0,
        mask=~np.isnan(tsdf_vol),
    )
    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--use-proj-occ", default=True, type=bool)
    parser.add_argument("--n-imgs", default=60, type=int)
    parser.add_argument("--cropsize", default=64, type=int)
    parser.add_argument("--icl-dir", type=str)
    args = parser.parse_args()

    pl.seed_everything(0)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    cropsize = (args.cropsize, args.cropsize, 48)

    with torch.cuda.amp.autocast():

        # load info files for each scene.
        test_scenes = ICLSet(args.icl_dir)
        model = load_model(args.ckpt, args.use_proj_occ, config)
        for scene_data in tqdm.tqdm(test_scenes):
            scene_name = scene_data['name']
            outdir = os.path.join(args.outputdir, scene_name)
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, "prediction.ply")

            if os.path.exists(outfile):
                print(outfile, 'exists, skipping')
                continue

            try:
                # get 3dmesh
                mesh = inference(model, scene_data, outfile,
                                 args.n_imgs, cropsize, config)
                o3d.io.write_triangle_mesh(outfile, mesh)
            except Exception as e:
                print(e)
