import collections

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision

import torchsparse
import torchsparse.nn.functional as spf

from IOAR import cnn2d, cnn3d, utils, view_direction_encoder
from IOAR import mv_fusion


class IOAR(torch.nn.Module):
    def __init__(self, attn_heads, attn_layers, use_proj_occ):
        super().__init__()
        self.use_proj_occ = use_proj_occ
        self.n_attn_heads = attn_heads
        self.resolutions = collections.OrderedDict(
            [
                ["coarse", 0.16],
                ["medium", 0.08],
                ["fine", 0.04],
            ]
        )

        cnn2d_output_depths = [80, 40, 24]
        cnn3d_base_depths = [32, 16, 8]

        self.cnn2d = cnn2d.MnasMulti(cnn2d_output_depths, pretrained=True)
        self.upsampler = Upsampler()

        self.output_layers = torch.nn.ModuleDict()
        self.output_layers_occ = torch.nn.ModuleDict()
        self.cnns3d = torch.nn.ModuleDict()
        self.cnns3d_occ = torch.nn.ModuleDict()
        self.view_embedders = torch.nn.ModuleDict()
        self.layer_norms = torch.nn.ModuleDict()
        self.mv_fusion = torch.nn.ModuleDict()
        prev_output_depth = 0
        prev_output_depth_occ = 0
        for i, (resname, res) in enumerate(self.resolutions.items()):
            self.view_embedders[resname] = view_direction_encoder.ViewDirectionEncoder(
                cnn2d_output_depths[i], L=4
            )
            self.layer_norms[resname] = torch.nn.LayerNorm(cnn2d_output_depths[i])

            if self.n_attn_heads > 0:
                self.mv_fusion[resname] = mv_fusion.MVFusionTransformer(
                    cnn2d_output_depths[i], attn_layers, self.n_attn_heads
                )
            else:
                self.mv_fusion[resname] = mv_fusion.MVFusionMean()

            input_depth = prev_output_depth + cnn2d_output_depths[i]
            input_depth_occ = prev_output_depth_occ + cnn2d_output_depths[i]
            if i > 0:
                input_depth += 1
                input_depth_occ += 3
            conv = cnn3d.SPVCNN(
                in_channels=input_depth,
                base_depth=cnn3d_base_depths[i],
                dropout=False,
            )
            conv_occ = cnn3d.SPVCNN(
                in_channels=input_depth_occ,
                base_depth=cnn3d_base_depths[i],
                dropout=False,
            )
            output_depth = conv.output_depth
            output_depth_occ = conv_occ.output_depth
            self.cnns3d[resname] = conv
            self.cnns3d_occ[resname] = conv_occ
            self.output_layers[resname] = torchsparse.nn.Conv3d(
                output_depth, 1, kernel_size=1, stride=1)
            self.output_layers_occ[resname] = torchsparse.nn.Conv3d(
                output_depth_occ, 3, kernel_size=1, stride=1)
            prev_output_depth = conv.output_depth
            prev_output_depth_occ = conv_occ.output_depth

    def get_img_feats(self, rgb_imgs, proj_mats, cam_positions):
        batchsize, n_imgs, _, imheight, imwidth = rgb_imgs.shape
        feats = self.cnn2d(rgb_imgs.reshape((batchsize * n_imgs, *rgb_imgs.shape[2:])))
        for resname in self.resolutions:
            f = feats[resname]
            f = self.view_embedders[resname](f, proj_mats[resname], cam_positions)
            f = f.reshape((batchsize, n_imgs, *f.shape[1:]))
            feats[resname] = f
        return feats

    def forward(self, batch, voxel_inds_16):
        feats_2d = self.get_img_feats(
            batch["rgb_imgs"], batch["proj_mats"], batch["cam_positions"]
        )

        device = voxel_inds_16.device
        proj_occ_logits = {}
        voxel_logits_all = {}
        voxel_logits_occ_all = {}
        bp_data = {}
        n_subsample = {
            "medium": 2 ** 14,
            "fine": 2 ** 16,
        }

        voxel_inds = voxel_inds_16
        voxel_features = torch.empty(
            (len(voxel_inds), 0), dtype=feats_2d["coarse"].dtype, device=device
        )
        voxel_features_occ = torch.empty(
            (len(voxel_inds), 0), dtype=feats_2d["coarse"].dtype, device=device
        )
        voxel_logits = torch.empty(
            (len(voxel_inds), 0), dtype=feats_2d["coarse"].dtype, device=device
        )
        voxel_logits_occ = torch.empty(
            (len(voxel_inds), 0), dtype=feats_2d["coarse"].dtype, device=device
        )
        for resname, res in self.resolutions.items():
            if self.training and resname in n_subsample:
                subsample_inds = get_subsample_inds(voxel_inds, n_subsample[resname])
                voxel_inds = voxel_inds[subsample_inds]
                voxel_features = voxel_features[subsample_inds]
                voxel_features_occ = voxel_features_occ[subsample_inds]
                voxel_logits = voxel_logits[subsample_inds]
                voxel_logits_occ = voxel_logits_occ[subsample_inds]

            voxel_batch_inds = voxel_inds[:, 3].long()
            voxel_coords = voxel_inds[:, :3] * res + batch["origin"][voxel_batch_inds]

            featheight, featwidth = feats_2d[resname].shape[-2:]
            bp_uv, bp_depth, bp_mask = self.project_voxels(
                voxel_coords,
                voxel_batch_inds,
                batch["proj_mats"][resname].transpose(0, 1),
                featheight,
                featwidth,
            )
            bp_data[resname] = {
                "voxel_coords": voxel_coords,
                "voxel_batch_inds": voxel_batch_inds,
                "bp_uv": bp_uv,
                "bp_depth": bp_depth,
                "bp_mask": bp_mask,
            }
            bp_feats, cur_proj_occ_logits = self.back_project_features(
                bp_data[resname],
                feats_2d[resname].transpose(0, 1),
                self.mv_fusion[resname],
            )
            proj_occ_logits[resname] = cur_proj_occ_logits

            bp_feats = self.layer_norms[resname](bp_feats)

            voxel_features = torch.cat((voxel_features, bp_feats, voxel_logits), dim=-1)
            voxel_features_occ = torch.cat((voxel_features_occ, bp_feats, voxel_logits_occ), dim=-1)
            voxel_features = torchsparse.SparseTensor(voxel_features, voxel_inds)
            voxel_features_occ = torchsparse.SparseTensor(voxel_features_occ, voxel_inds)

            all_n_voxels = voxel_features.C.shape[0]
            if all_n_voxels == 0 or all_n_voxels == 1:
                return voxel_logits_all, voxel_logits_occ_all, proj_occ_logits, bp_data

            batch_inds = torch.unique(voxel_batch_inds)
            if len(batch_inds) == 1:
                coords = voxel_features.C
                max_coords = torch.max(coords, dim=0)[0]
                min_coords = torch.min(coords, dim=0)[0]
                gap = max_coords - min_coords
                biggest_gap = torch.max(gap, dim=0)[0]
                if biggest_gap < 4:
                    return voxel_logits_all, voxel_logits_occ_all, proj_occ_logits, bp_data
            

            voxel_features = self.cnns3d[resname](voxel_features)
            voxel_features_occ = self.cnns3d_occ[resname](voxel_features_occ)

            voxel_logits = self.output_layers[resname](voxel_features)
            voxel_logits_all[resname] = voxel_logits

            voxel_logits_occ = self.output_layers_occ[resname](voxel_features_occ)
            voxel_logits_occ_all[resname] = voxel_logits_occ

            if resname in ["coarse", "medium"]:
                occupancy = voxel_logits_occ.F.softmax(-1).argmax(-1) == 1
                if not torch.any(occupancy):
                    return voxel_logits_all, voxel_logits_occ_all, proj_occ_logits, bp_data
                voxel_features = self.upsampler.upsample_feats(
                    voxel_features.F[occupancy]
                )
                voxel_features_occ = self.upsampler.upsample_feats(
                    voxel_features_occ.F[occupancy]
                )
                voxel_inds = self.upsampler.upsample_inds(voxel_logits_occ.C[occupancy])
                voxel_logits = self.upsampler.upsample_feats(voxel_logits.F[occupancy])
                voxel_logits_occ = self.upsampler.upsample_feats(voxel_logits_occ.F[occupancy])

        return voxel_logits_all, voxel_logits_occ_all, proj_occ_logits, bp_data

    def losses(self, voxel_logits, voxels_logits_occ, voxel_gt, proj_occ_logits, bp_data, depth_imgs):
        voxel_losses = {}
        proj_occ_losses = {}
        for resname in voxel_logits:
            logits = voxel_logits[resname]
            logits_occ = voxels_logits_occ[resname]
            gt = voxel_gt[resname]
            cur_loss = torch.zeros(1, device=logits.F.device, dtype=torch.float32)
            if len(logits.C) > 0 and len(logits_occ.C) > 0:
                if resname == "fine":
                    pred_hash = spf.sphash(logits.C)
                    gt_hash = spf.sphash(gt.C)
                    idx_query = spf.sphashquery(pred_hash, gt_hash)
                    good_query = idx_query != -1
                    gt = gt.F[idx_query[good_query]]
                    logits = logits.F.squeeze(1)[good_query]
                    cur_loss = torch.nn.functional.l1_loss(
                        utils.log_transform(1.05 * torch.tanh(logits)),
                        utils.log_transform(gt),
                    )
                else:
                    pred_hash = spf.sphash(logits_occ.C)
                    gt_hash = spf.sphash(gt.C)
                    idx_query = spf.sphashquery(pred_hash, gt_hash)
                    good_query = idx_query != -1
                    gt = gt.F[idx_query[good_query]]
                    logits_occ = logits_occ.F[good_query]
                    cur_loss = torch.nn.functional.cross_entropy(
                        logits_occ, gt.long()
                    )
                voxel_losses[resname] = cur_loss

            proj_occ_losses[resname] = compute_proj_occ_loss(
                proj_occ_logits[resname],
                depth_imgs,
                bp_data[resname],
                truncation_distance=3 * self.resolutions[resname],
            )

        loss = sum(voxel_losses.values()) + sum(proj_occ_losses.values())
        logs = {
            **{
                f"voxel_loss_{resname}": voxel_losses[resname].item()
                for resname in voxel_losses
            },
            **{
                f"proj_occ_loss_{resname}": proj_occ_losses[resname].item()
                for resname in proj_occ_losses
            },
        }
        return loss, logs

    def project_voxels(
        self, voxel_coords, voxel_batch_inds, projmat, imheight, imwidth
    ):
        device = voxel_coords.device
        n_voxels = len(voxel_coords)
        n_imgs = len(projmat)
        bp_uv = torch.zeros((n_imgs, n_voxels, 2), device=device, dtype=torch.float32)
        bp_depth = torch.zeros((n_imgs, n_voxels), device=device, dtype=torch.float32)
        bp_mask = torch.zeros((n_imgs, n_voxels), device=device, dtype=torch.bool)
        batch_inds = torch.unique(voxel_batch_inds)
        for batch_ind in batch_inds:
            batch_mask = voxel_batch_inds == batch_ind
            if torch.sum(batch_mask) == 0:
                continue
            cur_voxel_coords = voxel_coords[batch_mask]

            ones = torch.ones(
                (len(cur_voxel_coords), 1), device=device, dtype=torch.float32
            )
            voxel_coords_h = torch.cat((cur_voxel_coords, ones), dim=-1)

            im_p = projmat[:, batch_ind] @ voxel_coords_h.t()
            im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
            im_x = im_x / im_z
            im_y = im_y / im_z
            im_grid = torch.stack(
                [2 * im_x / (imwidth - 1) - 1, 2 * im_y / (imheight - 1) - 1],
                dim=-1,
            )
            im_grid[torch.isinf(im_grid)] = -2
            mask = im_grid.abs() <= 1
            mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

            bp_uv[:, batch_mask] = im_grid.to(bp_uv.dtype)
            bp_depth[:, batch_mask] = im_z.to(bp_uv.dtype)
            bp_mask[:, batch_mask] = mask

        return bp_uv, bp_depth, bp_mask

    def back_project_features(self, bp_data, feats, mv_fuser):
        n_imgs, batch_size, in_channels, featheight, featwidth = feats.shape
        device = feats.device
        n_voxels = len(bp_data["voxel_batch_inds"])
        feature_volume_all = torch.zeros(
            n_voxels, in_channels, device=device, dtype=torch.float32
        )
        proj_occ_logits = torch.full(
            (n_imgs, n_voxels), 100, device=device, dtype=feats.dtype
        )
        batch_inds = torch.unique(bp_data["voxel_batch_inds"])
        for batch_ind in batch_inds:
            batch_mask = bp_data["voxel_batch_inds"] == batch_ind
            if torch.sum(batch_mask) == 0:
                continue

            cur_bp_uv = bp_data["bp_uv"][:, batch_mask]
            cur_bp_depth = bp_data["bp_depth"][:, batch_mask]
            cur_bp_mask = bp_data["bp_mask"][:, batch_mask]
            cur_feats = feats[:, batch_ind].view(
                n_imgs, in_channels, featheight, featwidth
            )
            cur_bp_uv = cur_bp_uv.view(n_imgs, 1, -1, 2)
            features = torch.nn.functional.grid_sample(
                cur_feats,
                cur_bp_uv.to(cur_feats.dtype),
                padding_mode="reflection",
                align_corners=True,
            )
            features = features.view(n_imgs, in_channels, -1)
            if isinstance(mv_fuser, mv_fusion.MVFusionTransformer):
                pooled_features, cur_proj_occ_logits = mv_fuser(
                    features,
                    cur_bp_depth,
                    cur_bp_mask,
                    self.use_proj_occ,
                )
                feature_volume_all[batch_mask] = pooled_features
                proj_occ_logits[:, batch_mask] = cur_proj_occ_logits
            else:
                pooled_features = mv_fuser(features.transpose(1, 2), cur_bp_mask)
                feature_volume_all[batch_mask] = pooled_features

        return (feature_volume_all, proj_occ_logits)


class Upsampler(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.upsample_offsets = torch.nn.Parameter(
            torch.Tensor(
                [
                    [
                        [0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [1, 1, 0, 0],
                        [0, 1, 1, 0],
                        [1, 0, 1, 0],
                        [1, 1, 1, 0],
                    ]
                ]
            ).to(torch.int32),
            requires_grad=False,
        )
        self.upsample_mul = torch.nn.Parameter(
            torch.Tensor([[[2, 2, 2, 1]]]).to(torch.int32), requires_grad=False
        )

    def upsample_inds(self, voxel_inds):
        return (
            voxel_inds[:, None] * self.upsample_mul + self.upsample_offsets
        ).reshape(-1, 4)

    def upsample_feats(self, feats):
        return (
            feats[:, None]
            .repeat(1, 8, 1)
            .reshape(-1, feats.shape[-1])
            .to(torch.float32)
        )


def get_subsample_inds(coords, max_per_example):
    keep_inds = []
    batch_inds = coords[:, 3].unique()
    for batch_ind in batch_inds:
        batch_mask = coords[:, -1] == batch_ind
        n = torch.sum(batch_mask)
        if n > max_per_example:
            keep_inds.append(batch_mask.float().multinomial(max_per_example))
        else:
            keep_inds.append(torch.where(batch_mask)[0])
    subsample_inds = torch.cat(keep_inds).long()
    return subsample_inds


def compute_proj_occ_loss(proj_occ_logits, depth_imgs, bp_data, truncation_distance):
    batch_inds = torch.unique(bp_data["voxel_batch_inds"])
    for batch_ind in batch_inds:
        batch_mask = bp_data["voxel_batch_inds"] == batch_ind
        cur_bp_uv = bp_data["bp_uv"][:, batch_mask]
        cur_bp_depth = bp_data["bp_depth"][:, batch_mask]
        cur_bp_mask = bp_data["bp_mask"][:, batch_mask]
        cur_proj_occ_logits = proj_occ_logits[:, batch_mask]

        depth = torch.nn.functional.grid_sample(
            depth_imgs[batch_ind, :, None],
            cur_bp_uv[:, None].to(depth_imgs.dtype),
            padding_mode="zeros",
            mode="nearest",
            align_corners=False,
        )[:, 0, 0]

        proj_occ_mask = cur_bp_mask & (depth > 0)
        if torch.sum(proj_occ_mask) > 0:
            proj_occ_gt = torch.abs(cur_bp_depth - depth) < truncation_distance
            return torch.nn.functional.binary_cross_entropy_with_logits(
                cur_proj_occ_logits[proj_occ_mask],
                proj_occ_gt[proj_occ_mask].to(cur_proj_occ_logits.dtype),
            )
        else:
            return torch.zeros((), dtype=torch.float32, device=depth_imgs.device)
