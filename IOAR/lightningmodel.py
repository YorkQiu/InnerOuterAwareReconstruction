import os

import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import torch

from IOAR import collate, data, utils
from IOAR import IOAR


class FineTuning(pl.callbacks.BaseFinetuning):
    def __init__(self, initial_epochs):
        super().__init__()
        self.initial_epochs = initial_epochs

    def freeze_before_training(self, pl_module):
        modules = [
            pl_module.vortx.cnn2d.conv0,
            pl_module.vortx.cnn2d.conv1,
            pl_module.vortx.cnn2d.conv2,
        ]
        for mod in modules:
            self.freeze(mod, train_bn=False)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        if current_epoch >= self.initial_epochs:
            self.unfreeze_and_add_param_group(
                modules=[
                    pl_module.vortx.cnn2d.conv0,
                    pl_module.vortx.cnn2d.conv1,
                    pl_module.vortx.cnn2d.conv2,
                ],
                optimizer=optimizer,
                train_bn=False,
                lr=pl_module.config["finetune_lr"],
            )
            pl_module.vortx.use_proj_occ = True
            for group in pl_module.optimizers().param_groups:
                group["lr"] = pl_module.config["finetune_lr"]


class LightningModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.vortx = IOAR.IOAR(
            config["attn_heads"], config["attn_layers"], config["use_proj_occ"]
        )
        self.config = config

    def configure_optimizers(self):
        return torch.optim.Adam(
            [param for param in self.parameters() if param.requires_grad],
            lr=self.config["initial_lr"],
        )

    def on_train_epoch_start(self):
        self.epoch_train_logs = []

    def step(self, batch, batch_idx):
        voxel_coords_16 = batch["input_voxels_16"].C
        voxel_logits, voxel_logits_occ, proj_occ_logits, bp_data = self.vortx(batch, voxel_coords_16)
        voxel_gt = {
            "coarse": batch["voxel_gt_coarse"],
            "medium": batch["voxel_gt_medium"],
            "fine": batch["voxel_gt_fine"],
        }
        loss, logs = self.vortx.losses(
            voxel_logits, voxel_logits_occ, voxel_gt, proj_occ_logits, bp_data, batch["depth_imgs"]
        )
        logs["loss"] = loss.item()
        return loss, logs, voxel_logits, voxel_logits_occ

    def training_step(self, batch, batch_idx):
        n_warmup_steps = 2_000
        if self.global_step < n_warmup_steps:
            target_lr = self.config["initial_lr"]
            lr = 1e-10 + self.global_step / n_warmup_steps * target_lr
            for group in self.optimizers().param_groups:
                group["lr"] = lr

        loss, logs, voxel_logits, voxel_logits_occ = self.step(batch, batch_idx)
        self.epoch_train_logs.append(logs)
        return loss

    def on_validation_epoch_start(self):
        self.epoch_val_logs = []

    def validation_step(self, batch, batch_idx):
        loss, logs, voxel_logits, voxel_logits_occ = self.step(batch, batch_idx)
        self.epoch_val_logs.append(logs)

    def training_epoch_end(self, outputs):
        self.epoch_end(self.epoch_train_logs, "train")

    def validation_epoch_end(self, outputs):
        self.epoch_end(self.epoch_val_logs, "val")

    def epoch_end(self, logs, prefix):
        keys = set([key for log in logs for key in log])
        results = {key: [] for key in keys}
        for log in logs:
            for key, value in log.items():
                results[key].append(value)
        logs = {f"{prefix}/{key}": np.nanmean(results[key]) for key in keys}
        self.log_dict(logs)

    def train_dataloader(self):
        return self.dataloader("train", augment=True)

    def val_dataloader(self):
        return self.dataloader("val")

    def dataloader(self, split, augment=False):
        if split == "val":
            batch_size = 1
        elif self.current_epoch < self.config["initial_epochs"]:
            batch_size = self.config["initial_batch_size"]
        else:
            batch_size = self.config["finetune_batch_size"]

        info_files = utils.load_info_files(self.config["scannet_dir"], split)
        dset = data.Dataset(
            info_files,
            self.config["tsdf_dir"],
            self.config[f"n_imgs_{split}"],
            self.config[f"crop_size_{split}"],
            augment=augment,
        )
        return torch.utils.data.DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=4,
            collate_fn=collate.sparse_collate_fn,
            drop_last=True,
            pin_memory=False
        )
