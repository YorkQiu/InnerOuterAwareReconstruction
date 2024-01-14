import argparse
import glob
import json
import os
import random
import subprocess

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.plugins.precision import NativeMixedPrecisionPlugin
import torch
import yaml

from IOAR import collate, data, utils

from IOAR import lightningmodel 

os.environ["PYOPENGL_PLATFORM"] = "egl"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--version", type=str)
    parser.add_argument("--gpu", type=str, default="0,")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    pl.seed_everything(config["seed"])
    
    logger = pl.loggers.WandbLogger(project=config["wandb_project_name"], config=config, version=args.version)
    
    checkpointer = pl.callbacks.ModelCheckpoint(
        save_last=True,
        dirpath=config["ckpt_dir"],
        verbose=True,
        save_top_k=2,
        monitor="val/loss",
    )
    exist_last_ckpt =  os.path.exists(os.path.join(config["ckpt_dir"], "last.ckpt"))
    # load specific ckpt if set
    if config["ckpt"] is not None:
        ckpt = os.path.join(config["ckpt_dir"], config["ckpt"])
    # load last ckpt if exist
    elif exist_last_ckpt:
        ckpt = os.path.join(config["ckpt_dir"], "last.ckpt")
    else:
        ckpt = None
    print("load ckpt:{}".format(ckpt))

    
    model = lightningmodel.LightningModel(config)
    callbacks = [checkpointer, lightningmodel.FineTuning(config["initial_epochs"])]
    print("Using model: {}.".format(config["model_name"]))

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpu,
        logger=logger,
        benchmark=True,
        max_epochs=config["initial_epochs"] + config["finetune_epochs"],
        check_val_every_n_epoch=5,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1, 
        plugins=[NativeMixedPrecisionPlugin(16, "cuda", torch.cuda.amp.GradScaler())]
    )
    trainer.fit(model, ckpt_path=ckpt)
