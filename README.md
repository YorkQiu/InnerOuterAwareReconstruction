# Inner Outer Aware Reconstruction Model for Monocular 3D Scene Reconstruction

This is an official Pytorch implementation of our paper: Inner Outer Aware Reconstruction Model for Monocular 3D Scene Reconstruction. Advances in Neural Information Processing Systems (NeurIPS), 2023.

![Pipeline](https://cdn.statically.io/gh/YorkQiu/picx-images-hosting@master/20240114/image.4a0pzg2ejcc0.webp)

## setup

Operation System: Ubuntu 18.04

GPU: one Nvidia RTX 3090

CPU: Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz

### Dependencies

```
conda create -n IOAR python=3.9 -y
conda activate IOAR

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

pip install \
  pytorch-lightning==1.5 \
  scikit-image==0.18 \
  numba \
  pillow \
  wandb \
  tqdm \
  open3d \
  pyrender \
  ray \
  trimesh \
  pyyaml \
  matplotlib \
  black \
  pycuda \
  opencv-python \
  imageio \
  transforms3d
  
sudo apt install libsparsehash-dev

pip install git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

pip install -e .
```

However, the newest `setuptools ` package is not compatible with `tensorboard`.  And will raise an error:

```
AttributeError: module 'distutils' has no attribute 'version'
```

So, we have to downgrade it:

```
pip uninstall setuptools

pip install setuptools==58.0.4
```

Now, the code should work properly.

## Config

We provide an example config in `configs/example_config.yaml`. users should copy this config and modify it.

## Data

### ScanNet

You can download and extract the data from [ScanNet](https://github.com/ScanNet/ScanNet). 


To prepare the data for IOAR:

```
python tools/preprocess_scannet.py --src path/to/raw_scannet --dst path/to/processed_scannet
```

In config, you should set `scannet_dir` to `--dst`. 

Then, you can generate ground truth TSDF:

```
# For training data
python tools/generate_gt.py --data_path path/to/processed_scannet --save_name TSDF_OUTPUT_DIR
# For testing data
python tools/generate_gt.py --test --data_path path/to/processed_scannet --save_name TSDF_OUTPUT_DIR
```

Finally, set `tsdf_dir` to `path/to/processed_scannet/TSDF_OUTPUT_DIR`.

**NOTE: the depth map of scene0186_01 seems to have some problems in my case. It can not be processed by generate_gt.py. So, we simply remove it from the training set (delete it in scannetv2_train.txt).**

### ICL-NUIM

You can download the ICL-NUIM dataset from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html.

To prepare the data for training, testing, and evaluating, you can use our provided tools:

```
# preprocess the data.
tools/ICL-NUIM/preprocess_ICL.py
# generate the GT.
tools/ICL-NUIM/ICL_depth2mesh.py
```

 **NOTE: you need to set the save_root and data_root in these .py files.**

Or You can simply download the processed ICL-NUIM data from [Here](https://drive.google.com/file/d/1ypS_lcndytoNNngU3-r-7-F3ECTVwmFs/view?usp=drive_link).

### TUM-RGBD

You can download the TUM-RGBD dataset from https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download.

Following [VoRTX](https://github.com/noahstier/vortx/issues/3), 13 scenes are selected.

```
freiburg1_desk
freiburg1_plant
freiburg1_room
freiburg1_teddy
freiburg2_desk
freiburg2_dishes
freiburg2_large_no_loop
freiburg3_cabinet
freiburg3_long_office_household
freiburg3_nostructure_notexture_far
freiburg3_nostructure_texture_far
freiburg3_structure_notexture_far
freiburg3_structure_texture_far
```

To prepare the data for training, testing, and evaluating, you can use our provided tools:

```
# preprocess the data.
tools/TUM/preprocess_TUM.py
# generate the GT.
tools/TUM/TUM_depth2mesh.py
```

 **NOTE: you need to set the save_root and data_root in these .py files.**

Or You can simply download the processed TUM-RGBD data from [Here](https://drive.google.com/file/d/104YzM59v4TMkCJVr6ZjlU2JJreY3WbrM/view?usp=drive_link).

## Training

To train an IOAR model:

```
python scripts/train.py --config path/to/your_config.yml --gpu 0,
```

**NOTE: The comma after the GPU index is necessary.**

## Inference

After training, you can generate the meshes by:

```
python scripts/inference.py \
  --ckpt path/to/checkpoint.ckpt \
  --split [train / val / test] \
  --outputdir path/to/desired_output_directory \
  --n-imgs 60 \
  --config /path/to/your_config.yaml \
  --cropsize 64
```

For ICL-NUIM and TUM-RGBD datasets, use:

```
scripts/ICL/inference_ICL.py
scripts/TUM/inference_TUM_RGBD.py
```

**NOTE: remember to set the args.**

To reproduce the results, you can download the pre-trained models and inference on your own server.

| Datasets | BestCkptLink                                                 |
| -------- | ------------------------------------------------------------ |
| ScanNet  | https://drive.google.com/file/d/1Vg62XzzS2vecvCNllb2Frv7ZoqCYW8Di/view?usp=drive_link |
| TUM-RGBD | https://drive.google.com/file/d/1ei-bxswhTHF8S5FoV7i2byLbep_4kILB/view?usp=drive_link |
| ICL-NUIM | https://drive.google.com/file/d/1Cyy0S9RukT20_V5999qs730hXdlph6hK/view?usp=drive_link |

Using these models, you can easily reproduce the results on the paper (just like the following pictures).

### ScanNet

![ScanNetResult](https://cdn.statically.io/gh/YorkQiu/picx-images-hosting@master/20240114/image.2pjxkfdexro0.webp)

### ICL-NUIM

![ICLResults](https://cdn.statically.io/gh/YorkQiu/picx-images-hosting@master/20240114/image.430y6c8bsja0.webp)

### TUM-RGBD

![TUMResults](https://cdn.statically.io/gh/YorkQiu/picx-images-hosting@master/20240114/image.11kpr03yz28g.webp)

Or you can also simply download the results (i.e., meshes and evaluated metrics):

| Datasets | Results                                                      |
| -------- | ------------------------------------------------------------ |
| ScanNet  | https://drive.google.com/file/d/1w5ravgxSvrzNSKJMxta92C0V97r_ReJD/view?usp=drive_link |
| TUM-RGBD | https://drive.google.com/file/d/1XIwTlTPKHOyNAXFrlusua5rx44Jn36tV/view?usp=drive_link |
| ICL-NUIM | https://drive.google.com/file/d/11dEvVPSWaN6HJD_UHJfdQpcSOp9WwNjt/view?usp=drive_link |

Using these results, you can compare with our model easily.

## Evaluation

```
python scripts/evaluate.py \
  --results-dir path/to/inference_output_directory \
  --split [train / val / test] \
  --config /path/to/your_config.yaml
```

If you want to evaluate in another protocol, please follow [TransformerFusion](https://github.com/AljazBozic/TransformerFusion).

For ICL-NUIM and TUM-RGBD datasets, use:

```
scripts/ICL/evaluate_ICL.py

scripts/TUM/evaluate_TUM_RGBD.py
```

**NOTE: remember to set the args.**

## Related Works

The code of IOAR refers to several works:

[ATLAS: End-to-End 3D Scene Reconstruction from Posed Images](https://github.com/magicleap/Atlas)

[NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video](https://github.com/zju3dv/NeuralRecon)

[VoRTX: Volumetric 3D Reconstruction With Transformers for Voxelwise View Selection and Fusion](https://github.com/noahstier/vortx)

[TransformerFusion: Monocular RGB Scene Reconstruction using Transformers](https://github.com/AljazBozic/TransformerFusion)

Sincerely thanks for their excellent work!

## Citation

If you find our work helpful, please consider citing our paper:

```
@inproceedings{qiu2023inner,
  title={Inner-Outer Aware Reconstruction Model for Monocular 3D Scene Reconstruction},
  author={Qiu, Yu-Kun and Xu, Guohao and Zheng, Wei-Shi},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```