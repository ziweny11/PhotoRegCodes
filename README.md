
# PhotoReg Code Releases

Welcome to the official repository for **PhotoReg**, a project dedicated to the registration of 3D Gaussian Splatting models.

## Quick Links
- **Website:** [Visit PhotoReg Project](https://ziweny11.github.io/photoreg/)
- **Paper on ArXiv:** [Read our paper](https://arxiv.org/abs/2410.05044)

## Plan

## Setup Instructions

### Clone the Necessary Repositories
Begin by cloning the required repositories. You'll need `gaussian-splatting` for preprocessing datasets and preparing input 3DGS models:
```
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```
Then, clone the PhotoReg repository:
```
git clone --recursive https://github.com/ziweny11/PhotoRegCodes
```

### Environment and Dependencies
Navigate to the PhotoReg project directory and set up the environment:
```
cd PhotoRegCodes
conda create -n photoreg python=3.8 cmake=3.14.0
conda activate photoreg
conda install pytorch torchvision torchaudio cuda-toolkit=11.8 -c pytorch -c nvidia
conda install -c conda-forge numpy plyfile tqdm matplotlib scipy einops trimesh tensorboard "pyglet<2" open3d pip
pip install roma opencv-python e3nn pytorch3d submodules/diff-gaussian-rasterization submodules/simple-knn
```
For the `diff-gaussian-rasterization-depth` package:
```
git clone https://github.com/leo-frank/diff-gaussian-rasterization-depth.git
cd diff-gaussian-rasterization-depth
python setup.py install
```
Download the pretrained model for DUSt3R [here](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth) and place it in the `PhotoRegCodes` folder. Learn more about DUSt3R on their [GitHub page](https://github.com/naver/dust3r).

### Running the Code

#### Demo
Download the example dataset from this [Google Drive link](#) and place it in the `PhotoRegCodes` folder.

Run the demo with:
```
python main.py -s data/truckright3 --start_checkpoint data/truckright3_model/chkpnt10000.pth --start_checkpoint2 data/truckleft3_model/chkpnt10000.pth --campose1 data/truckright3 --campose2 data/truckleft3
```

To visualize the final fused GS model (`gnew.pth`), first run:
```
python realtime_gsviz.py -s data/truckright3/ --start_checkpoint data/truckright3_model/chkpnt10000.pth
```
Then, use the SIBR viewer from `gaussian-splatting`:
```
./<SIBR install dir>/bin/SIBR_remoteGaussian_app
```
For installation details and more information, refer to the [gaussian-splatting README](https://github.com/graphdeco-inria/gaussian-splatting).

#### General Commands
Use the following command structure for general datasets:
```
python main.py -s path/to/source1 --start_checkpoint path/to/model/chkpntXXX.pth --start_checkpoint2 path/to/model2/chkpntXXX.pth --campose1 path/to/source1 --campose2 path/to/source2
```
`path/to/source` are paths to the <location> of dataset after running convert.py in 3DGS, where the <location> folder should have similar structure as below:
```
<location>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

`path/to/model/chkpntXXX.pth` are paths to the output 3DGS model. By default, it is in the output folder in gaussian-splatting after training, where the chkpntXXX.pth is stoed in folder 
structured like: 
```
<location>
|---point_cloud
|   |---...
|---cameras.json
|---cfg_args
|---input.ply
|---chkpntXXX.pth
```
You can visualize the result gnew.pth using SIBR the same as in demo.


## Citation
If you find our work useful, please consider citing:
```bibtex
@misc{yuan2024photoreg,
  title={PhotoReg: Photometrically Registering 3D Gaussian Splatting Models},
  author={Ziwen Yuan and Tianyi Zhang and Matthew Johnson-Roberson and Weiming Zhi},
  year={2024},
  eprint={2410.05044},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2410.05044}
}
