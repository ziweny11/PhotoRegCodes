# Code is updating, current version facing issues
# PhotoReg Code Releases

Welcome to the official repository for PhotoReg, a project dedicated to the registration of 3D Gaussian Splatting models.

## Quick Links
- **Our Website:** [PhotoReg Project](https://ziweny11.github.io/photoreg/)
- **Our Paper on Arxiv:** [*link to the paper*](https://arxiv.org/abs/2410.05044)


## Plan
- [ ] Solve all issues
- [ ] Provide toy datasets
- [ ] Update visualization and evaluation
## Setup Instructions

Follow these steps to set up and run the PhotoReg code on your local machine. First, 3DGS repo https://github.com/graphdeco-inria/gaussian-splatting is needed.

### 1. Clone the Necessary Repositories
First, clone the required repositories using the following commands. gaussian-splatting is needed for preprocessing datesets and trainining input 3DGS models
  ```
  git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
  ```
  ```
  git clone --recursive https://github.com/ziweny11/PhotoRegCodes
  ```

### 2. Install Dependencies
Navigate to the PhotoReg project directory and install the required Python packages:

```
cd PhotoRegCodes
conda env create --file environment.yaml
conda activate photoreg
```

### 3. Run the Code
To process multiple 3DGS models, run the following command:

```
python main.py -s path/to/source1 --start_checkpoint path/to/model/chkpntXXX.pth --start_checkpoint2 path/to/model2/chkpntXXX.pth --campose1 path/to/source1 --campose2 path/to/source2

```
path/to/source are paths to the <location> of dataset after running convert.py in 3DGS, where the <location> folder should have similar structure as below:
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

path/to/model/chkpntXXX.pth are paths to the output 3DGS model. By default, it is in the output folder in gaussian-splatting after training, where the chkpntXXX.pth is stoed in folder structured like: 
```
<location>
|---point_cloud
|   |---...
|---cameras.json
|---cfg_args
|---input.ply
|---chkpntXXX.pth

```
### Visualization Tools
For visualizing the Gaussian Splatting and Point Clouds, use the following commands:

- **Gaussian Splatting Visualization**
  ```
  to be updated
  ```
  
- **Point Cloud Visualization**
  ```
  to be updated
  ```

### Evaluation
  ```
  to be updated
  ```

## Citation
If you find our work useful in your research, please consider citing:

```bibtex
@misc{yuan2024photoregphotometricallyregistering3d,
title={PhotoReg: Photometrically Registering 3D Gaussian Splatting Models}, 
author={Ziwen Yuan and Tianyi Zhang and Matthew Johnson-Roberson and Weiming Zhi},
year={2024},
eprint={2410.05044},
archivePrefix={arXiv},
primaryClass={cs.RO},
url={https://arxiv.org/abs/2410.05044}, 
}
```
## Support
For any issues or questions, please open an issue on this GitHub repository.
