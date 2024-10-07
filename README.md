# PhotoReg Code Releases

Welcome to the official repository for PhotoReg, a project dedicated to the registration of 3D Gaussian Splatting models.

## Quick Links
- **Our Website:** [PhotoReg Project](https://ziweny11.github.io/photoreg/)
- **Our Paper on Arxiv:** *link to the paper*

## Setup Instructions

Follow these steps to set up and run the PhotoReg code on your local machine.

### 1. Clone the Necessary Repositories
First, clone the required repositories using the following commands:

- **3DGS Repository**
  ```
  git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
  ```
  
- **DUSt3R Repository**
  ```
  git clone --recursive https://github.com/naver/dust3r
  ```
  
- **PhotoReg Repository**
  ```
  git clone --recursive https://github.com/ziweny11/PhotoRegCodes
  ```

### 2. Install Dependencies
Navigate to the PhotoReg project directory and install the required Python packages:

```
cd PhotoRegCodes
pip install -r requirements.txt
```

### 3. Run the Code
To process multiple 3DGS models, run the following command:

```
python main.py --checkpoint_multi <path to trained model1> <path to trained model2>
```

### Visualization Tools
For visualizing the Gaussian Splatting and Point Clouds, use the following commands:

- **Gaussian Splatting Visualization**
  ```
  python gs_viz.py --checkpoint gnew.pth
  ```
  
- **Point Cloud Visualization**
  ```
  python PC_viz.py --checkpoint gnew.pth
  ```

## Support
For any issues or questions, please open an issue on this GitHub repository.
