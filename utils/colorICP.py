#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import open3d as o3d
import cv2
from e3nn import o3
import einops
from einops import einsum
import kornia
from lietorch import SO3
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from rembg import remove
from PC_color import gau2colorpc
from utils.general_utils import build_rotation
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.optim import Adam
import torch.nn.functional as F
import os
from scene.cameras import MiniCam, Camera, myMiniCam2
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.optim.lr_scheduler import ExponentialLR
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


#utils:

def rgb_to_grayscale(img):
    """
    Convert a RGB image to grayscale.
    Parameters:
        img (Tensor): A PyTorch tensor of shape [C, H, W] or [B, C, H, W] where C = 3.
    Returns:
        Tensor: Grayscale image of shape [1, H, W] or [B, 1, H, W].
    """
    if img.dim() == 3:  # Single image [C, H, W]
        img = img.unsqueeze(0)  # Add batch dimension [1, C, H, W]
    # Weights for converting to grayscale. These values are typical weights for the RGB channels.
    weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=img.dtype, device=img.device)
    gray_img = torch.sum(img * weights[None, :, None, None], dim=1, keepdim=True)
    return gray_img.squeeze(0) if gray_img.size(0) == 1 else gray_img


def myloss(img1, img2):
    # Convert images to grayscale using the luminosity method
    # Since the image format is [channels, height, width], index with 0, 1, 2 directly
    gray1 = 0.299 * img1[0, :, :] + 0.587 * img1[1, :, :] + 0.114 * img1[2, :, :]
    gray2 = 0.299 * img2[0, :, :] + 0.587 * img2[1, :, :] + 0.114 * img2[2, :, :]

    # Compute MSE loss
    loss = F.mse_loss(gray1, gray2)
    return loss


def rot2q(R):
    # Ensure the matrix is suitable for conversion
    assert R.shape == (3, 3), "Rotation matrix must be 3x3"
    
    # Allocate space for the quaternion
    q = np.zeros(4)
    
    # Calculate the trace of the matrix
    tr = np.trace(R.cpu())
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S=4*qw
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q


def q2rot(q):
    # Normalize the quaternion to ensure it's a unit quaternion
    q = q / np.linalg.norm(q)
    
    # Extract components
    qw, qx, qy, qz = q
    
    # Compute the rotation matrix
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,         1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]
    ])
    
    return R



def xyztransform(pc, R, t, sc):
    res = torch.mm(pc, R.t())
    res = res + t
    return res * sc


def tensor_to_pcd(tensor):
    pcd = o3d.geometry.PointCloud()
    points = tensor.cpu().detach().numpy()  # Ensure tensor is on CPU and convert to numpy
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint: str, checkpoint2: str, debug_from, 
             RI1, TI1, RI2, TI2, R1, T1, R2, T2):

    g1 = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, g1)
    g1.training_setup(opt)


    g2 = GaussianModel(dataset.sh_degree)  
    g2.training_setup(opt)


    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        g1.restore(model_params, opt)
    
    if checkpoint2:
        (model_params, first_iter) = torch.load(checkpoint2)
        g2.restore(model_params, opt)
        print("Second model loaded from checkpoint2.")
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    g1copy = GaussianModel(dataset.sh_degree)
    g1copy.training_setup(opt)    
    g2copy = GaussianModel(dataset.sh_degree)
    g2copy.training_setup(opt)


    gaus_copy(g1, g1copy)
    gaus_copy(g2, g2copy)
    sc = torch.tensor([0.8261324763298035], dtype=torch.float32, device="cuda")
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    rescale(g2copy, sc)
    color1, pc1 = gau2colorpc(g1copy)
    color2, pc2 = gau2colorpc(g2copy)

    # pc2_align = xyztransform(pc2_g, R2, 2.2 * T2, sc3)
    voxel_size = 2

    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(pc1)
    source_pcd.colors = o3d.utility.Vector3dVector(color1)
    target_pcd.points = o3d.utility.Vector3dVector(pc2)
    target_pcd.colors = o3d.utility.Vector3dVector(color2)

    source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    source_pcd = source_pcd.voxel_down_sample(voxel_size)
    target_pcd = target_pcd.voxel_down_sample(voxel_size)

    o3d.visualization.draw_geometries([source_pcd, target_pcd])
    # Perform ICP
    threshold = 0.2  # set a suitable threshold for your application
    icp_result = o3d.pipelines.registration.registration_colored_icp(
        source_pcd, target_pcd, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))

    # Extract optimized R2, T2 from icp_result.transformation
    optimized_transformation = icp_result.transformation
    optimized_R2 = optimized_transformation[:3, :3]
    optimized_T2 = optimized_transformation[:3, 3]

    print(optimized_R2, optimized_T2)

    source_pcd.transform(icp_result.transformation)

    # Visualize the point clouds
    o3d.visualization.draw_geometries([source_pcd, target_pcd])



    return optimized_R2, optimized_T2
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--start_checkpoint2", type=str, default=None)  # New argument for the second checkpoint
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)


    R0 = [[1,0,0],[0,1,0],[0,0,1]]
    T0 = [ 0, 0, 0]

    R1tmp = np.array(R0)
    T1tmp = np.array(T0)


    RI1tmp = np.array([[ 0.9845, -0.1054,  0.1402],
        [ 0.0968,  0.9930,  0.0670],
        [-0.1463, -0.0524,  0.9879]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([-0.9779,  0.1564, -0.0101])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    RI2tmp =  np.array([[ 0.1627,  0.3500, -0.9225],
        [-0.1494,  0.9329,  0.3277],
        [ 0.9753,  0.0845,  0.2040]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([ 2.0282, -1.0095,  2.5417])
    TI2 = torch.tensor(TI2tmp, dtype=torch.float32, device="cuda")


    R2tmp = np.array( [[0.9560, -0.1253,  0.2654],
                        [0.1289,  0.9916,  0.0039],
                        [-0.2636,  0.0305,  0.9641]]

)
    T2tmp = np.array([-0.1618, 0.0286, -0.0588])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")
