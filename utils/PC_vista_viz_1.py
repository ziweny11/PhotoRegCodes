# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

import pyvista as pv
import cv2
from e3nn import o3
import einops
from einops import einsum
import kornia
from lietorch import SO3
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from rembg import remove
from utils.sh_utils import SH2RGB

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


# utils:

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


def gau2colorpc(g):
    shs = g.get_features[:,0,:]
    color = SH2RGB(shs)
    color = color.detach().cpu().numpy()
    pc1 = g._xyz
    pc1 = pc1.detach().cpu().numpy()
    return color, pc1


def get_camera_params():
    """Prompt the user to input camera position parameters."""
    print("Please enter the camera position parameters.")
    position = tuple(map(float, input("Camera Position (x, y, z): ").split()))
    focal_point = tuple(map(float, input("Focal Point (x, y, z): ").split()))
    view_up = tuple(map(float, input("View Up Direction (x, y, z): ").split()))
    return position, focal_point, view_up


def visualize_point_cloud(point_cloud, colors, camera_params=None):
    plotter = pv.Plotter()
    plotter.background_color = 'white'
    plotter.add_points(point_cloud, scalars=colors, rgb=True, render_points_as_spheres=True, point_size=4)
    
    if camera_params:
        plotter.camera_position = camera_params
    else:
        # Add key binding for 'p' to take a screenshot
        def take_screenshot():
            screenshot_path = "point_cloud_screenshot.png"
            plotter.screenshot(screenshot_path)
            print(f"Screenshot saved as '{screenshot_path}'")
        
        plotter.add_key_event("p", take_screenshot)
    
    plotter.show()


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint: str, debug_from):
    g1 = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, g1)
    g1.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        g1.restore(model_params, opt)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    g1copy = GaussianModel(dataset.sh_degree)
    g1copy.training_setup(opt)    

    gaus_copy(g1, g1copy)
    color, pc1 = gau2colorpc(g1)
    
    # Downsample point cloud

    # Visualize the point cloud
    camera_params = [(4.665472493784354, 3.700559417363724, -24.18179201622766),
 (-0.5448486888834937, -4.3599165689932144, 9.842314641058518),
 (-0.033542960076152505, -0.9713546360310882, -0.23525526750796014)]

    visualize_point_cloud(pc1, color, camera_params)


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
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
