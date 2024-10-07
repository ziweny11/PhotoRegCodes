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

import matplotlib.pyplot as plt
from myUtils import pos_gen, measure_blurriness
import numpy as np
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch
from scene.cameras import MiniCam, Camera, myMiniCam2
from gaussian_renderer import render
from traj_cam import straight_right_train, ellipse_train
import cv2
from scipy.spatial.transform import Rotation as R
def myCam(width, height, fovy, fovx, znear, zfar, wvt, fpt):
    world_view_transform = torch.tensor(wvt, dtype=torch.float32, device="cuda")
    full_proj_transform = torch.tensor(fpt, dtype=torch.float32, device="cuda")
    if torch.cuda.is_available():
        world_view_transform = world_view_transform.to('cuda:0')
        full_proj_transform = full_proj_transform.to('cuda:0')
    return MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
#get custom_cam from myCam
def get_img(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, custom_cam, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint: str, debug_from):
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    # gaussians.filter_y_larger_than_mean()
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    #pipe para temporarily set to false, modifier to 1.0
    pipe.convert_SHs_python = False 
    pipe.compute_cov3D_python = False
    scaling_modifer = 1.0
    renderes = render(custom_cam, gaussians, pipe, background, scaling_modifer)
    net_image = renderes["render"]
    masked_img = renderes["mask"]

    print("this is blur", measure_blurriness(net_image))
    return net_image, masked_img



def create_transparent_masked_image(image, mask):
    mask = (mask > 0).astype(np.float32)
    H, W, C = image.shape
    alpha_channel = mask * 255
    transparent_image = np.dstack((image, alpha_channel))
    return transparent_image

if __name__ == "__main__":
    # Set up command line argument parser
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
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    fovy = 0.8733320236206055
    fovx = 1.398024082183838
    znear = 0.01
    zfar = 100.0

    
    # fovx = 0.812657831303291
    # fovy = 0.5579197285849142
    # znear = 0.01
    # zfar = 100.0


    def cam2cvimg(cam):
        net_img, masked_img = get_img(lp.extract(args), op.extract(args), pp.extract(args), cam, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
        
        img = net_img.permute(1,2,0).cpu().detach().numpy()
        masked = masked_img.cpu().detach().numpy()
        image = cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2BGR)
        # aimage = create_transparent_masked_image(image, masked)
        return image



# 0.6440,  0.0697,  0.7618, -0.0000],
#         [ 0.3483,  0.8600, -0.3731, -0.0000],
#         [-0.6811,  0.5056,  0.5296,  0.0000],
#         [-8.4229, -1.5544,  1.7102,  1.0000]]

    T = np.array([-8.4229, -1.5544,  1.7102])

    # Initial and final rotation matrices
    R1 = np.array([[0.6440,  0.0697,  0.7618],
                [0.3483,  0.8600, -0.3731],
                [-0.6811,  0.5056,  0.5296]])
    R2 = np.array([[0.0646, 0.0697, -0.9925],
                [-0.5975,  1.8458, 0.0928],
                [0.8625,  0.5056,  -0.0205]])

    # Convert rotation matrices to quaternions
    quat1 = R.from_matrix(R1).as_quat()
    quat2 = R.from_matrix(R2).as_quat()

    # Create a list of interpolation steps (length of the list is 200)
    t_values = np.linspace(0, 1, 200)

    # Manual interpolation of quaternions (Slerp)
    def quaternion_slerp(q1, q2, t):
        # Compute quaternion dot product
        dot = np.dot(q1, q2)
        
        # If the dot product is negative, the quaternions have opposite handedness and the interpolation path should be reversed
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        
        # Clamp dot product to be within the range of acos()
        dot = np.clip(dot, -1.0, 1.0)
        
        # Calculate coefficients
        theta_0 = np.arccos(dot)  # theta_0 = angle between input vectors
        sin_theta_0 = np.sin(theta_0)
        
        # Avoid division by zero
        if sin_theta_0 < 1e-6:
            return q1
        
        theta = theta_0 * t  # theta = angle between v0 and result
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return (s0 * q1) + (s1 * q2)

    # Compute the interpolated quaternions
    interpolated_quats = np.array([quaternion_slerp(quat1, quat2, t) for t in t_values])

    # Convert quaternions back to rotation matrices
    R_matrices = R.from_quat(interpolated_quats).as_matrix()

    # Assuming myMiniCam2 and cam2cvimg are defined and work as expected
    for i, R in enumerate(R_matrices):
        # Setup camera with current rotation R and fixed translation T
        custom_cam3 = myMiniCam2(978, 543, R, T, fovx, fovy, znear, zfar)
        
        # Capture image using the camera settings
        image = cam2cvimg(custom_cam3)
        
        # Create filename
        filename = f"./playground/videoframe/image_{i+1:03d}.png"
        
        # Save the image
        cv2.imwrite(filename, 255 * image)

        # Optionally display the image (comment this out for faster processing)
        # cv2.imshow('Image', image)
        # cv2.waitKey(1)

    # Clean up any open windows (if you use cv2.imshow)
    cv2.destroyAllWindows()
