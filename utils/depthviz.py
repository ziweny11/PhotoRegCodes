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

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    #pipe para temporarily set to false, modifier to 1.0
    pipe.convert_SHs_python = False 
    pipe.compute_cov3D_python = False
    scaling_modifer = 1.0
    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["depth"]
    weight_img = render(custom_cam, gaussians, pipe, background, scaling_modifer)["weight"]
    # net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
    return net_image, weight_img


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
    znear = 0.008999999612569809
    zfar = 1100.0


    #left example
    # R = np.array([[ 0.0788, -0.0589, 0.9951],
    #             [ -0.1201,  0.9915, 0.0492],
    #             [ -0.9896, -0.1157,  0.0852]])
    # T = np.array([0.7, 0.1853, 5.5])



    #right example


    def cam2cvimg(cam):
        net_img, wei_img = get_img(lp.extract(args), op.extract(args), pp.extract(args), cam, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
        img = net_img.cpu().detach().numpy()
        wei_img = wei_img.cpu().detach().numpy()
        # np.save('gsdepth.npy', img)
        # np.save('gsweight.npy', wei_img)
        depth_normalized = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)  # Convert to unsigned byte
        # image = cv2.cvtColor(depth_normalized.astype('float32'), cv2.COLOR_RGB2BGR)
        wei_normalized = cv2.normalize(wei_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        wei_normalized = wei_normalized.astype(np.uint8)
        return depth_normalized, wei_normalized

    # list of image
    # T0 = np.array([1, 0.5853, -1])
    # T1 = np.array([-2.5, 0.5853, 2.5])
    # for idx, trans in enumerate(straight_right_train(T0, T1, 40)):
    #     print(trans)
    #     cam = myMiniCam2(978, 543, R, trans, fovx, fovy, znear, zfar)
    #     img = cam2cvimg(cam)
    #     # cv2.imshow('Image', img)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     filename = f"image_{idx:03d}.png"
    #     output_dir = "./playground/img_split/rightview_seq/"
    #     file_path = os.path.join(output_dir, filename)
    #     if not os.path.exists(output_dir):
    #         print("bad")
    #         break
    #     success = cv2.imwrite(file_path, img * 255)
    #     if not success:
    #         print(f"Failed to write image to {file_path}")
    #     else:
    #         print(f"Image saved to {file_path}, img num {idx}")
    # print("complete")

    
    front = np.array([4.5248e-01, 0.4233,  2.8969e+00])
    right = np.array([6.5460e-01,  0.4233,  2.9536e+00])
    back = np.array([-0.3777,  0.4233,  3.1877])
    left = np.array([-0.5564,  0.4233,  3.1446])
    
    center = (left + right) / 2
    a1 = np.sqrt(np.sum(np.square(right - center)))
    a2 = np.sqrt(np.sum(np.square(left - center)))
    b1 = np.sqrt(np.sum(np.square(front - center)))
    b2 = np.sqrt(np.sum(np.square(back - center)))
    a = (a1 + a2)
    b = (b1 + b2)

    traj = ellipse_train(b, a, center, 20)


    R = np.array([[-0.8416,  0.2707, -0.4674],
                 [-0.0789,  0.7944,  0.6023],
                 [ 0.5343,  0.5437, -0.6472]])
    T = np.array([-1.7232, -0.5442,  1.8809])
    custom_cam3 = myMiniCam2(978, 543, R, T, fovx, fovy, znear, zfar)
    image, wei_img = cam2cvimg(custom_cam3)
    cv2.imwrite("./playground/img_split/rightview_seq/xxx.png", 255 * wei_img)
    cv2.imshow('Image', wei_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
















    # plt.imshow(img)
    # plt.title('Image from Tensor')
    # plt.axis('off')
    # plt.show()

    # folder_path = 'playground/imgs'
    # filename = f'{uuid.uuid4()}.png'
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    # file_path = os.path.join(folder_path, filename)
    # plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    # print("\nTraining complete.")