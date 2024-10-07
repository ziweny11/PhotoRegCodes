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
from scene.cameras import MiniCam, myMiniCam, Camera, myMiniCam2
from gaussian_renderer import render
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
    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
    return net_image


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
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    #def myCam(width, height, fovy, fovx, znear, zfar, wvt, fpt):



    # custom_cam = Camera(colmap_id = 0, R = np.empty((3,3)), T = np.array([0,0,0]), FoVx, FoVy, image, gt_alpha_mask,
    #              image_name, uid,
    #              trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
    #              ):
    fovy = 0.8733320236206055
    fovx = 1.398024082183838
    znear = 0.008999999612569809
    zfar = 1100.0
    custom_cam = myCam(978, 543, fovy, fovx,
                       znear, zfar,
                       np.array([[ 0.0788, -0.0589, -0.9951, -0.0000],
                                [ 0.1201,  0.9915, -0.0492,  0.0000],
                                [ 0.9896, -0.1157,  0.0852,  0.0000],
                                [-1.9372,  0.5853,  5.2759,  1.0000]]), 
                       np.array([[ 0.0937, -0.1263, -0.9952, -0.9951],
                                [ 0.1429,  2.1245, -0.0492, -0.0492],
                                [ 1.1773, -0.2478,  0.0852,  0.0852],
                                [-1.5908,  1.2541,  5.2292,  5.2471]]))
    #(self, width, height, R, T, fovx, fovy, znear, zfar, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):

    R = np.array([[ 0.0788, -0.0589, -0.9951],
                [ 0.1201,  0.9915, -0.0492],
                [ 0.9896, -0.1157,  0.0852]])
    camtype = np.array([
        [1.18967140e+00, 1.03320887e-04, -7.87922931e-06, 1.65189563e-16],
        [2.28726582e-05, 2.14270567e+00, 5.89275209e-06, 3.00177144e-17],
        [4.49792555e-05, 1.03283759e-04, 1.00009952e+00, 1.00000000e+00],
        [7.13580736e-01, -3.70390753e-04, -4.72437686e-02, -2.88000000e-02]])
    custom_cam3 = myMiniCam2(978, 543, R, np.array([-1.9372, 0.5853, 5.2759]), fovx, fovy, znear, zfar)
    custom_cam2 = myMiniCam(978, 543, R, np.array([-1.9372,  0.5853,  5.2759]), fovx, fovy, znear, zfar, camtype)
    custom_cam.printinfo()
    custom_cam3.printinfo()
    net_img = get_img(lp.extract(args), op.extract(args), pp.extract(args), custom_cam3, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    img = net_img.permute(1,2,0).cpu().detach().numpy()

    plt.imshow(img)
    plt.title('Image from Tensor')
    plt.axis('off')
    plt.show()

    folder_path = 'playground/imgs'
    filename = f'{uuid.uuid4()}.png'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, filename)
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    print("\nTraining complete.")