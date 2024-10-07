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
import cv2
from e3nn import o3
import einops
from einops import einsum
import kornia
from lietorch import SO3
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from rembg import remove
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
import random
from dinov2_utils import classify_image


from myUtils import measure_blurriness
from utils.general_utils import build_rotation
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.optim import Adam
import torch.nn.functional as F
import os
from scene.cameras import MiniCam, Camera, myMiniCam2, myMiniCam3
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

#maskedl1_loss
def masked_l1_loss(img_new, img_ref, mask):
    mask = mask.unsqueeze(0)  
    masked_output = img_new * mask
    masked_gt = img_ref * mask
    diff = torch.abs(masked_output - masked_gt)
    sum_diff = torch.sum(diff)
    num_elements = torch.sum(mask)
    masked_l1_loss = sum_diff / num_elements
    return masked_l1_loss


#pick suitable campos and images for dust3r input by dinov2 similarity
def camloop(g1, extrinsics1, dataset, pipe, prefix):
    
    #canvas set up
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    fovx = 0.812657831303291
    fovy = 0.5579197285849142
    znear = 0.01
    zfar = 100.0
    scaling_modifer = 1.0
    pipe.convert_SHs_python = False 
    pipe.compute_cov3D_python = False
    
    for key in list(extrinsics1.keys()):
        img = extrinsics1[key]
        qvec, tvec = img.qvec, img.tvec
        rotmat = qvec2rotmat(qvec)
        rotmat = torch.tensor(rotmat, dtype=torch.float32, device="cuda")
        tvec = torch.tensor(tvec, dtype=torch.float32, device="cuda")
        rotmat = rotmat.T
        cam_pos = myMiniCam3(980, 653, rotmat, tvec, fovx, fovy, znear, zfar)
        img1 = render(cam_pos, g1, pipe, background, scaling_modifer)["render"]
        npimg1 = tensor2np(img1)
        # if npimg1.dtype != np.uint8:
        #     npimg1 = npimg1.astype(np.uint8)
        # if npimg1.max() > 255 or npimg1.min() < 0:
        #     npimg1 = cv2.normalize(npimg1, None, 0, 255, cv2.NORM_MINMAX)
        # npimg1_bgr = cv2.cvtColor(npimg1, cv2.COLOR_RGB2BGR)
        filename = f'playground/colmapgen/{prefix}_{key:05}.jpg'
        result = cv2.normalize(npimg1, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(filename, result)

def tensor2np(img):
    img = img.permute(1,2,0).cpu().detach().numpy()
    image = cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2BGR)
    return image




def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint: str, checkpoint2: str, debug_from, 
             RI1, TI1, RI2, TI2, R1, T1, R2, T2):

    g1 = GaussianModel(dataset.sh_degree)
    g1.training_setup(opt)
    g2 = GaussianModel(dataset.sh_degree)  
    g2.training_setup(opt)
    scene = Scene(dataset, g1)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        g1.restore(model_params, opt)
    if checkpoint2:
        (model_params, first_iter) = torch.load(checkpoint2)
        g2.restore(model_params, opt)
        print("Second model loaded from checkpoint2.")
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float64, device="cuda")
    g1copy = GaussianModel(dataset.sh_degree)
    g1copy.training_setup(opt)    
    g2copy = GaussianModel(dataset.sh_degree)
    g2copy.training_setup(opt)
    gaus_copy(g1, g1copy)
    gaus_copy(g2, g2copy)

    #set up extrinsincs cam dicts:
    path1 = 'playground/wr_v2/wr1'
    cameras_extrinsic_file1 = os.path.join(path1, "sparse/0", "images.bin")
    cam_extrinsics1 = read_extrinsics_binary(cameras_extrinsic_file1)
    path2 = 'playground/wr_v2/wr2_high'
    cameras_extrinsic_file2 = os.path.join(path2, "sparse/0", "images.bin")
    cam_extrinsics2 = read_extrinsics_binary(cameras_extrinsic_file2)
    
    camloop(g1, cam_extrinsics1, dataset, pipe, 0)
    camloop(g2copy, cam_extrinsics2, dataset, pipe, 1)



    


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
    # # safe_state(args.quiet)
    # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)

    #trucktwin_right:
    # [[ 0.1714, -0.0919,  0.9809,  0.0000],
    #     [ 0.3924,  0.9196,  0.0176,  0.0000],
    #     [-0.9037,  0.3819,  0.1936,  0.0000],
    #     [ 0.2364, -0.1797,  4.4662,  1.0000]]
    RI1tmp = np.array([[-0.1238, -0.2341,  0.9643],
                  [0.2545,  0.9318,  0.2589],
                  [-0.9591,  0.2775, -0.0558]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([2.9596, -0.9691,  3.5393])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[0.1761,  0.1369, -0.9748],
                 [0.1506,  0.9749,  0.1642],
                 [0.9728, -0.1757,  0.1510]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([-2.0454, -0.1671,  3.9528])
    TI2 = torch.tensor(TI2tmp, dtype=torch.float32, device="cuda")
    # TI1 = -torch.mm(RI1.t(), TI1.view(3,1)).view(1,3)
    # TI2 = -torch.mm(RI2.t(), TI2.view(3,1)).view(1,3)



    R0 = [[1,0,0],[0,1,0],[0,0,1]]
    T0 = [ 0, 0, 0]

    R1tmp = np.array(R0)
    T1tmp = np.array(T0)

    R2tmp = np.array( [[0.9200, -0.2703,  0.2837], 
                       [0.2966,  0.9535, -0.0534], 
                       [-0.2561,  0.1333,  0.9574]]

)
    T2tmp = np.array([-0.1084, -0.0047, 0.0224])


    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")
