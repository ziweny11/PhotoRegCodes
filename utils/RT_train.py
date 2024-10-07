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

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)


    scale = torch.tensor([15.0], requires_grad=True, device='cuda')
    g2scale = torch.tensor([1.0], requires_grad=True, device='cuda')
    offset = torch.tensor([0.0, 0.0, 0.0], requires_grad=True, device='cuda')

    rot = matrix_to_quaternion(R2)

    rot.to(dtype=torch.float32, device="cuda")
    rot.requires_grad_(True)
    
    optimizer = Adam([
    {'params': [offset], 'lr': 0.001},  # Learning rate for 'offset'
    {'params': [rot], 'lr': 0.0001},    # Different learning rate for 'rot'
    {'params': [g2scale], 'lr': 0.00001} # Another different learning rate for 'g2scale'
    ])

   
    
    # optimizer = Adam([scale], lr=0.01)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    for iter in range(1000):

        optimizer.zero_grad()
        gnew = GaussianModel(dataset.sh_degree)
        gnew.training_setup(opt)


        g1copy = GaussianModel(dataset.sh_degree)
        g1copy.training_setup(opt)    
        g2copy = GaussianModel(dataset.sh_degree)
        g2copy.training_setup(opt)
        gnew = GaussianModel(dataset.sh_degree)
        gnew.training_setup(opt)

        gaus_copy(g1, g1copy)
        gaus_copy(g2, g2copy)

        R_opt = quaternion_to_matrix(rot)

        gaus_transform(g1copy, RI1.t(), TI1)
        gaus_transform(g1copy, R1, T1)
        gaus_transform(g2copy, RI2.t(), TI2)
        gaus_transform(g2copy, R_opt, scale * T2 + offset)
        rescale(g2copy, g2scale)
        gaus_append(g1copy, g2copy, gnew)
        #gnew is the new gaussian model

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        #set cam pos 
        fovy = 0.8733320236206055
        fovx = 1.398024082183838
        znear = 0.008999999612569809
        zfar = 1100.0
        scaling_modifer = 1.0
        pipe.convert_SHs_python = False 
        pipe.compute_cov3D_python = False



# [[-0.2868, -0.3123,  0.9057,  0.0000],
#         [ 0.3545,  0.8436,  0.4032,  0.0000],
#         [-0.8900,  0.4367, -0.1312, -0.0000],
#         [ 2.4494, -1.7586,  3.1399,  1.0000]]

# [[ 0.4878,  0.1310, -0.8631,  0.0000],
#         [-0.2511,  0.9679,  0.0050,  0.0000],
#         [ 0.8360,  0.2143,  0.5051,  0.0000],
#         [-2.4914, -0.6911,  4.1806,  1.0000]]

# [[ 0.6353, -0.0665, -0.7694, -0.0000],
#         [-0.2271,  0.9361, -0.2685,  0.0000],
#         [ 0.7381,  0.3453,  0.5797,  0.0000],
#         [-2.9389, -0.5036,  3.4852,  1.0000]

# [[-0.4645,  0.0105, -0.8855, -0.0000],
#         [-0.0820,  0.9951,  0.0548,  0.0000],
#         [ 0.8818,  0.0981, -0.4614, -0.0000],
#         [-1.6066,  0.3750,  4.7704,  1.0000]]


# [[ 0.2257,  0.0760,  0.9712, -0.0000],
#         [ 0.0169,  0.9965, -0.0819, -0.0000],
#         [-0.9740,  0.0349,  0.2237,  0.0000],
#         [ 1.6356,  0.6640,  6.9358,  1.0000]]

# [[ 0.9171, -0.1624,  0.3641, -0.0000],
#         [ 0.1199,  0.9833,  0.1366,  0.0000],
#         [-0.3802, -0.0816,  0.9213,  0.0000],
#         [ 3.2764, -0.4214,  3.3093,  1.0000]]

# [[-0.2397,  0.6231, -0.7445,  0.0000],
#         [-0.5232,  0.5630,  0.6397,  0.0000],
#         [ 0.8178,  0.5429,  0.1911,  0.0000],
#         [-1.5881,  1.5815, -1.7562,  1.0000]]

# [[ 0.9034,  0.2393, -0.3560, -0.0000],
#         [ 0.0280,  0.7953,  0.6056,  0.0000],
#         [ 0.4280, -0.5570,  0.7117, -0.0000],
#         [ 4.6093, -1.7971, -1.8927,  1.0000]]

# -0.5270,  0.2229, -0.8201, -0.0000],
#         [-0.4209,  0.7699,  0.4797,  0.0000],
#         [ 0.7383,  0.5980, -0.3119, -0.0000],
#         [ 1.4479, -0.9426, -1.1789,  1.0000]

# [[ 0.9473,  0.0198, -0.3198, -0.0000],
#         [-0.0672,  0.9881, -0.1380,  0.0000],
#         [ 0.3133,  0.1522,  0.9374,  0.0000],
#         [-0.2253,  0.3500, -0.3742,  1.0000]]
        cam_R_right2_np = np.array([[0.9473,  0.0198, -0.3198],
                    [-0.0672,  0.9881, -0.1380],
                    [0.3133,  0.1522,  0.9374]])
        cam_R_right2 = torch.tensor(cam_R_right2_np, dtype=torch.float32, device="cuda")
        cam_T_right2_np = np.array([-0.2253,  0.3500, -0.3742]) 
        cam_T_right2 = torch.tensor(cam_T_right2_np, dtype=torch.float32, device="cuda")
        cam_pos_right2 = myMiniCam3(450, 250, cam_R_right2, cam_T_right2, fovx, fovy, znear, zfar)

        cam_R_new3 = (R2) @ (RI2).T @ cam_R_right2
        cam_T_new3 = cam_T_right2 - ((TI2) @ ((R2).T) + 
                                     (scale) * (T2) + offset) @ cam_R_new3

        cam_pos_new2 = myMiniCam3(450, 250, cam_R_new3, cam_T_new3, fovx, fovy, znear, zfar)
        img_new2 = render(cam_pos_new2, gnew, pipe, background, scaling_modifer)["render"]
        with torch.no_grad():
            img_ref2 = render(cam_pos_right2, g2, pipe, background, scaling_modifer)["render"]

        weight = 0.0
        Ll12 = l1_loss(img_new2, img_ref2)
        loss2 = (1.0 - weight) * Ll12 + weight * (1.0 - ssim(img_new2, img_ref2))


        # [[ 0.8065, -0.1051,  0.5818, -0.0000],
        # [-0.1066,  0.9421,  0.3180,  0.0000],
        # [-0.5816, -0.3185,  0.7486, -0.0000],
        # [ 2.7798,  1.0441,  0.1016,  1.0000]]

        # [[ 0.2295, -0.3270,  0.9167,  0.0000],
        # [ 0.0976,  0.9449,  0.3126,  0.0000],
        # [-0.9684,  0.0177,  0.2488,  0.0000],
        # [ 2.4433, -0.1978,  2.7014,  1.0000]]

        # [[ 0.2930, -0.5385,  0.7900,  0.0000],
        # [ 0.4357,  0.8107,  0.3910,  0.0000],
        # [-0.8511,  0.2296,  0.4722,  0.0000],
        # [ 2.9577, -1.5154,  1.1664,  1.0000]

        # [[ 0.5434,  0.2186, -0.8105,  0.0000],
        # [-0.3226,  0.9457,  0.0388,  0.0000],
        # [ 0.7750,  0.2404,  0.5845,  0.0000],
        # [-2.2661, -1.1410,  4.1233,  1.0000]]

# [[-0.1944, -0.3995,  0.8959,  0.0000],
#         [ 0.4236,  0.7896,  0.4440, -0.0000],
#         [-0.8847,  0.4658,  0.0157,  0.0000],
#         [ 2.5998, -1.9005,  2.8667,  1.0000]]

# [[ 0.9262,  0.0198,  0.3765, -0.0000],
#         [ 0.0392,  0.9881, -0.1484, -0.0000],
#         [-0.3749,  0.1522,  0.9145,  0.0000],
#         [ 3.7123,  0.3195,  3.9183,  1.0000]]

# [[ 0.9884,  0.0198, -0.1505, -0.0000],
#         [-0.0422,  0.9881, -0.1476,  0.0000],
#         [ 0.1458,  0.1522,  0.9775,  0.0000],
#         [ 1.1775,  0.3195,  2.0914,  1.0000]]

# [[ 0.3971,  0.0198,  0.9175, -0.0000],
#         [ 0.1320,  0.9881, -0.0784, -0.0000],
#         [-0.9082,  0.1522,  0.3898,  0.0000],
#         [ 2.5047, -0.1374,  6.7945,  1.0000]]

#pr1
# [[ 0.3456,  0.5720, -0.7439,  0.0000],
#         [-0.6501,  0.7176,  0.2498,  0.0000],
#         [ 0.6767,  0.3973,  0.6198,  0.0000],
#         [ 2.8357,  1.7231,  0.4987,  1.0000]]
        cam_R_right = np.array([[ 0.3456,  0.5720, -0.7439],
                 [-0.6501,  0.7176,  0.2498],
                 [ 0.6767,  0.3973,  0.6198]])
        cam_T_right = np.array([2.8357,  1.7231,  0.4987])
        cam_pos_right = myMiniCam2(450, 250, cam_R_right, cam_T_right, fovx, fovy, znear, zfar)

        cam_R_new = (RI1.cpu().numpy()).T @ cam_R_right 
        cam_T_new = cam_T_right - (TI1.cpu().numpy()) @ (RI1.cpu().numpy()).T @ cam_R_right
        cam_pos_new = myMiniCam2(450, 250, cam_R_new, cam_T_new, fovx, fovy, znear, zfar)
        img_new = render(cam_pos_new, gnew, pipe, background, scaling_modifer)["render"]
        with torch.no_grad():
            img_ref = render(cam_pos_right, g1, pipe, background, scaling_modifer)["render"]



        # img2 = img_new2.permute(1,2,0).cpu().detach().numpy()
        # image = cv2.cvtColor(img2.astype('float32'), cv2.COLOR_RGB2BGR)
        # cv2.imshow('Image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



        # img_new = rgb_to_grayscale(img_new)
        # img_ref = rgb_to_grayscale(img_ref)
        Ll1 = l1_loss(img_new, img_ref)
        loss1 = (1.0 - weight) * Ll1 + weight * (1.0 - ssim(img_new, img_ref))

        loss = loss1 + loss2
        torch.autograd.set_detect_anomaly(True)
        # loss = myloss(img_new, img_ref)
        loss.backward()        # Compute gradients

        optimizer.step()       # Update the parameter
        if iter % 100 == 0:  # Update learning rate every 20 iterations
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
            print(f'Iteration {iter}: Loss = {loss.item()}, Learning Rate = {current_lr}')



        update_rot = quaternion_to_matrix(rot).tolist()
        update_scale = scale.item()
        update_offset = offset.tolist()
        update_g2scale = g2scale.item()
        print(f'Iteration {iter}: Loss = {loss.item()}, Updated offset = {update_offset}, {update_rot}', {update_g2scale}, {update_scale})


        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                # custom_cam.printinfo()
                if custom_cam != None:
                    net_image = render(custom_cam, gnew, pipe, background, scaling_modifer)["render"]
                    # #check type
                    # print("type of custom_cam:")
                    # print(type(custom_cam), end="\n")
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iter < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None







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
    RI1tmp = np.array([[-0.8416,  0.2707, -0.4674],
                 [-0.0789,  0.7944,  0.6023],
                 [ 0.5343,  0.5437, -0.6472]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([-1.7232, -0.5442,  1.8809])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[0.9907, -0.0350,  0.1314],
                 [0.0371,  0.9992, -0.0135],
                 [-0.1309,  0.0182,  0.9912]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([0.2722, -0.1554, -2.1680])
    TI2 = torch.tensor(TI2tmp, dtype=torch.float32, device="cuda")
    # TI1 = -torch.mm(RI1.t(), TI1.view(3,1)).view(1,3)
    # TI2 = -torch.mm(RI2.t(), TI2.view(3,1)).view(1,3)



    R0 = [[1,0,0],[0,1,0],[0,0,1]]
    T0 = [ 0, 0, 0]

    R1tmp = np.array(R0)
    T1tmp = np.array(T0)

    # [[ 0.4986, -0.2385,  0.8334, -0.1995],
    #      [ 0.2861,  0.9528,  0.1016,  0.0221],
    #      [-0.8183,  0.1878,  0.5433,  0.0976],
    #      [ 0.0000,  0.0000,  0.0000,  1.0000]]])
    R2tmp = np.array( [[0.9646,  0.0270,  0.2624],
                        [-0.0223,  0.9995, -0.0210],
                          [-0.2629,  0.0144,  0.9647]]

)
    T2tmp = np.array([-0.0495,-0.0072,0.1238])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")







#g1, g2, scale => gnew
#pick a campos
#i1 = imggen(gnew, campos)
#i2 = imggen(g1 or g2, campos)
#loss function(i1, i2)
#min loss