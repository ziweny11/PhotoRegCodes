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

#maskedl1_loss
def masked_l1_loss(img_new, img_ref, mask1, mask2):
    mask1 = mask1.unsqueeze(0)
    mask2 = mask2.unsqueeze(0) 
    masked_output = img_new * mask1 * mask2
    masked_gt = img_ref * mask1 * mask2
    diff = torch.abs(masked_output - masked_gt)
    sum_diff = torch.sum(diff)
    num_elements = torch.sum(mask1 * mask2)
    masked_l1_loss = sum_diff / num_elements
    return masked_l1_loss

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

def RTrandom(cam_dict):
    random_key = random.choice(list(cam_dict.keys()))
    random_img = cam_dict[random_key]
    print("name is", random_img.name)
    qvec, tvec = random_img.qvec, random_img.tvec
    rotmat = qvec2rotmat(qvec)
    rotmat = torch.tensor(rotmat, dtype=torch.float32, device="cuda")
    tvec = torch.tensor(tvec, dtype=torch.float32, device="cuda")
    return rotmat, tvec


def filtercam(g1, g2copy, extrinsics, dataset, pipe, threshold = 0.1, sim_threshold = 0.6, flag = 1):
    res = []
    for key in list(extrinsics.keys()):
        try:
            img = extrinsics[key]
            qvec, tvec = img.qvec, img.tvec
            rotmat = qvec2rotmat(qvec)
            rotmat = torch.tensor(rotmat, dtype=torch.float32, device="cuda")
            tvec = torch.tensor(tvec, dtype=torch.float32, device="cuda")
            rotmat = rotmat.T

            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            #set cam pos 
            fovx = 0.812657831303291
            fovy = 0.5579197285849142
            znear = 0.01
            zfar = 100.0
            scaling_modifer = 1.0
            pipe.convert_SHs_python = False 
            pipe.compute_cov3D_python = False

            cam_pos_right = myMiniCam3(979/2,652/2, rotmat, tvec, fovx, fovy, znear, zfar)

            cam_R_new = (RI1).T @ rotmat 
            cam_T_new = tvec - (TI1) @ (RI1).T @ rotmat
            cam_pos_new = myMiniCam3(979/2,652/2, cam_R_new, cam_T_new, fovx, fovy, znear, zfar)
            render_new = render(cam_pos_new, g2copy, pipe, background, scaling_modifer)["render"]
            img_new = render_new
            render_ref = render(cam_pos_right, g1, pipe, background, scaling_modifer)["render"]
            img_ref = render_ref
            dv1 = classify_image(img_new)
            dv2 = classify_image(img_ref)
            cos_sim = F.cosine_similarity(dv1, dv2, dim = 0)
            c1 = measure_blurriness(img_new) < threshold
            c2 = measure_blurriness(img_ref) < threshold
            c3 = cos_sim > sim_threshold
            if c1 and c2 and c3:
                print(img.name)
                res.append((rotmat, tvec))
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Skipping iteration {key} due to out of memory error.")
                torch.cuda.empty_cache()  # Optionally clear cache
                continue  # Skip this iteration
            else:
                raise e  # If 
    return res




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


    print(g1._xyz.shape)
    print(g2._xyz.shape)
    #ini
    scale = torch.tensor([10.0], requires_grad=True, device='cuda')
    g2scale = torch.tensor([1.0], requires_grad=True, device='cuda')
    offset = torch.tensor([0.0, 0.0, 0.0], requires_grad=True, device='cuda')

    rot = matrix_to_quaternion(R2)
    rot.to(dtype=torch.float32, device="cuda")
    rot.requires_grad_(True)

    best_param = (offset, rot, g2scale)

    lr_list = [0.001, 0.001, 0.00001]

    threshold = 0.3

    #set up extrinsincs cam dicts:
    path1 = 'playground/multi/multi_1'
    cameras_extrinsic_file1 = os.path.join(path1, "sparse/0", "images.bin")
    cam_extrinsics1 = read_extrinsics_binary(cameras_extrinsic_file1)
    path2 = 'playground/multi/multi_2'
    cameras_extrinsic_file2 = os.path.join(path2, "sparse/0", "images.bin")
    cam_extrinsics2 = read_extrinsics_binary(cameras_extrinsic_file2)
            
            
    g1copy = GaussianModel(dataset.sh_degree)
    g1copy.training_setup(opt)    
    g2copy = GaussianModel(dataset.sh_degree)
    g2copy.training_setup(opt)

    gaus_copy(g1, g1copy)
    gaus_copy(g2, g2copy)

    R_opt = quaternion_to_matrix(rot)

    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R_opt, scale * T2 + offset)
    rescale(g2copy, g2scale)

    cam_RT_list1 = filtercam(g1, g2copy, cam_extrinsics1, dataset, pipe, threshold = 0.5, 
                             sim_threshold = 0.5, flag = 1)
    print("len of usable RT_list:", len(cam_RT_list1))




    random.shuffle(cam_RT_list1)
    index = 0
    n = len(cam_RT_list1)

    for epoch in range(20000):

        #training initialization
        scale = torch.tensor([1.0], requires_grad=True, device='cuda')
        if epoch == 0:
            g2scale = torch.tensor([1.0], requires_grad=True, device='cuda')
            offset = torch.tensor( [0.0, 0.0, 0.0], requires_grad=True, device='cuda')
            rot = matrix_to_quaternion(R2)
            rot.to(dtype=torch.float32, device="cuda")
            rot.requires_grad_(True)
        else:
            offset, rot, g2scale = best_param

        lr_list =  [x * 1.0 for x in lr_list]
        
        optimizer = Adam([
        {'params': [offset], 'lr': lr_list[0]},
        {'params': [rot], 'lr': lr_list[1]},
        {'params': [g2scale], 'lr': lr_list[2]}
        ])

        #set up ref cams V1
        # cam_R_right2, cam_T_right2 = RTrandom(cam_extrinsics1)
        # cam_R_right2 = cam_R_right2.T
        # cam_R_right, cam_T_right = RTrandom(cam_extrinsics1)
        # cam_R_right = cam_R_right.T



        #set up ref cams V2
        if index + 2 > n:
                random.shuffle(cam_RT_list1)
                index = 0

        # Select two elements
        sample = cam_RT_list1[index:index + 2]
        index += 2
        cam_R_right2, cam_T_right2 = sample[0]
        cam_R_right, cam_T_right = sample[1]




        # optimizer = Adam([scale], lr=0.01)
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        
        best_loss = 100


        for iter in range(700):

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
            fovx = 0.812657831303291
            fovy = 0.5579197285849142
            znear = 0.01
            zfar = 100.0
            scaling_modifer = 1.0
            pipe.convert_SHs_python = False 
            pipe.compute_cov3D_python = False

            cam_pos_right2 = myMiniCam3(979/2,652/2, cam_R_right2, cam_T_right2, fovx, fovy, znear, zfar)

            # cam_R_new3 = (R2) @ (RI2).T @ cam_R_right2
            # cam_T_new3 = cam_T_right2 - ((TI2) @ ((R2).T) + 
            #                             (scale) * (T2) + offset) @ cam_R_new3

            cam_R_new3 = (RI1).T @ cam_R_right2 
            cam_T_new3 = cam_T_right2 - (TI1) @ (RI1).T @ cam_R_right2

            cam_pos_new2 = myMiniCam3(979/2,652/2, cam_R_new3, cam_T_new3, fovx, fovy, znear, zfar)
            render2_new = render(cam_pos_new2, g2copy, pipe, background, scaling_modifer)
            img_new2 = render2_new["render"]
            with torch.no_grad():
                render2_ref = render(cam_pos_right2, g1, pipe, background, scaling_modifer)
                img_ref2 = render2_ref["render"]
            
            

            weight = 0.0
            Ll12 = l1_loss(img_new2, img_ref2)
            loss2 = (1.0 - weight) * Ll12
            cam_pos_right = myMiniCam3(979/2,652/2, cam_R_right, cam_T_right, fovx, fovy, znear, zfar)

            cam_R_new = (RI1).T @ cam_R_right 
            cam_T_new = cam_T_right - (TI1) @ (RI1).T @ cam_R_right
            cam_pos_new = myMiniCam3(979/2,652/2, cam_R_new, cam_T_new, fovx, fovy, znear, zfar)
            render_new = render(cam_pos_new, g2copy, pipe, background, scaling_modifer)

            img_new = render_new["render"]
            with torch.no_grad():
                render_ref = render(cam_pos_right, g1, pipe, background, scaling_modifer)
                img_ref = render_ref["render"]


            Ll1 = l1_loss(img_new, img_ref)
            loss1 = (1.0 - weight) * Ll1
            loss = loss1 + loss2
            torch.autograd.set_detect_anomaly(True)
    

            loss.backward()
            optimizer.step()

            # if iter % 20 == 0:
            #     img2 = img_new.permute(1, 2, 0).cpu().detach().numpy()
            #     image = cv2.cvtColor(img2.astype('float32'), cv2.COLOR_RGB2BGR)

            #     img3 = img_ref.permute(1, 2, 0).cpu().detach().numpy()
            #     image3 = cv2.cvtColor(img3.astype('float32'), cv2.COLOR_RGB2BGR)
            #     concatenated_image = cv2.hconcat([image, image3])
            #     cv2.imshow('Combined Image', concatenated_image)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            #     img2 = img_new2.permute(1, 2, 0).cpu().detach().numpy()
            #     image = cv2.cvtColor(img2.astype('float32'), cv2.COLOR_RGB2BGR)

            #     img3 = img_ref2.permute(1, 2, 0).cpu().detach().numpy()
            #     image3 = cv2.cvtColor(img3.astype('float32'), cv2.COLOR_RGB2BGR)
            #     concatenated_image = cv2.hconcat([image, image3])
            #     cv2.imshow('Combined Image', concatenated_image)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

            if iter % 300 == 0:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                print(f'Iteration {iter}: Loss = {loss.item()}, Learning Rate = {current_lr}')

            update_rot = quaternion_to_matrix(rot).tolist()
            update_scale = scale.item()
            update_offset = offset.tolist()
            update_g2scale = g2scale.item()
            
            loss_val = loss.item()
            print(f'Iteration {iter}: Loss = {loss_val}, Updated offset = {update_offset}, {update_rot}', {update_g2scale}, {update_scale})
            if best_loss > loss_val:
                best_loss = loss_val
                best_param = (offset, rot, g2scale)
            elif loss_val > 1.01 * best_loss:
                break

            


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
        
        if epoch != 0:
            print("-----", epoch, "end, best loss is ", best_loss, offset, rot, g2scale)
        # if best_loss < 0.2:
        #     break



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
    RI1tmp = np.array([[ 0.5004,  0.3654, -0.7849],
        [-0.2768,  0.9265,  0.2548],
        [ 0.8204,  0.0898,  0.5648]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([-2.6533,  2.0358, -2.9062])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[ 0.9471, -0.1637,  0.2762],
        [ 0.1319,  0.9827,  0.1302],
        [-0.2928, -0.0869,  0.9522]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([-4.9053,  0.8034, -0.3393])
    TI2 = torch.tensor(TI2tmp, dtype=torch.float32, device="cuda")
    # TI1 = -torch.mm(RI1.t(), TI1.view(3,1)).view(1,3)
    # TI2 = -torch.mm(RI2.t(), TI2.view(3,1)).view(1,3)



    R0 = [[1,0,0],[0,1,0],[0,0,1]]
    T0 = [ 0, 0, 0]

    R1tmp = np.array(R0)
    T1tmp = np.array(T0)

# [[ 0.9633,  0.1462, -0.2250,  0.1422],
#          [-0.1374,  0.9890,  0.0542, -0.0315],
#          [ 0.2304, -0.0213,  0.9729, -0.0319]
    R2tmp2 = np.array( [[ 0.9971,  0.0166, -0.0741], 
                       [-0.0089,  0.9947,  0.1024], 
                       [ 0.0754, -0.1014,  0.9920]]).T
    R2tmp = np.array( [[ 0.9971,  0.0166, -0.0741], 
                       [-0.0089,  0.9947,  0.1024], 
                       [ 0.0754, -0.1014,  0.9920]]).T
    T2tmp = np.array([0.0279, -0.0073,0.0178]) @ R2tmp2

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



#depth dif: 1.284793734550476