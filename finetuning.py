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

from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary
import random


from myUtils import measure_blurriness
import numpy as np
from torch.optim import Adam
import os
from scene.cameras import myMiniCam3
from myUtils import gaus_transform, gaus_copy, rescale
import torch
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.optim.lr_scheduler import ExponentialLR
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def masked_l1_loss(img_new, img_ref, mask):
    mask = mask.unsqueeze(0)  
    masked_output = img_new * mask
    masked_gt = img_ref * mask
    diff = torch.abs(masked_output - masked_gt)
    sum_diff = torch.sum(diff)
    num_elements = torch.sum(mask)
    masked_l1_loss = sum_diff / num_elements
    return masked_l1_loss

def maskgen(x, y, k):

    total_elements = x * y
    num_ones = int(total_elements * k)
    
    mask = torch.zeros(total_elements, dtype=torch.float32, device="cuda", requires_grad= False)
    indices = torch.randperm(total_elements)[:num_ones]
    mask[indices] = 1
    mask = mask.reshape(x, y)
    return mask

def RTrandom(cam_dict):
    random_key = random.choice(list(cam_dict.keys()))
    random_img = cam_dict[random_key]
    print("name is", random_img.name)
    qvec, tvec = random_img.qvec, random_img.tvec
    rotmat = qvec2rotmat(qvec)
    rotmat = torch.tensor(rotmat, dtype=torch.float32, device="cuda")
    tvec = torch.tensor(tvec, dtype=torch.float32, device="cuda")
    return rotmat, tvec

def finetuning(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint: str, checkpoint2: str, debug_from, 
             RI1, TI1, RI2, TI2, R1, T1, R2, T2, campose1, campose2):


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


    #set up extrinsincs cam dicts:
    path1 = campose1
    cameras_extrinsic_file1 = os.path.join(path1, "sparse/0", "images.bin")
    cam_extrinsics1 = read_extrinsics_binary(cameras_extrinsic_file1)
    path2 = campose2
    cameras_extrinsic_file2 = os.path.join(path2, "sparse/0", "images.bin")
    cam_extrinsics2 = read_extrinsics_binary(cameras_extrinsic_file2)

    #training initialization
    Sscale = torch.tensor([1.0], requires_grad=True, device='cuda')
    offset = torch.tensor([0.0, 0.0, 0.0], requires_grad=True, device='cuda')
    rot = matrix_to_quaternion(R2)
    rot.to(dtype=torch.float32, device="cuda")
    rot.requires_grad_(True)
    best_param = (offset, rot, Sscale)
    lr_list = [0.01, 0.001, 0.01]
    Tscale = torch.tensor([12.2], requires_grad=True, device='cuda')


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

    for epoch in tqdm(range(200), desc="Epochs"):
        if epoch == 0:
            Sscale = torch.tensor([1.0], requires_grad=True, device='cuda')
            offset = torch.tensor([0.0, 0.0, 0.0], requires_grad=True, device='cuda')
            rot = matrix_to_quaternion(R2)
            rot.to(dtype=torch.float32, device="cuda")
            rot.requires_grad_(True)
        else:
            offset, rot, Sscale = best_param

        lr_list =  [x * 1.0 for x in lr_list]
        
        optimizer = Adam([
        {'params': [offset], 'lr': lr_list[0]},
        {'params': [rot], 'lr': lr_list[1]},
        {'params': [Sscale], 'lr': lr_list[2]}
        ])

        #set up ref cams
        cam_R_right2, cam_T_right2 = RTrandom(cam_extrinsics2)
        cam_R_right2 = cam_R_right2.T
        cam_R_right, cam_T_right = RTrandom(cam_extrinsics1)
        cam_R_right = cam_R_right.T

        mask = maskgen(250, 450, 0.5)

        scheduler = ExponentialLR(optimizer, gamma=0.95)
        
        best_loss = 100


        for iter in tqdm(range(10000), desc="Training Iterations", leave=False):

            optimizer.zero_grad()
            R_opt = quaternion_to_matrix(rot)
            gaus_transform(g1copy, RI1.t(), TI1)
            gaus_transform(g1copy, R1, T1)
            gaus_transform(g2copy, RI2.t(), TI2)
            gaus_transform(g2copy, R_opt, Tscale * T2 + offset)
            rescale(g2copy, Sscale)

            cam_pos_right2 = myMiniCam3(450, 250, cam_R_right2, cam_T_right2, fovx, fovy, znear, zfar)

            cam_R_new3 = (R2) @ (RI2).T @ cam_R_right2
            cam_T_new3 = cam_T_right2 - ((TI2) @ ((R2).T) + 
                                        (Tscale) * (T2) + offset) @ cam_R_new3

            cam_pos_new2 = myMiniCam3(450, 250, cam_R_new3, cam_T_new3, fovx, fovy, znear, zfar)
            render2_new = render(cam_pos_new2, g2copy, pipe, background, scaling_modifer)["render"]
            img_new2 = render2_new
            with torch.no_grad():
                if measure_blurriness(img_new2) > 0.1:
                    break
                render2_ref = render(cam_pos_right2, g2, pipe, background, scaling_modifer)
                img_ref2 = render2_ref["render"]
                if measure_blurriness(img_ref2) > 0.1:
                    break


            weight = 0.0
            Ll12 = masked_l1_loss(img_new2, img_ref2, mask)
            loss2 = (1.0 - weight) * Ll12

            cam_pos_right = myMiniCam3(450, 250, cam_R_right, cam_T_right, fovx, fovy, znear, zfar)

            cam_R_new = (RI1).T @ cam_R_right 
            cam_T_new = cam_T_right - (TI1) @ (RI1).T @ cam_R_right
            cam_pos_new = myMiniCam3(450, 250, cam_R_new, cam_T_new, fovx, fovy, znear, zfar)
            render_new = render(cam_pos_new, g2copy, pipe, background, scaling_modifer)["render"]

            img_new = render_new
            with torch.no_grad():
                if measure_blurriness(img_new) > 0.1:
                    break
                render_ref = render(cam_pos_right, g1, pipe, background, scaling_modifer)
                img_ref = render_ref["render"]
                if measure_blurriness(img_ref) > 0.1:
                    break

            Ll1 = masked_l1_loss(img_new, img_ref, mask)
            loss1 = (1.0 - weight) * Ll1

            loss = loss1 + loss2
            torch.autograd.set_detect_anomaly(True)
            
            
            loss.backward()
            optimizer.step()

            
            if iter % 100 == 0:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                # print(f'Iteration {iter}: Loss = {loss.item()}, Learning Rate = {current_lr}')

            # update_rot = quaternion_to_matrix(rot).tolist()
            # update_scale = Tscale.item()
            # update_offset = offset.tolist()
            # update_g2scale = Sscale.item()
            
            loss_val = loss.item()
            # print(f'Iteration {iter}: Loss = {loss_val}, Updated offset = {update_offset}, {update_rot}', {update_g2scale}, {update_scale})
            
            if best_loss > loss_val:
                best_loss = loss_val
                best_param = (offset, rot, Sscale)
            elif loss_val > 1.05 * best_loss:
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
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iter < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None
        
    return best_loss, offset, rot, Sscale, Tscale