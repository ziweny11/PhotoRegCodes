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
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary
from utils.loss_utils import ssim
import random
from dinov2_utils import classify_image
from myUtils import measure_blurriness
import numpy as np
import torch.nn.functional as F
import os
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from myUtils import measure_blurriness, gaus_append
import numpy as np
from torch.optim import Adam
import os
from scene.cameras import myMiniCam3
from myUtils import gaus_transform, gaus_copy, rescale
import torch
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

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

def filtercam(g1, g2copy, RI1, TI1, extrinsics, dataset, pipe, threshold = 0.1, sim_threshold = 0.6, flag = 1):
    res = []
    for key in list(extrinsics.keys()):
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

        cam_pos_right = myMiniCam3(450, 250, rotmat, tvec, fovx, fovy, znear, zfar)

        cam_R_new = (RI1).T @ rotmat 
        cam_T_new = tvec - (TI1) @ (RI1).T @ rotmat
        cam_pos_new = myMiniCam3(450, 250, cam_R_new, cam_T_new, fovx, fovy, znear, zfar)
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
    return res

def finetuning(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint: str, checkpoint2: str, debug_from, 
             RI1, TI1, RI2, TI2, R1, T1, R2, T2, campose1, campose2, ep = 100, it = 1000, lr_update_iter = 200, early_stop_threshold = 20, show_images = False):

    torch.autograd.set_detect_anomaly(True)
    
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

    #ini
    Tscale = torch.tensor([1.0], requires_grad=True, device='cuda')
    Sscale = torch.tensor([1.0], requires_grad=True, device='cuda')
    offset = torch.tensor([0.0, 0.0, 0.0], requires_grad=True, device='cuda')

    rot = matrix_to_quaternion(R2)
    rot.to(dtype=torch.float32, device="cuda")
    rot.requires_grad_(True)

    best_param = (Tscale, rot, Sscale, offset)

    lr_list = [0.001, 0.0001, 0.00001, 0.001]

    #set up extrinsincs cam dicts:
    path1 = campose1
    cameras_extrinsic_file1 = os.path.join(path1, "sparse/0", "images.bin")
    cam_extrinsics1 = read_extrinsics_binary(cameras_extrinsic_file1)
    path2 = campose2
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
    gaus_transform(g2copy, R_opt, Tscale * T2 + offset)
    rescale(g2copy, Sscale)

    cam_RT_list1 = filtercam(g1, g2copy, RI1, TI1, cam_extrinsics1, dataset, pipe, threshold = 0.7, 
                             sim_threshold = 0.5, flag = 1)
    print("len of usable RT_list:", len(cam_RT_list1))

    for epoch in tqdm(range(ep), desc="Epochs"):

        #training initialization
        if epoch == 0:
            Sscale = torch.tensor([1.0], requires_grad=True, device='cuda')
            Tscale = torch.tensor([1.0], requires_grad=True, device='cuda')
            rot = matrix_to_quaternion(R2)
            rot.to(dtype=torch.float32, device="cuda")
            rot.requires_grad_(True)
        else:
            Tscale, rot, Sscale, offset = best_param

        lr_list =  [x * 1.0 for x in lr_list]
        
        optimizer = Adam([
        {'params': [Tscale], 'lr': lr_list[0]},
        {'params': [rot], 'lr': lr_list[1]},
        {'params': [Sscale], 'lr': lr_list[2]},
        {'params': [offset], 'lr': lr_list[3]}
        ])

        #set up ref cams V2
        sample = random.sample(cam_RT_list1, 2)
        cam_R_right2, cam_T_right2 = sample[0]
        cam_R_right, cam_T_right = sample[1]

        # optimizer = Adam([Tscale], lr=0.01)
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        
        best_loss = float('inf')
        loss_increase_counter = 0

        for iter in range(it):

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
            gaus_transform(g2copy, R_opt, Tscale * T2 + offset)
            rescale(g2copy, Sscale)
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

            cam_pos_right2 = myMiniCam3(450, 250, cam_R_right2, cam_T_right2, fovx, fovy, znear, zfar)

            cam_R_new3 = (RI1).T @ cam_R_right2 
            cam_T_new3 = cam_T_right2 - (TI1) @ (RI1).T @ cam_R_right2

            cam_pos_new2 = myMiniCam3(450, 250, cam_R_new3, cam_T_new3, fovx, fovy, znear, zfar)
            render2_new = render(cam_pos_new2, g2copy, pipe, background, scaling_modifer)
            img_new2 = render2_new["render"]
            mask2_new = render2_new["mask"]
            with torch.no_grad():
                render2_ref = render(cam_pos_right2, g1, pipe, background, scaling_modifer)
                img_ref2 = render2_ref["render"]
                mask2_ref = render2_ref["mask"]
            
            weight = 0.0
            Ll12 = masked_l1_loss(img_new2, img_ref2, mask2_ref, mask2_new)
            loss2 = (1.0 - weight) * Ll12 + weight * (1.0 - ssim(img_new2, img_ref2))
            cam_pos_right = myMiniCam3(450, 250, cam_R_right, cam_T_right, fovx, fovy, znear, zfar)

            cam_R_new = (RI1).T @ cam_R_right 
            cam_T_new = cam_T_right - (TI1) @ (RI1).T @ cam_R_right
            cam_pos_new = myMiniCam3(450, 250, cam_R_new, cam_T_new, fovx, fovy, znear, zfar)
            render_new = render(cam_pos_new, g2copy, pipe, background, scaling_modifer)

            img_new = render_new["render"]
            mask_new = render_new["mask"]
            with torch.no_grad():
                render_ref = render(cam_pos_right, g1, pipe, background, scaling_modifer)
                img_ref = render_ref["render"]
                mask_ref = render_ref["mask"]

            Ll1 = masked_l1_loss(img_new, img_ref, mask_ref, mask_new)
            loss1 = (1.0 - weight) * Ll1 + weight * (1.0 - ssim(img_new, img_ref))
            loss = loss1 + loss2
            
            loss.backward()
    
            optimizer.step()

            if iter % lr_update_iter == 0:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                print(f'\rIteration {iter}: Loss = {loss.item()}, Learning Rate = {current_lr}', end = '')
            
            loss_val = loss.item()
            print(f'\rIteration {iter}: Loss = {loss_val}', end='')

            if best_loss > loss_val:
                best_loss = loss_val
                best_param = (Tscale, rot, Sscale, offset)
                loss_increase_counter = 0
            else:
                loss_increase_counter += 1

            if loss_increase_counter >= early_stop_threshold:
                print("\nStopping training - loss has increased for 20 consecutive iterations.")
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
        
        # show trained images for each or every few epochs, controlled by flag
        if show_images and epoch % 5 == 0:
            img2 = img_new.permute(1, 2, 0).cpu().detach().numpy()
            image = cv2.cvtColor(img2.astype('float32'), cv2.COLOR_RGB2BGR)

            img3 = img_ref.permute(1, 2, 0).cpu().detach().numpy()
            image3 = cv2.cvtColor(img3.astype('float32'), cv2.COLOR_RGB2BGR)
            concatenated_image = cv2.hconcat([image, image3])
            cv2.imshow('Combined Image', concatenated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            img2 = img_new2.permute(1, 2, 0).cpu().detach().numpy()
            image = cv2.cvtColor(img2.astype('float32'), cv2.COLOR_RGB2BGR)

            img3 = img_ref2.permute(1, 2, 0).cpu().detach().numpy()
            image3 = cv2.cvtColor(img3.astype('float32'), cv2.COLOR_RGB2BGR)
            concatenated_image = cv2.hconcat([image, image3])
            cv2.imshow('Combined Image', concatenated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    Tscale, rot, Sscale, offset = best_param
    return best_loss, offset, rot, Sscale, Tscale