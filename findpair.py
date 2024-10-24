#function takes in two GS and camera infos, output best image pair for dust3r alignment 

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
from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary
from dinov2_utils import classify_image
import numpy as np
import torch.nn.functional as F
import os
from scene.cameras import myMiniCam3
from myUtils import gaus_copy
import torch
from gaussian_renderer import render
from scene import Scene, GaussianModel


#pick suitable campos and images for dust3r input by dinov2 similarity
def campick(g1, g2, extrinsics1, extrinsics2, dataset, pipe):
    
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


    d2list1 = []
    for key in list(extrinsics1.keys()):
        img = extrinsics1[key]
        qvec, tvec = img.qvec, img.tvec
        rotmat = qvec2rotmat(qvec)
        rotmat = torch.tensor(rotmat, dtype=torch.float32, device="cuda")
        tvec = torch.tensor(tvec, dtype=torch.float32, device="cuda")
        rotmat = rotmat.T
        cam_pos = myMiniCam3(512, 336, rotmat, tvec, fovx, fovy, znear, zfar)
        img1 = render(cam_pos, g1, pipe, background, scaling_modifer)["render"]
        dv1 = classify_image(img1)
        d2list1.append((dv1, rotmat, tvec))

    d2list2 = []
    for key in list(extrinsics2.keys()):
        img = extrinsics2[key]
        qvec, tvec = img.qvec, img.tvec
        rotmat = qvec2rotmat(qvec)
        rotmat = torch.tensor(rotmat, dtype=torch.float32, device="cuda")
        tvec = torch.tensor(tvec, dtype=torch.float32, device="cuda")
        rotmat = rotmat.T
        cam_pos = myMiniCam3(512, 336, rotmat, tvec, fovx, fovy, znear, zfar)
        img2 = render(cam_pos, g2, pipe, background, scaling_modifer)["render"]
        dv2 = classify_image(img2)
        d2list2.append((dv2, rotmat, tvec))

    max_sim = 0
    res = ()

    for v1, r1, t1 in d2list1:
        for v2, r2, t2 in d2list2:
            # v1, r1, t1 = d2list1[13]
            # v2, r2, t2 = d2list2[22]
            cos_sim = F.cosine_similarity(v1, v2, dim = 0)
            if cos_sim > max_sim and cos_sim < 0.8:
                max_sim = cos_sim
                res = (r1, t1, r2, t2)
    r1, t1, r2, t2 = res
    cam_pos1 = myMiniCam3(512, 336, r1, t1, fovx, fovy, znear, zfar)
    img1 = render(cam_pos1, g1, pipe, background, scaling_modifer)["render"]
    cam_pos2 = myMiniCam3(512, 336, r2, t2, fovx, fovy, znear, zfar)
    img2 = render(cam_pos2, g2, pipe, background, scaling_modifer)["render"]
    print (res)
    print(max_sim)
    return img1, img2, r1, t1, r2, t2
    
def tensor2np(img):
    img = img.permute(1,2,0).cpu().detach().numpy()
    image = cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2BGR)
    return image

def findpair(dataset, opt, pipe, checkpoint: str, checkpoint2: str, campose1, campose2):

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
    print(g1._xyz.shape, g2._xyz.shape)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    g1copy = GaussianModel(dataset.sh_degree)
    g1copy.training_setup(opt)    
    g2copy = GaussianModel(dataset.sh_degree)
    g2copy.training_setup(opt)
    gaus_copy(g1, g1copy)
    gaus_copy(g2, g2copy)

    #set up extrinsincs cam dicts:
    path1 = campose1
    cameras_extrinsic_file1 = os.path.join(path1, "sparse/0", "images.bin")
    cam_extrinsics1 = read_extrinsics_binary(cameras_extrinsic_file1)
    path2 = campose2
    cameras_extrinsic_file2 = os.path.join(path2, "sparse/0", "images.bin")
    cam_extrinsics2 = read_extrinsics_binary(cameras_extrinsic_file2)
    
    img1, img2, r1, t1, r2, t2 = campick(g1copy, g2copy, cam_extrinsics1, cam_extrinsics2, dataset, pipe)
    
    npimg1 = tensor2np(img1)
    npimg2 = tensor2np(img2)

    cv2.imwrite('img1.png', npimg1)
    cv2.imwrite('img2.png', npimg2)

    cv2.imshow('Image', npimg1)
    cv2.waitKey(0)
    cv2.imshow('Image', npimg2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return r1, t1, r2, t2

