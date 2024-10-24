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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    def printinfo(self):
        print(self.image_width, self.image_height, self.FoVx, self.FoVy, self.znear, self.zfar,
              self.world_view_transform, self.full_proj_transform, self.camera_center, end = '\n')
    def copy(self):
    # Create a new instance with copies of the data
        new_camera = Camera(
        colmap_id=self.colmap_id,
        R=self.R,
        T=self.T,
        FoVx=self.FoVx,
        FoVy=self.FoVy,
        image=self.original_image.clone(),  # Assuming original_image is already a tensor on the right device
        gt_alpha_mask=None,  # Since we're applying it directly to the image, just pass None
        image_name=self.image_name,
        uid=self.uid,
        trans=np.copy(self.trans),
        scale=self.scale,
        data_device=self.data_device.type
        )
        # Assuming there is no need to recompute transformations as they should be identical
        return new_camera
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
    def printinfo(self):
        print(self.image_width, self.image_height, self.FoVx, self.FoVy, self.znear, self.zfar,
              self.world_view_transform, self.full_proj_transform, self.camera_center, end = '\n')

class myMiniCam2:
    def __init__(self, width, height, R, T, fovx, fovy, znear, zfar, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0,1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    def printinfo(self):
        print(self.image_width, self.image_height, self.FoVx, self.FoVy, self.znear, self.zfar,
              self.world_view_transform, self.full_proj_transform, self.camera_center, end = '\n')
        
class myMiniCam3:
    def __init__(self, width, height, R, T, fovx, fovy, znear, zfar, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = getWorld2View3(R, T, trans, scale).transpose(0,1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    def printinfo(self):
        print(self.image_width, self.image_height, self.FoVx, self.FoVy, self.znear, self.zfar,
              self.world_view_transform, self.full_proj_transform, self.camera_center, end = '\n')


import torch

def getWorld2View3(R, t, translate=None, scale=1.0):
    device = R.device  # Assume R is already on the desired device, typically passed by the caller

    # If translate is not provided or not a tensor, initialize it properly
    if translate is None:
        translate = torch.tensor([0.0, 0.0, 0.0], device=device)
    elif not isinstance(translate, torch.Tensor):
        translate = torch.tensor(translate, device=device)

    # Ensure scale is a tensor on the same device as R
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor([scale], device=device)

    # Initialize Rt matrix on the same device as R
    Rt = torch.zeros(4, 4, device=device)
    Rt[:3, :3] = R.t()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    # Compute the inverse to get the camera to world matrix
    C2W = torch.inverse(Rt)

    # Calculate the camera center and adjust it
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale

    # Update C2W with the new camera center
    C2W[:3, 3] = cam_center

    # Recompute the world to view matrix by inverting C2W
    Rt = torch.inverse(C2W)

    return Rt.type(torch.float32)


