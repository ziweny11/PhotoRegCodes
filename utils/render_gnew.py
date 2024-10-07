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

import numpy as np
from scene.cameras import Camera
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel



#utils, merge into camera class later
def trans_cam(cam, R, T):
    R_old = cam.R
    T_old = cam.T
    R_new = R.T @ R_old
    T_new = T_old - T @ R.T @ R_old
    rescam = Camera(colmap_id=cam.colmap_id,
        R=R_new,
        T=T_new,
        FoVx=cam.FoVx,
        FoVy=cam.FoVy,
        image=cam.original_image.clone(),  # Assuming original_image is already a tensor on the right device
        gt_alpha_mask=None,  # Since we're applying it directly to the image, just pass None
        image_name=cam.image_name,
        uid=cam.uid,
        trans=np.copy(cam.trans),
        scale=cam.scale,
        data_device=cam.data_device.type)
    return rescam 

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    R = np.array([[ 9.6280e-01,  5.2044e-02, -2.6517e-01],
        [-5.0157e-02,  9.9864e-01,  1.3887e-02],
        [ 2.6553e-01, -7.0159e-05,  9.6410e-01]])
    T = np.array([ 3.0873,  0.3105, -2.3074])

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        trans_view = trans_cam(view, R, T)
        rendering = render(trans_view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        (model_params, first_iter) = torch.load('cs_ICP_eval/gnew.pth')
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.restore2(model_params)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        print(dataset.model_path)
        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)