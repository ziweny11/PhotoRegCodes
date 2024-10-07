from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale

import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from utils.general_utils import build_rotation
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.cameras import MiniCam, Camera, myMiniCam2, myMiniCam3
from utils.general_utils import safe_state
import uuid
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import pyvista as pv
from utils.sh_utils import SH2RGB

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# Utils:


def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity

def gau2colorpc(g):
    shs = g.get_features[:, 0, :]
    color = SH2RGB(shs)
    color = color.detach().cpu().numpy()
    pc1 = g._xyz
    pc1 = pc1.detach().cpu().numpy()
    return color, pc1

def map_to_shade(colors, base_color, blend_factor=0.4):
    return (colors * (1 - blend_factor) + base_color * blend_factor)


# RI1, TI1, RI2, TI2, R1, T1, R2, T2
def merge(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, testing_iterations, saving_iterations, 
             checkpoint_iterations, checkpoint: str, checkpoint2: str, debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2):
    g1 = GaussianModel(dataset.sh_degree)
    g1.training_setup(opt)
    g2 = GaussianModel(dataset.sh_degree)  
    g2.training_setup(opt)
    first_iter = 0
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        g1.restore(model_params, opt)
        print("First model loaded from checkpoint.")

    if checkpoint2:
        (model_params, first_iter) = torch.load(checkpoint2)
        g2.restore(model_params, opt)
        print("Second model loaded from checkpoint2.")
    
    g1copy = GaussianModel(dataset.sh_degree)
    g1copy.training_setup(opt)    
    g2copy = GaussianModel(dataset.sh_degree)
    g2copy.training_setup(opt)
    gnew = GaussianModel(dataset.sh_degree)
    gnew.training_setup(opt)

    gaus_copy(g1, g1copy)
    gaus_copy(g2, g2copy)
    print("here", g1copy._xyz.shape, g2copy._xyz.shape)
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([0.6], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    color1, points1 = gau2colorpc(g1copy)
    color2, points2 = gau2colorpc(g2copy)
    pcd1 = pv.PolyData(points1)
    pcd2 = pv.PolyData(points2)
    red_color = np.tile([1, 0, 0], (len(points1), 1))
    blue_color = np.tile([0, 0, 1], (len(points2), 1))

    # Map to shade and ensure the color is in the correct range [0, 1]
    pcd1['colors'] = map_to_shade(color1, red_color)
    pcd2['colors'] = map_to_shade(color2, blue_color)
    
    def take_screenshot0():
        screenshot_path = "point_cloud_screenshot.png"
        plotter.screenshot(screenshot_path)
        print(f"Screenshot saved as '{screenshot_path}'")
        
    def take_screenshot1():
        screenshot_path = "point_cloud_screenshot1.png"
        plotter1.screenshot(screenshot_path)
        print(f"Screenshot saved as '{screenshot_path}'")
        
    def take_screenshot2():
        screenshot_path = "point_cloud_screenshot2.png"
        plotter2.screenshot(screenshot_path)
        print(f"Screenshot saved as '{screenshot_path}'")
        
    plotter = pv.Plotter()
    plotter.background_color = 'white'
    plotter.add_points(pcd1, scalars='colors', rgb=True, render_points_as_spheres=True, point_size=1)
    plotter.add_points(pcd2, scalars='colors', rgb=True, render_points_as_spheres=True, point_size=1)
    plotter.add_key_event("p", take_screenshot0)
    plotter.show(auto_close=False)
    saved_camera_position = plotter.camera_position
    print(saved_camera_position)
    plotter1 = pv.Plotter()
    plotter1.background_color = 'white'
    plotter1.add_points(pcd1, scalars='colors', rgb=True, render_points_as_spheres=True, point_size=1)
    plotter1.camera_position = saved_camera_position
    plotter1.add_key_event("p", take_screenshot1)
    plotter1.show()
    plotter2 = pv.Plotter()
    plotter2.background_color = 'white'
    plotter2.add_points(pcd2, scalars='colors', rgb=True, render_points_as_spheres=True, point_size=1)
    plotter2.camera_position = saved_camera_position
    plotter2.add_key_event("p", take_screenshot2)
    plotter2.show()
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
    # safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    RI1tmp = np.array([[ 9.6280e-01,  5.2044e-02, -2.6517e-01],
        [-5.0157e-02,  9.9864e-01,  1.3887e-02],
        [ 2.6553e-01, -7.0159e-05,  9.6410e-01]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([ 3.0873,  0.3105, -2.3074])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")

    RI2tmp =  np.array([[ 0.9888,  0.1016,  0.1093],
        [-0.1280,  0.9540,  0.2710],
        [-0.0767, -0.2819,  0.9564]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([ 3.6159,  0.2146, -2.3983])
    TI2 = torch.tensor(TI2tmp, dtype=torch.float32, device="cuda")
    # TI1 = -torch.mm(RI1.t(), TI1.view(3,1)).view(1,3)
    # TI2 = -torch.mm(RI2.t(), TI2.view(3,1)).view(1,3)


    R0 = [[1,0,0],[0,1,0],[0,0,1]]
    T0 = [ 0, 0, 0]

    R1tmp = np.array(R0)
    T1tmp = np.array(T0)

    R2tmp = np.array([[0.9822, -0.0210,  0.1869],
        [0.0264,  0.9993, -0.0268],
        [-0.1862,  0.0313,  0.9820]])


    T2tmp = np.array([0.0027, -0.0015, -0.0182])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([0.0, 0.0, 0.0]), dtype=torch.float32, device = 'cuda')


    T2 = T2 * 14 + offset
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")
