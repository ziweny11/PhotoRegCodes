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
import open3d as o3d
from utils.sh_utils import SH2RGB
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

#Utils:
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity

def gau2colorpc(g):
    shs = g.get_features[:,0,:]
    color = SH2RGB(shs)
    color = color.detach().cpu().numpy()
    pc1 = g._xyz
    pc1 = pc1.detach().cpu().numpy()
    return color, pc1

def map_to_shade(colors, base_color, blend_factor = 0.2):
    return colors * (1 - blend_factor) + base_color * blend_factor


def visualize_pcd_with_cam_params(pcd, cam_params):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(cam_params)
    vis.run()
    vis.destroy_window()


def manual_view_adjustment(point_cloud):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    
    # User manually adjusts the camera
    print("Adjust the camera as needed and close the window when done.")
    
    # Main loop
    while True:
        vis.poll_events()
        vis.update_renderer()
        if not vis.poll_events():
            break
    
    # Get camera parameters after adjustment
    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    vis.destroy_window()
    print (camera_params.extrinsic)
    return camera_params

def capture_screenshots_with_params(point_clouds, camera_params, file_names):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Set camera parameters
    for pc, filename in zip(point_clouds, file_names):
        vis.add_geometry(pc)
        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(filename)
        camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        print(camera_params.extrinsic)
        vis.clear_geometries()  # Clear for the next point cloud
    
    vis.destroy_window()



def visualize_pcd_with_extrinsics(point_cloud, extrinsic_matrix):
    extrinsic_matrix = np.array(extrinsic_matrix)

    # Initialize visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add point cloud to visualizer
    vis.add_geometry(point_cloud)

    # Retrieve and set camera parameters
    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()

    # Apply the extrinsic matrix
    camera_params.extrinsic = extrinsic_matrix
    view_control.convert_from_pinhole_camera_parameters(camera_params)

    # IMPORTANT: Poll events and update the renderer after setting the camera
    vis.poll_events()
    vis.update_renderer()

    # Run the visualizer
    print("Visualization initialized. Adjust the view if necessary and close the window when finished.")
    vis.run()
    vis.destroy_window() 



#RI1, TI1, RI2, TI2, R1, T1, R2, T2
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
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([1.9350299835205078], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    gaus_transform(g2copy, RI1, - RI1 @ TI1)
    color1, points1 = gau2colorpc(g1copy)
    color2, points2 = gau2colorpc(g2copy)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    red_color = np.tile([1, 0, 0], (len(points1), 1))
    blue_color = np.tile([0, 0, 1], (len(points2), 1))

    # pcd1.colors = o3d.utility.Vector3dVector(red_color)
    # pcd2.colors = o3d.utility.Vector3dVector(blue_color)
    pcd1.colors = o3d.utility.Vector3dVector(map_to_shade(color1, red_color))
    pcd2.colors = o3d.utility.Vector3dVector(map_to_shade(color2, blue_color))

    voxel_size = 0.1  # Adjust based on your specific needs

    # Apply voxel downsampling
    down_pcd1 = pcd1.voxel_down_sample(voxel_size)
    down_pcd2 = pcd2.voxel_down_sample(voxel_size)
    
    pcd_merge = down_pcd1 + down_pcd2


    o3d.io.write_point_cloud("multi_21_viz.ply", pcd_merge)
    
    o3d.visualization.draw_geometries([pcd_merge])





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

    RI1tmp = np.array([[ 0.1742,  0.7024, -0.6901],
        [-0.7600,  0.5415,  0.3594],
        [ 0.6261,  0.4619,  0.6282]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([ 6.8444,  1.8075, -1.9562])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")

    RI2tmp =  np.array([[ 0.6059, -0.6049,  0.5167],
        [ 0.6092,  0.7705,  0.1877],
        [-0.5117,  0.2011,  0.8353]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([ 1.7184,  0.3150, -1.2049])
    TI2 = torch.tensor(TI2tmp, dtype=torch.float32, device="cuda")
    # TI1 = -torch.mm(RI1.t(), TI1.view(3,1)).view(1,3)
    # TI2 = -torch.mm(RI2.t(), TI2.view(3,1)).view(1,3)


# [[[ 0.7641, -0.1013,  0.6371,  0.0218],
#          [ 0.0290,  0.9920,  0.1228,  0.0109],
#          [-0.6444, -0.0754,  0.7610,  c],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]


    R0 = [[1,0,0],[0,1,0],[0,0,1]]
    T0 = [ 0, 0, 0]

    R1tmp = np.array(R0)
    T1tmp = np.array(T0)

    R2tmp = np.array( [[0.8278205394744873, -0.4885037839412689, 0.2758210301399231], [0.5057887434959412, 0.8626025319099426, 0.009724809788167477], [-0.2426745444536209, 0.13145679235458374, 0.9611598253250122]])
    
    T2tmp = np.array([-0.1260, -0.0339,-0.0138])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([-0.02491196058690548, -0.02893947809934616, -0.03709326311945915]), dtype=torch.float32, device = 'cuda')


    T2 = T2 * 10 + offset
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")
