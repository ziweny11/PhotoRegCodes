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
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
# from scene.shading import ShadingModel

from utils.general_utils import safe_state
import uuid
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

#left   I2 in g2

    # R = np.array([[ 0.0788, -0.0589, 0.9951],
    #             [ -0.1201,  0.9915, 0.0492],
    #             [ -0.9896, -0.1157,  0.0852]])
    # T = np.array([0.7, 0.1853, 5.5])

#right I1 in g1

    # R = np.array([[0.0079,  0.0115, -0.9999],
    #             [  0.0042,  0.9999,  0.0115],
    #             [ 1.0000, -0.0043,  0.0079]])
    # T = np.array([-3.1356,  0.4687,  3.7352])

def xyztransform(xyz, K, R, T):
    xyz_new = torch.mm(xyz, R.t()) + T 
    return torch.mm(xyz_new, K)
    # return torch.mm(xyz_new, K)


def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, testing_iterations, saving_iterations, 
             checkpoint_iterations, checkpoint: str, checkpoint2: str, debug_from, K, R, T, K2, R2, T2):
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.training_setup(opt)
    g2 = GaussianModel(dataset.sh_degree)  
    g2.training_setup(opt)

    first_iter = 0
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print("First model loaded from checkpoint.")

    if checkpoint2:
        (model_params, first_iter) = torch.load(checkpoint2)
        g2.restore(model_params, opt)
        print("Second model loaded from checkpoint2.")
    

    gnew = GaussianModel(dataset.sh_degree)
    gnew.training_setup(opt)


# [[ 0.0838, -0.1847,  0.9792,  0.0000],
#         [ 0.3079,  0.9394,  0.1508,  0.0000],
#         [-0.9477,  0.2888,  0.1356,  0.0000],
#         [ 3.7298, -1.0476,  4.0178,  1.0000]]
    RI1tmp = np.array([[0.0838, -0.1847,  0.9792],
                 [0.3079,  0.9394,  0.1508],
                 [-0.9477,  0.2888,  0.1356]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([3.7298, -1.0476,  4.0178]) 
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")

# [[ 0.5767, -0.0584, -0.8148, -0.0000],
#         [ 0.0948,  0.9955, -0.0043,  0.0000],
#         [ 0.8114, -0.0748,  0.5797,  0.0000],
#         [-3.1256,  0.1980,  3.9239,  1.0000]]
    RI2tmp = np.array([[ 0.5767, -0.0584, -0.8148],
                 [ 0.0948,  0.9955, -0.0043],
                 [ 0.8114, -0.0748,  0.5797]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([-3.1256,  0.1980,  3.9239])
    TI2 = torch.tensor(TI2tmp, dtype=torch.float32, device="cuda")

    # TI1 = -torch.mm(RI1.t(), TI1.view(3,1)).view(1,3)
    # TI2 = -torch.mm(RI2.t(), TI2.view(3,1)).view(1,3)

    xyzI = xyztransform(xyz, K, RI1.t(), TI1)
    xyz = xyztransform(xyzI, K, R, T)
    xyz2I = xyztransform(xyz2, K, RI2.t(), TI2)
    xyz2 = xyztransform(xyz2I, K2, R2, T2)


    # xyz = xyzI + T
    # xyz2 = xyz2I + T2


    xyz_new = torch.cat((xyz, xyz2), dim=0)
    print("this is xyz new", xyz_new)
    features_dc_new = torch.cat((features_dc, features_dc2), dim=0)
    features_rest_new = torch.cat((features_rest, features_rest2), dim=0)
    scaling_new = torch.cat((scaling, scaling2), dim=0)
    rotation_new = torch.cat((rotation, rotation2), dim=0)
    opacity_new = torch.cat((opacity, opacity2), dim=0)


    gnew._xyz = xyz_new
    gnew._features_dc = features_dc_new
    gnew._features_rest = features_rest_new
    gnew._scaling = scaling_new
    gnew._rotation = rotation_new
    gnew._opacity = opacity_new

    # tl = [type(item) for item in g2.capture()]
    # print (tl, "\n")
    # # print(gaussians.capture())

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    first_iter += 1
    # for iteration in range(first_iter, opt.iterations + 1): 
    while True:       
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gnew, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
            except Exception as e:
                network_gui.conn = None
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

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
    # R = torch.tensor([[ 0.8395,  0.0043, -0.5434],
    #     [ 0.1061,  0.9794,  0.1717],
    #     [ 3.5031e-03,  6.0576e-04,  9.9999e-01]], dtype=torch.float32, device = 'cuda') 
    # T = torch.tensor([ 0.0307, -0.0062,  0.0398], dtype=torch.float32, device = 'cuda')
    Rxx = [[-0.6000, -0.0272, -0.7995],
        [0.2348, -0.9614, -0.1435],
        [-0.7648, -0.2738,  0.5833]]
    RR = np.linalg.inv(np.array(Rxx))
    #[ 4.8086, -1.1736, 0.0880]
    R0 = [[1,0,0],[0,1,0],[0,0,1]]
    T0 = [ 0, 0, 0]


    Kxx = [[512.,   0., 256.],
         [  0., 512., 136.],
         [  0.,   0.,   1.]]
    K = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    K2 = K
    Ryy = [[ 0.8934, -0.4305, -0.1281],
        [-0.1724, -0.0652, -0.9829],
        [ 0.4148,  0.9002, -0.1325]]
    RRR = np.linalg.inv(np.array(Ryy))
    relR = np.dot(RR, Ryy)

# 0.6087, -0.1995, -0.7679,  0.0719],
#          [ 0.1973,  0.9755, -0.0971,  0.0103],
#          [ 0.7685, -0.0924,  0.6332,  0.0526],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]])

# #[[ 0.7123,  0.1032,  0.6942, -0.0966],
#          [-0.0718,  0.9947, -0.0742,  0.0132],
#          [-0.6982,  0.0031,  0.7159,  0.1016],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]])


#truck exam:
# [[ 0.5493,  0.2762, -0.7887,  0.1645],
#          [-0.0753,  0.9563,  0.2825, -0.0455],
#          [ 0.8322, -0.0957,  0.5461,  0.1176],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]
    R22 = np.array([[ 0.5493,  0.2762, -0.7887],
                    [-0.0753,  0.9563,  0.2825],
                    [ 0.8322, -0.0957,  0.5461]])
    T22 = np.array([0.1645, -0.0455, 0.1176])
    # rot = np.array([[1,0,0], [0,1,0], [-1,0,-1]])


    R2 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R = torch.tensor(R22, dtype=torch.float32, device = 'cuda') 
    T = torch.tensor(T22, dtype=torch.float32, device = 'cuda')
    
    T = T * 21
    #[ -9.2656, -0.0252, -1.9029]
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, K, R, T, K2, R2, T2)

    print("\nTraining complete.")

# [ 0.8628, -0.0304, -0.5046,  0.0687],
#          [ 0.0807,  0.9937,  0.0781,  0.0032],
#          [ 0.4990, -0.1081,  0.8598,  0.0597],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]



# tensor([[[-0.6000, -0.0272, -0.7995,  0.3344],
#          [ 0.2348, -0.9614, -0.1435, -1.0492],
#          [-0.7648, -0.2738,  0.5833,  0.2428],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]],

#         [[ 0.8934, -0.4305, -0.1281, -2.2315],
#          [-0.1724, -0.0652, -0.9829, -1.5556],
#          [ 0.4148,  0.9002, -0.1325, -1.8182],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]]

# [[ 0.2031,  0.7122, -0.6720,  4.8086],
#          [-0.0555, -0.6768, -0.7341, -1.1736],
#          [-0.9776,  0.1864, -0.0979,  0.0880],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]],

#         [[ 0.4838,  0.8574,  0.1755, -9.2656],
#          [ 0.6848, -0.2460, -0.6860, -0.0252],
#          [-0.5450,  0.4521, -0.7061, -1.9029],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]



#  [9.9934e-01, -3.6212e-02, -3.4789e-03,  0.0000e+00],
#          [ 3.6209e-02,  9.9934e-01, -7.3222e-04,  0.0000e+00],
#          [ 3.5031e-03,  6.0576e-04,  9.9999e-01,  0.0000e+00],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#         [[ 8.3785e-01,  1.1006e-02, -5.4579e-01,  2.9450e-02],
#          [ 9.8952e-02,  9.8017e-01,  1.7167e-01, -6.9754e-03],
#          [ 5.3686e-01, -1.9784e-01,  8.2015e-01,  3.8085e-02],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]],




#     set all pairwise pose, num = 0
# R and T: tensor([[ 1.0000e+00,  1.5860e-07, -1.8665e-07],
#         [-7.9991e-08,  1.0000e+00, -8.5180e-08],
#         [ 7.6919e-08,  9.1715e-08,  1.0000e+00]], device='cuda:0') tensor([ 5.2154e-08,  8.8476e-09, -1.4901e-08], device='cuda:0')
# set all pairwise pose, num = 1
# R and T: tensor([[ 0.8248,  0.0543, -0.5628],
#         [ 0.0673,  0.9789,  0.1931],
#         [ 0.5614, -0.1972,  0.8037]], device='cuda:0') tensor([ 0.0572, -0.0145,  0.0892], device='cuda:0')
# init all image poses, num =  0
# R and T: tensor([[ 0.8395,  0.0043, -0.5434],
#         [ 0.1061,  0.9794,  0.1717],
#         [ 0.5330, -0.2018,  0.8217]], device='cuda:0') tensor([ 0.0307, -0.0062,  0.0398], device='cuda:0')
# init all image poses, num =  1
# R and T: tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]], device='cuda:0') tensor([0., 0., 0.], device='cuda:0')
#  init loss = 0.0037714005447924137
# Global alignement - optimizing for:
# ['pw_poses', 'im_depthmaps', 'im_poses', 'im_focals', 'im_conf.0', 'im_conf.1']
# 100% 300/300 [00:10<00:00, 28.20it/s, lr=0.01 loss=0.00153705]
# found 248 matches
