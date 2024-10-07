#d1
#train right and left 2, not good:(g2merge)

#contains functions and main for merging two gaussian models.

from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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



#utils:

# def qua_rotate(Q, R):
#     QM = quaternion_to_matrix(Q)
#     return torch.matmul(QM, R)
# def gaus_rotate(G, R):
#     G._xyz = torch.mm(G._xyz, R.t())
#     new_rotation = build_rotation(G._rotation)
#     new_rotation = R @ new_rotation
#     # G._rotation = extract_rotation_scipy(new_rotation)
#     G._rotation = matrix_to_quaternion(qua_rotate(G._rotation, R))
# def transform_shs(shs_feat, rotation_matrix):

#     ## rotate shs
#     P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) # switch axes: yzx -> xyz
#     rotation_matrix = rotation_matrix.cpu().numpy()
#     permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix @ P
#     rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotation_matrix))
    
#     # Construction coefficient
#     D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device='cuda', dtype=torch.float32)
#     D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device='cuda', dtype=torch.float32)
#     D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device='cuda', dtype=torch.float32)
#     #rotation of the shs features
#     one_degree_shs = shs_feat[:, 0:3]
#     one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     one_degree_shs = einsum(
#             D_1,
#             one_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat = torch.cat((one_degree_shs, shs_feat[:, 3:]), dim=1)

#     two_degree_shs = shs_feat[:, 3:8]
#     two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     two_degree_shs = einsum(
#             D_2,
#             two_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat[:, 3:8] = two_degree_shs

#     three_degree_shs = shs_feat[:, 8:15]
#     three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     three_degree_shs = einsum(
#             D_3,
#             three_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat[:, 8:15] = three_degree_shs

#     return shs_feat

# def gaus_translate(G, t):
#     G._xyz = G._xyz + t
# def gaus_append(g1, g2, gnew):
#     xyz_new = torch.cat((g1._xyz, g2._xyz), dim=0)
#     # print("this is xyz new", xyz_new)
#     features_dc_new = torch.cat((g1._features_dc, g2._features_dc), dim=0)
#     features_rest_new = torch.cat((g1._features_rest, g2._features_rest), dim=0)
#     scaling_new = torch.cat((g1._scaling, g2._scaling), dim=0)
#     rotation_new = torch.cat((g1._rotation, g2._rotation), dim=0)
#     opacity_new = torch.cat((g1._opacity, g2._opacity), dim=0)
#     gnew._xyz = xyz_new
#     gnew._features_dc = features_dc_new
#     gnew._features_rest = features_rest_new
#     gnew._scaling = scaling_new
#     gnew._rotation = rotation_new
#     gnew._opacity = opacity_new

# def gaus_transform(G, R, t):
#     rotate_by_matrix(G, R)
#     gaus_translate(G, t)


# def rotate_by_matrix(G, rotation_matrix, keep_sh_degree: bool = True):
#     # rotate xyz
#     G._xyz = torch.mm(G._xyz, rotation_matrix.t())
#     gaussian_rotation = build_rotation(G._rotation)
#     gaussian_rotation = rotation_matrix @ gaussian_rotation
#     # gaussian_rotation = torch.matmul(gaussian_rotation, rotation_matrix.t())
#     xyzw_quaternions = R.from_matrix(gaussian_rotation.cpu().detach().numpy()).as_quat()
#     wxyz_quaternions = xyzw_quaternions
#     wxyz_quaternions[:, [0, 1, 2, 3]] = wxyz_quaternions[:, [3, 0, 1, 2]]
#     rotations_from_matrix = torch.tensor(wxyz_quaternions, dtype=torch.float32, device="cuda")
#     G._rotation = rotations_from_matrix
#     if keep_sh_degree is False:
#         print("set sh_degree=0 when rotation transform enabled")
#         G.sh_degrees = 0
#     G._features_rest = transform_shs(G._features_rest, rotation_matrix)

# #write into class later
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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

    #trucktwin_right:
    # [[ 0.1714, -0.0919,  0.9809,  0.0000],
    #     [ 0.3924,  0.9196,  0.0176,  0.0000],
    #     [-0.9037,  0.3819,  0.1936,  0.0000],
    #     [ 0.2364, -0.1797,  4.4662,  1.0000]]

    #truckscale right:
    #    [[ 0.2751, -0.3128,  0.9091,  0.0000],
    #     [ 0.3723,  0.9065,  0.1993,  0.0000],
    #     [-0.8864,  0.2836,  0.3658,  0.0000],
    #     [ 1.1816, -0.8960,  3.5893,  1.0000]]

        # [[ 0.0202, -0.3514,  0.9360,  0.0000],
    #     [ 0.4272,  0.8495,  0.3097,  0.0000],
    #     [-0.9039,  0.3936,  0.1672,  0.0000],
    #     [ 2.0416, -1.3046,  2.8997,  1.0000]]
    RI1tmp = np.array([[0.7709,  0.0198,  0.6367],
                 [0.0819,  0.9881, -0.1298],
                 [-0.6317,  0.1522,  0.7601]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([3.4646,  0.3195,  4.7689])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")



    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[0.1042,  0.1007, -0.9894],
                 [ 0.1596,  0.9803,  0.1166],
                 [0.9817, -0.1701,  0.0861]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([-1.4229, -0.0060,  5.9085])
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

    #truck_scale case
    # [ 0.3595, -0.3023,  0.8828, -0.1953],
    #      [ 0.4284,  0.8939,  0.1316, -0.0555],
    #      [-0.8290,  0.3309,  0.4509,  0.1100],
    #      [ 0.0000,  0.0000,  0.0000,  1.0000]]
    #truckLR:
    # [[ 0.5586, -0.3078,  0.7702, -0.1802],
    #      [ 0.2557,  0.9473,  0.1931, -0.0490],
    #      [-0.7891,  0.0891,  0.6078,  0.1383],
    #      [ 0.0000,  0.0000,  0.0000,  1.0000]]


# [[[ 0.7481, -0.2017,  0.6322, -0.2223],
#          [ 0.2931,  0.9551, -0.0422,  0.0202],
#          [-0.5953,  0.2169,  0.7737,  0.1172],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]



# [[ 0.8982, -0.0013,  0.4396, -0.0796],
#          [-0.0379,  0.9960,  0.0804,  0.0041],
#          [-0.4380, -0.0889,  0.8946,  0.0401],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]


# [[ 6.6254e-01, -1.5983e-02,  7.4886e-01, -1.0150e-01],
#          [-7.9955e-02,  9.9255e-01,  9.1923e-02,  8.0308e-04],
#          [-7.4475e-01, -1.2078e-01,  6.5632e-01,  7.4551e-02],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]
    R2tmp = np.array( [[6.6254e-01, -1.5983e-02,  7.4886e-01],
                        [-7.9955e-02,  9.9255e-01,  9.1923e-02],
                          [-7.4475e-01, -1.2078e-01,  6.5632e-01]]

)
    T2tmp = np.array([-1.0150e-01,8.0308e-04,7.4551e-02])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([0,0,0]), dtype=torch.float32, device = 'cuda')


#     R2delta = torch.tensor(np.array( [[ 9.99999998e-01, -5.50246651e-05,  3.05926189e-05],
#  [ 5.50255531e-05,  9.99999998e-01, -2.90237284e-05],
#  [-3.05910218e-05 , 2.90254117e-05  ,9.99999999e-01]]), dtype=torch.float32, device = 'cuda')
#     T2delta = torch.tensor(np.array([-0.00108903,  0.00056625 ,-0.00060884]), dtype=torch.float32, device = 'cuda')
#     #scale
    T2 = T2 * 21 + offset
    # T2 = torch.mm(T2.reshape(1,3), R2delta.T) + T2delta
    # R2 = torch.mm(R2, R2delta.T)
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")

# ([[[ 1.0000,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  1.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  1.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]],

#         [[ 0.2498, -0.1963,  0.9482, -0.2265],
#          [ 0.2211,  0.9649,  0.1415, -0.0051],
#          [-0.9427,  0.1743,  0.2844,  0.1766],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]])



#d2:
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



#contains functions and main for merging two gaussian models.

from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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



#utils:

# def qua_rotate(Q, R):
#     QM = quaternion_to_matrix(Q)
#     return torch.matmul(QM, R)
# def gaus_rotate(G, R):
#     G._xyz = torch.mm(G._xyz, R.t())
#     new_rotation = build_rotation(G._rotation)
#     new_rotation = R @ new_rotation
#     # G._rotation = extract_rotation_scipy(new_rotation)
#     G._rotation = matrix_to_quaternion(qua_rotate(G._rotation, R))
# def transform_shs(shs_feat, rotation_matrix):

#     ## rotate shs
#     P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) # switch axes: yzx -> xyz
#     rotation_matrix = rotation_matrix.cpu().numpy()
#     permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix @ P
#     rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotation_matrix))
    
#     # Construction coefficient
#     D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device='cuda', dtype=torch.float32)
#     D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device='cuda', dtype=torch.float32)
#     D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device='cuda', dtype=torch.float32)
#     #rotation of the shs features
#     one_degree_shs = shs_feat[:, 0:3]
#     one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     one_degree_shs = einsum(
#             D_1,
#             one_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat = torch.cat((one_degree_shs, shs_feat[:, 3:]), dim=1)

#     two_degree_shs = shs_feat[:, 3:8]
#     two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     two_degree_shs = einsum(
#             D_2,
#             two_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat[:, 3:8] = two_degree_shs

#     three_degree_shs = shs_feat[:, 8:15]
#     three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     three_degree_shs = einsum(
#             D_3,
#             three_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat[:, 8:15] = three_degree_shs

#     return shs_feat

# def gaus_translate(G, t):
#     G._xyz = G._xyz + t
# def gaus_append(g1, g2, gnew):
#     xyz_new = torch.cat((g1._xyz, g2._xyz), dim=0)
#     # print("this is xyz new", xyz_new)
#     features_dc_new = torch.cat((g1._features_dc, g2._features_dc), dim=0)
#     features_rest_new = torch.cat((g1._features_rest, g2._features_rest), dim=0)
#     scaling_new = torch.cat((g1._scaling, g2._scaling), dim=0)
#     rotation_new = torch.cat((g1._rotation, g2._rotation), dim=0)
#     opacity_new = torch.cat((g1._opacity, g2._opacity), dim=0)
#     gnew._xyz = xyz_new
#     gnew._features_dc = features_dc_new
#     gnew._features_rest = features_rest_new
#     gnew._scaling = scaling_new
#     gnew._rotation = rotation_new
#     gnew._opacity = opacity_new

# def gaus_transform(G, R, t):
#     rotate_by_matrix(G, R)
#     gaus_translate(G, t)


# def rotate_by_matrix(G, rotation_matrix, keep_sh_degree: bool = True):
#     # rotate xyz
#     G._xyz = torch.mm(G._xyz, rotation_matrix.t())
#     gaussian_rotation = build_rotation(G._rotation)
#     gaussian_rotation = rotation_matrix @ gaussian_rotation
#     # gaussian_rotation = torch.matmul(gaussian_rotation, rotation_matrix.t())
#     xyzw_quaternions = R.from_matrix(gaussian_rotation.cpu().detach().numpy()).as_quat()
#     wxyz_quaternions = xyzw_quaternions
#     wxyz_quaternions[:, [0, 1, 2, 3]] = wxyz_quaternions[:, [3, 0, 1, 2]]
#     rotations_from_matrix = torch.tensor(wxyz_quaternions, dtype=torch.float32, device="cuda")
#     G._rotation = rotations_from_matrix
#     if keep_sh_degree is False:
#         print("set sh_degree=0 when rotation transform enabled")
#         G.sh_degrees = 0
#     G._features_rest = transform_shs(G._features_rest, rotation_matrix)

# #write into class later
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([0.846], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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

    #trucktwin_right:
    # [[ 0.1714, -0.0919,  0.9809,  0.0000],
    #     [ 0.3924,  0.9196,  0.0176,  0.0000],
    #     [-0.9037,  0.3819,  0.1936,  0.0000],
    #     [ 0.2364, -0.1797,  4.4662,  1.0000]]

    #truckscale right:
    #    [[ 0.2751, -0.3128,  0.9091,  0.0000],
    #     [ 0.3723,  0.9065,  0.1993,  0.0000],
    #     [-0.8864,  0.2836,  0.3658,  0.0000],
    #     [ 1.1816, -0.8960,  3.5893,  1.0000]]

        # [[ 0.0202, -0.3514,  0.9360,  0.0000],
    #     [ 0.4272,  0.8495,  0.3097,  0.0000],
    #     [-0.9039,  0.3936,  0.1672,  0.0000],
    #     [ 2.0416, -1.3046,  2.8997,  1.0000]]
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

    #truck_scale case
    # [ 0.3595, -0.3023,  0.8828, -0.1953],
    #      [ 0.4284,  0.8939,  0.1316, -0.0555],
    #      [-0.8290,  0.3309,  0.4509,  0.1100],
    #      [ 0.0000,  0.0000,  0.0000,  1.0000]]
    #truckLR:
    # [[ 0.5586, -0.3078,  0.7702, -0.1802],
    #      [ 0.2557,  0.9473,  0.1931, -0.0490],
    #      [-0.7891,  0.0891,  0.6078,  0.1383],
    #      [ 0.0000,  0.0000,  0.0000,  1.0000]]


# [[[ 0.7481, -0.2017,  0.6322, -0.2223],
#          [ 0.2931,  0.9551, -0.0422,  0.0202],
#          [-0.5953,  0.2169,  0.7737,  0.1172],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]



# [[ 0.8982, -0.0013,  0.4396, -0.0796],
#          [-0.0379,  0.9960,  0.0804,  0.0041],
#          [-0.4380, -0.0889,  0.8946,  0.0401],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]


# [[ 6.6254e-01, -1.5983e-02,  7.4886e-01, -1.0150e-01],
#          [-7.9955e-02,  9.9255e-01,  9.1923e-02,  8.0308e-04],
#          [-7.4475e-01, -1.2078e-01,  6.5632e-01,  7.4551e-02],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]

#pr
# [[ 0.9646,  0.0270,  0.2624, -0.0495],
#          [-0.0223,  0.9995, -0.0210, -0.0072],
#          [-0.2629,  0.0144,  0.9647,  0.1238],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]
    R2tmp = np.array( [[0.9666980504989624, 0.030665989965200424, 0.2540757358074188], [-0.027674710378050804, 0.9994992613792419, -0.015340091660618782], [-0.2544189393520355, 0.007797763217240572, 0.9670626521110535]]

)
    T2tmp = np.array([-0.0495,-0.0072,0.1238])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([-0.20643596351146698, -0.030631836503744125, 0.09977202862501144]), dtype=torch.float32, device = 'cuda')


#     R2delta = torch.tensor(np.array( [[ 9.99999998e-01, -5.50246651e-05,  3.05926189e-05],
#  [ 5.50255531e-05,  9.99999998e-01, -2.90237284e-05],
#  [-3.05910218e-05 , 2.90254117e-05  ,9.99999999e-01]]), dtype=torch.float32, device = 'cuda')
#     T2delta = torch.tensor(np.array([-0.00108903,  0.00056625 ,-0.00060884]), dtype=torch.float32, device = 'cuda')
#     #scale
    T2 = T2 * 15 + offset
    # T2 = torch.mm(T2.reshape(1,3), R2delta.T) + T2delta
    # R2 = torch.mm(R2, R2delta.T)
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")

# ([[[ 1.0000,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  1.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  1.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]],

#         [[ 0.2498, -0.1963,  0.9482, -0.2265],
#          [ 0.2211,  0.9649,  0.1415, -0.0051],
#          [-0.9427,  0.1743,  0.2844,  0.1766],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]])




#20240702
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



#contains functions and main for merging two gaussian models.
#playroom good
from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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



#utils:

# def qua_rotate(Q, R):
#     QM = quaternion_to_matrix(Q)
#     return torch.matmul(QM, R)
# def gaus_rotate(G, R):
#     G._xyz = torch.mm(G._xyz, R.t())
#     new_rotation = build_rotation(G._rotation)
#     new_rotation = R @ new_rotation
#     # G._rotation = extract_rotation_scipy(new_rotation)
#     G._rotation = matrix_to_quaternion(qua_rotate(G._rotation, R))
# def transform_shs(shs_feat, rotation_matrix):

#     ## rotate shs
#     P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) # switch axes: yzx -> xyz
#     rotation_matrix = rotation_matrix.cpu().numpy()
#     permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix @ P
#     rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotation_matrix))
    
#     # Construction coefficient
#     D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device='cuda', dtype=torch.float32)
#     D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device='cuda', dtype=torch.float32)
#     D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device='cuda', dtype=torch.float32)
#     #rotation of the shs features
#     one_degree_shs = shs_feat[:, 0:3]
#     one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     one_degree_shs = einsum(
#             D_1,
#             one_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat = torch.cat((one_degree_shs, shs_feat[:, 3:]), dim=1)

#     two_degree_shs = shs_feat[:, 3:8]
#     two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     two_degree_shs = einsum(
#             D_2,
#             two_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat[:, 3:8] = two_degree_shs

#     three_degree_shs = shs_feat[:, 8:15]
#     three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     three_degree_shs = einsum(
#             D_3,
#             three_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat[:, 8:15] = three_degree_shs

#     return shs_feat

# def gaus_translate(G, t):
#     G._xyz = G._xyz + t
# def gaus_append(g1, g2, gnew):
#     xyz_new = torch.cat((g1._xyz, g2._xyz), dim=0)
#     # print("this is xyz new", xyz_new)
#     features_dc_new = torch.cat((g1._features_dc, g2._features_dc), dim=0)
#     features_rest_new = torch.cat((g1._features_rest, g2._features_rest), dim=0)
#     scaling_new = torch.cat((g1._scaling, g2._scaling), dim=0)
#     rotation_new = torch.cat((g1._rotation, g2._rotation), dim=0)
#     opacity_new = torch.cat((g1._opacity, g2._opacity), dim=0)
#     gnew._xyz = xyz_new
#     gnew._features_dc = features_dc_new
#     gnew._features_rest = features_rest_new
#     gnew._scaling = scaling_new
#     gnew._rotation = rotation_new
#     gnew._opacity = opacity_new

# def gaus_transform(G, R, t):
#     rotate_by_matrix(G, R)
#     gaus_translate(G, t)


# def rotate_by_matrix(G, rotation_matrix, keep_sh_degree: bool = True):
#     # rotate xyz
#     G._xyz = torch.mm(G._xyz, rotation_matrix.t())
#     gaussian_rotation = build_rotation(G._rotation)
#     gaussian_rotation = rotation_matrix @ gaussian_rotation
#     # gaussian_rotation = torch.matmul(gaussian_rotation, rotation_matrix.t())
#     xyzw_quaternions = R.from_matrix(gaussian_rotation.cpu().detach().numpy()).as_quat()
#     wxyz_quaternions = xyzw_quaternions
#     wxyz_quaternions[:, [0, 1, 2, 3]] = wxyz_quaternions[:, [3, 0, 1, 2]]
#     rotations_from_matrix = torch.tensor(wxyz_quaternions, dtype=torch.float32, device="cuda")
#     G._rotation = rotations_from_matrix
#     if keep_sh_degree is False:
#         print("set sh_degree=0 when rotation transform enabled")
#         G.sh_degrees = 0
#     G._features_rest = transform_shs(G._features_rest, rotation_matrix)

# #write into class later
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([0.9938991069793701], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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

    #trucktwin_right:
    # [[ 0.1714, -0.0919,  0.9809,  0.0000],
    #     [ 0.3924,  0.9196,  0.0176,  0.0000],
    #     [-0.9037,  0.3819,  0.1936,  0.0000],
    #     [ 0.2364, -0.1797,  4.4662,  1.0000]]

    #truckscale right:
    #    [[ 0.2751, -0.3128,  0.9091,  0.0000],
    #     [ 0.3723,  0.9065,  0.1993,  0.0000],
    #     [-0.8864,  0.2836,  0.3658,  0.0000],
    #     [ 1.1816, -0.8960,  3.5893,  1.0000]]

        # [[ 0.0202, -0.3514,  0.9360,  0.0000],
    #     [ 0.4272,  0.8495,  0.3097,  0.0000],
    #     [-0.9039,  0.3936,  0.1672,  0.0000],
    #     [ 2.0416, -1.3046,  2.8997,  1.0000]]
    RI1tmp = np.array([[ 0.3164,  0.5144, -0.7970],
                  [-0.5658,  0.7767,  0.2767],
                  [ 0.7614,  0.3634,  0.5368]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([0.5026,  1.1814,  0.4488])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[-0.8694, -0.1858,  0.4578],
                 [0.3321,  0.4665,  0.8198],
                 [-0.3659,  0.8648, -0.3439]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([2.6519, -0.3652, -0.4066])
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

    #truck_scale case
    # [ 0.3595, -0.3023,  0.8828, -0.1953],
    #      [ 0.4284,  0.8939,  0.1316, -0.0555],
    #      [-0.8290,  0.3309,  0.4509,  0.1100],
    #      [ 0.0000,  0.0000,  0.0000,  1.0000]]
    #truckLR:
    # [[ 0.5586, -0.3078,  0.7702, -0.1802],
    #      [ 0.2557,  0.9473,  0.1931, -0.0490],
    #      [-0.7891,  0.0891,  0.6078,  0.1383],
    #      [ 0.0000,  0.0000,  0.0000,  1.0000]]


# [[[ 0.7481, -0.2017,  0.6322, -0.2223],
#          [ 0.2931,  0.9551, -0.0422,  0.0202],
#          [-0.5953,  0.2169,  0.7737,  0.1172],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]



# [[ 0.8982, -0.0013,  0.4396, -0.0796],
#          [-0.0379,  0.9960,  0.0804,  0.0041],
#          [-0.4380, -0.0889,  0.8946,  0.0401],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]


# [[ 6.6254e-01, -1.5983e-02,  7.4886e-01, -1.0150e-01],
#          [-7.9955e-02,  9.9255e-01,  9.1923e-02,  8.0308e-04],
#          [-7.4475e-01, -1.2078e-01,  6.5632e-01,  7.4551e-02],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]

#pr
# [[ 0.9646,  0.0270,  0.2624, -0.0495],
#          [-0.0223,  0.9995, -0.0210, -0.0072],
#          [-0.2629,  0.0144,  0.9647,  0.1238],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]
    # R2tmp  = torch.tensor([ 0.9820,  0.0064,  0.1127, -0.0459], device='cuda')
    # R2 = quaternion_to_matrix(R2tmp)

# [[ 0.9833, -0.1050, -0.1485, -0.1129],
#          [ 0.0794,  0.9824, -0.1691,  0.1156],
#          [ 0.1637,  0.1545,  0.9743,  0.0205],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]
    R2tmp = np.array( [[0.9833987355232239, -0.10865761339664459, -0.14532890915870667], [0.08402009308338165, 0.9825289845466614, -0.16606464982032776], [0.16083404421806335, 0.15109722316265106, 0.9753471612930298]]

)
    T2tmp = np.array([-0.1129, 0.1156,0.0205])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([-0.1060301885008812, -0.12232599407434464, 0.6570720076560974]), dtype=torch.float32, device = 'cuda')


#     R2delta = torch.tensor(np.array( [[ 9.99999998e-01, -5.50246651e-05,  3.05926189e-05],
#  [ 5.50255531e-05,  9.99999998e-01, -2.90237284e-05],
#  [-3.05910218e-05 , 2.90254117e-05  ,9.99999999e-01]]), dtype=torch.float32, device = 'cuda')
#     T2delta = torch.tensor(np.array([-0.00108903,  0.00056625 ,-0.00060884]), dtype=torch.float32, device = 'cuda')
#     #scale
    T2 = T2 * 12.199999809265137 + offset
    # T2 = torch.mm(T2.reshape(1,3), R2delta.T) + T2delta
    # R2 = torch.mm(R2, R2delta.T)
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")

# ([[[ 1.0000,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  1.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  1.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]],

#         [[ 0.2498, -0.1963,  0.9482, -0.2265],
#          [ 0.2211,  0.9649,  0.1415, -0.0051],
#          [-0.9427,  0.1743,  0.2844,  0.1766],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]])



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
def masked_l1_loss(img_new, img_ref, mask):
    mask = mask.unsqueeze(0)  
    masked_output = img_new * mask
    masked_gt = img_ref * mask
    diff = torch.abs(masked_output - masked_gt)
    sum_diff = torch.sum(diff)
    num_elements = torch.sum(mask)
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

#return list containing pair of most sim cam pos in order
# def RTpair(cam_dict1, cam_dict2):
#     res = []

#     return res


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



    #ini
    scale = torch.tensor([12.2], requires_grad=True, device='cuda')
    g2scale = torch.tensor([0.9900], requires_grad=True, device='cuda')
    offset = torch.tensor([0.0, 0.0, 0.0], requires_grad=True, device='cuda')

    rot = matrix_to_quaternion(R2)
    rot.to(dtype=torch.float32, device="cuda")
    rot.requires_grad_(True)

    best_param = (offset, rot, g2scale)

    lr_list = [0.001, 0.0001, 0.00001]

    for epoch in range(20000):
        #set up extrinsincs cam dicts:
        path1 = 'playground/PRmain2'
        cameras_extrinsic_file1 = os.path.join(path1, "sparse/0", "images.bin")
        cam_extrinsics1 = read_extrinsics_binary(cameras_extrinsic_file1)
        path2 = 'playground/PRstairs2'
        cameras_extrinsic_file2 = os.path.join(path2, "sparse/0", "images.bin")
        cam_extrinsics2 = read_extrinsics_binary(cameras_extrinsic_file2)

        #training initialization
        scale = torch.tensor([12.2], requires_grad=True, device='cuda')
        if epoch == 0:
            g2scale = torch.tensor([1.0], requires_grad=True, device='cuda')
            offset = torch.tensor([0.0, 0.0, 0.0], requires_grad=True, device='cuda')
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

        #set up ref cams
        cam_R_right2, cam_T_right2 = RTrandom(cam_extrinsics1)
        cam_R_right2 = cam_R_right2.T
        cam_R_right, cam_T_right = RTrandom(cam_extrinsics1)
        cam_R_right = cam_R_right.T

        # optimizer = Adam([scale], lr=0.01)
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        
        best_loss = 100


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

            cam_pos_right2 = myMiniCam3(450, 250, cam_R_right2, cam_T_right2, fovx, fovy, znear, zfar)

            # cam_R_new3 = (R2) @ (RI2).T @ cam_R_right2
            # cam_T_new3 = cam_T_right2 - ((TI2) @ ((R2).T) + 
            #                             (scale) * (T2) + offset) @ cam_R_new3

            cam_R_new3 = (RI1).T @ cam_R_right2 
            cam_T_new3 = cam_T_right2 - (TI1) @ (RI1).T @ cam_R_right2

            cam_pos_new2 = myMiniCam3(450, 250, cam_R_new3, cam_T_new3, fovx, fovy, znear, zfar)
            render2_new = render(cam_pos_new2, g2copy, pipe, background, scaling_modifer)["render"]
            img_new2 = render2_new
            if measure_blurriness(img_new2) > 0.05:
                break
            with torch.no_grad():
                render2_ref = render(cam_pos_right2, g1, pipe, background, scaling_modifer)
                img_ref2 = render2_ref["render"]
                mask2_ref = render2_ref["mask"]
                if measure_blurriness(img_ref2) > 0.05:
                    break
            
            

            weight = 0.0
            Ll12 = masked_l1_loss(img_new2, img_ref2, mask2_ref)
            loss2 = (1.0 - weight) * Ll12 + weight * (1.0 - ssim(img_new2, img_ref2))
            cam_pos_right = myMiniCam3(450, 250, cam_R_right, cam_T_right, fovx, fovy, znear, zfar)

            cam_R_new = (RI1).T @ cam_R_right 
            cam_T_new = cam_T_right - (TI1) @ (RI1).T @ cam_R_right
            cam_pos_new = myMiniCam3(450, 250, cam_R_new, cam_T_new, fovx, fovy, znear, zfar)
            render_new = render(cam_pos_new, g2copy, pipe, background, scaling_modifer)["render"]

            img_new = render_new
            if measure_blurriness(img_new) > 0.05:
                break
            with torch.no_grad():
                render_ref = render(cam_pos_right, g1, pipe, background, scaling_modifer)
                img_ref = render_ref["render"]
                if measure_blurriness(img_ref) > 0.05:
                    break
                mask_ref = render_ref["mask"]

            if iter % 300 == 0:
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


            Ll1 = masked_l1_loss(img_new, img_ref, mask_ref)
            loss1 = (1.0 - weight) * Ll1 + weight * (1.0 - ssim(img_new, img_ref))
            loss = loss1 + loss2
            torch.autograd.set_detect_anomaly(True)
    

            loss.backward()
            optimizer.step()

            # optimizer.zero_grad()
            # loss1.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            # loss2.backward()
            # optimizer.step()


            if iter % 100 == 0:
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
    RI1tmp = np.array([[ 0.3164,  0.5144, -0.7970],
                  [-0.5658,  0.7767,  0.2767],
                  [ 0.7614,  0.3634,  0.5368]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([0.5026,  1.1814,  0.4488])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[-0.8694, -0.1858,  0.4578],
                 [0.3321,  0.4665,  0.8198],
                 [-0.3659,  0.8648, -0.3439]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([2.6519, -0.3652, -0.4066])
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
    R2tmp = np.array( [[0.9833, -0.1050, -0.1485],
                        [0.0794,  0.9824, -0.1691],
                        [0.1637,  0.1545,  0.9743]]

)
    T2tmp = np.array([-0.1129, 0.1156,0.0205])


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





#truck three 1
# 934: Loss = 0.3215877413749695,
# Updated offset = [0.03650054708123207, 0.0009253827156499028, 0.03150207921862602], 
# [[0.9206866025924683, -0.27359268069267273, 0.2783580720424652], 
#  [0.2995023727416992, 0.9525442123413086, -0.05438578128814697], 
#  [-0.2502688467502594, 0.13344116508960724, 0.9589363932609558]] 
# {1.0420032739639282} {12.0}








#truck three 2  2024/7/6   final for truck, g2merge3 below
# 0.2800537943840027, 
# Updated offset = [0.047141481190919876, -0.002285347320139408, 0.014713875949382782], 
# [[0.9206722974777222, -0.2741420567035675, 0.27786436676979065], 
#  [0.2996361553668976, 0.9525808095932007, -0.05299083888530731], 
#  [-0.25016123056411743, 0.13204540312290192, 0.9591576457023621]] 
# {1.0505108833312988} {12.0}
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



#contains functions and main for merging two gaussian models.

from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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

# #write into class later
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([1.0505108833312988], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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

    #trucktwin_right:
    # [[ 0.1714, -0.0919,  0.9809,  0.0000],
    #     [ 0.3924,  0.9196,  0.0176,  0.0000],
    #     [-0.9037,  0.3819,  0.1936,  0.0000],
    #     [ 0.2364, -0.1797,  4.4662,  1.0000]]

    #truckscale right:
    #    [[ 0.2751, -0.3128,  0.9091,  0.0000],
    #     [ 0.3723,  0.9065,  0.1993,  0.0000],
    #     [-0.8864,  0.2836,  0.3658,  0.0000],
    #     [ 1.1816, -0.8960,  3.5893,  1.0000]]

        # [[ 0.0202, -0.3514,  0.9360,  0.0000],
    #     [ 0.4272,  0.8495,  0.3097,  0.0000],
    #     [-0.9039,  0.3936,  0.1672,  0.0000],
    #     [ 2.0416, -1.3046,  2.8997,  1.0000]]
    RI1tmp = np.array([[-0.1238, -0.2341,  0.9643],
                  [0.2545,  0.9318,  0.2589],
                  [-0.9591,  0.2775, -0.0558]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([2.9596, -0.9691,  3.5393])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[0.1761,  0.1369, -0.9748],
                 [0.1506,  0.9749,  0.1642],
                 [0.9728, -0.1757,  0.1510]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([-2.0454, -0.1671,  3.9528])
    TI2 = torch.tensor(TI2tmp, dtype=torch.float32, device="cuda")
    # TI1 = -torch.mm(RI1.t(), TI1.view(3,1)).view(1,3)
    # TI2 = -torch.mm(RI2.t(), TI2.view(3,1)).view(1,3)



    R0 = [[1,0,0],[0,1,0],[0,0,1]]
    T0 = [ 0, 0, 0]

    R1tmp = np.array(R0)
    T1tmp = np.array(T0)


    R2tmp = np.array( [[0.9206722974777222, -0.2741420567035675, 0.27786436676979065], 
 [0.2996361553668976, 0.9525808095932007, -0.05299083888530731], 
 [-0.25016123056411743, 0.13204540312290192, 0.9591576457023621]]

)
    T2tmp = np.array([-0.1084, -0.0047, 0.0224])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([0.047141481190919876, -0.002285347320139408, 0.014713875949382782]), dtype=torch.float32, device = 'cuda')


#     R2delta = torch.tensor(np.array( [[ 9.99999998e-01, -5.50246651e-05,  3.05926189e-05],
#  [ 5.50255531e-05,  9.99999998e-01, -2.90237284e-05],
#  [-3.05910218e-05 , 2.90254117e-05  ,9.99999999e-01]]), dtype=torch.float32, device = 'cuda')
#     T2delta = torch.tensor(np.array([-0.00108903,  0.00056625 ,-0.00060884]), dtype=torch.float32, device = 'cuda')
#     #scale
    T2 = T2 * 12 + offset
    # T2 = torch.mm(T2.reshape(1,3), R2delta.T) + T2delta
    # R2 = torch.mm(R2, R2delta.T)
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")

# ([[[ 1.0000,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  1.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  1.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]],

#         [[ 0.2498, -0.1963,  0.9482, -0.2265],
#          [ 0.2211,  0.9649,  0.1415, -0.0051],
#          [-0.9427,  0.1743,  0.2844,  0.1766],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]])




# 20240708 train
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



#contains functions and main for merging two gaussian models.

from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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

#Utils:
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([0.9656985998153687], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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

    #trucktwin_right:
    # [[ 0.1714, -0.0919,  0.9809,  0.0000],
    #     [ 0.3924,  0.9196,  0.0176,  0.0000],
    #     [-0.9037,  0.3819,  0.1936,  0.0000],
    #     [ 0.2364, -0.1797,  4.4662,  1.0000]]

    #truckscale right:
    #    [[ 0.2751, -0.3128,  0.9091,  0.0000],
    #     [ 0.3723,  0.9065,  0.1993,  0.0000],
    #     [-0.8864,  0.2836,  0.3658,  0.0000],
    #     [ 1.1816, -0.8960,  3.5893,  1.0000]]

        # [[ 0.0202, -0.3514,  0.9360,  0.0000],
    #     [ 0.4272,  0.8495,  0.3097,  0.0000],
    #     [-0.9039,  0.3936,  0.1672,  0.0000],
    #     [ 2.0416, -1.3046,  2.8997,  1.0000]]
    RI1tmp = np.array([[0.7709,  0.0198,  0.6367],
                 [0.0819,  0.9881, -0.1298],
                 [-0.6317,  0.1522,  0.7601]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([3.4646,  0.3195,  4.7689])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[0.1042,  0.1007, -0.9894],
                 [ 0.1596,  0.9803,  0.1166],
                 [0.9817, -0.1701,  0.0861]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([-1.4229, -0.0060,  5.9085])
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

    #truck_scale case
    # [ 0.3595, -0.3023,  0.8828, -0.1953],
    #      [ 0.4284,  0.8939,  0.1316, -0.0555],
    #      [-0.8290,  0.3309,  0.4509,  0.1100],
    #      [ 0.0000,  0.0000,  0.0000,  1.0000]]
    #truckLR:
    # [[ 0.5586, -0.3078,  0.7702, -0.1802],
    #      [ 0.2557,  0.9473,  0.1931, -0.0490],
    #      [-0.7891,  0.0891,  0.6078,  0.1383],
    #      [ 0.0000,  0.0000,  0.0000,  1.0000]]


# [[[ 0.7481, -0.2017,  0.6322, -0.2223],
#          [ 0.2931,  0.9551, -0.0422,  0.0202],
#          [-0.5953,  0.2169,  0.7737,  0.1172],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]



# [[ 0.8982, -0.0013,  0.4396, -0.0796],
#          [-0.0379,  0.9960,  0.0804,  0.0041],
#          [-0.4380, -0.0889,  0.8946,  0.0401],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]


# [[ 6.6254e-01, -1.5983e-02,  7.4886e-01, -1.0150e-01],
#          [-7.9955e-02,  9.9255e-01,  9.1923e-02,  8.0308e-04],
#          [-7.4475e-01, -1.2078e-01,  6.5632e-01,  7.4551e-02],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]

#pr
# [[ 0.9646,  0.0270,  0.2624, -0.0495],
#          [-0.0223,  0.9995, -0.0210, -0.0072],
#          [-0.2629,  0.0144,  0.9647,  0.1238],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]
    # R2tmp  = torch.tensor([ 0.9820,  0.0064,  0.1127, -0.0459], device='cuda')
    # R2 = quaternion_to_matrix(R2tmp)

# [[ 0.9833, -0.1050, -0.1485, -0.1129],
#          [ 0.0794,  0.9824, -0.1691,  0.1156],
#          [ 0.1637,  0.1545,  0.9743,  0.0205],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]



# [[ 0.9200, -0.2703,  0.2837, -0.1084],
#          [ 0.2966,  0.9535, -0.0534, -0.0047],
#          [-0.2561,  0.1333,  0.9574,  0.0224],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]

    R2tmp = np.array( [[0.696917712688446, -0.04208100959658623, 0.7159153819084167], [-0.05848004296422005, 0.9916176795959473, 0.11521480232477188], [-0.7147627472877502, -0.12216200679540634, 0.688615083694458]]

)
    T2tmp = np.array([-1.0150e-01,8.0308e-04,7.4551e-02])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([-0.743768572807312, -0.08816484361886978, 0.4554275572299957]), dtype=torch.float32, device = 'cuda')


#     R2delta = torch.tensor(np.array( [[ 9.99999998e-01, -5.50246651e-05,  3.05926189e-05],
#  [ 5.50255531e-05,  9.99999998e-01, -2.90237284e-05],
#  [-3.05910218e-05 , 2.90254117e-05  ,9.99999999e-01]]), dtype=torch.float32, device = 'cuda')
#     T2delta = torch.tensor(np.array([-0.00108903,  0.00056625 ,-0.00060884]), dtype=torch.float32, device = 'cuda')
#     #scale
    T2 = T2 * 12 + offset
    # T2 = torch.mm(T2.reshape(1,3), R2delta.T) + T2delta
    # R2 = torch.mm(R2, R2delta.T)
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")

# ([[[ 1.0000,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  1.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  1.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]],

#         [[ 0.2498, -0.1963,  0.9482, -0.2265],
#          [ 0.2211,  0.9649,  0.1415, -0.0051],
#          [-0.9427,  0.1743,  0.2844,  0.1766],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]])


#workroom 1 and low: edge case, cannot run optimization:
from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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

#Utils:
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([1.5407519340515137], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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
    # safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    RI1tmp = np.array([[ 0.8061,  0.1577, -0.5704],
        [-0.1009,  0.9864,  0.1301],
        [ 0.5832, -0.0473,  0.8110]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([-4.3178, -0.3215,  0.7372])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[ 0.8718, -0.1394,  0.4696],
        [ 0.1724,  0.9846, -0.0278],
        [-0.4585,  0.1052,  0.8824]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([ 3.3400, -0.7463,  2.2941])
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


# [[[ 0.8233,  0.0921, -0.5601, -0.2117],
#          [-0.0508,  0.9947,  0.0889,  0.0261],
#          [ 0.5654, -0.0447,  0.8236, -0.0729],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]

    R2tmp = np.array([[0.8233,  0.0921, -0.5601], 
                      [-0.0508,  0.9947,  0.0889], 
                      [0.5654, -0.0447,  0.8236]]

)
    T2tmp = np.array([-0.2117,0.0261,-0.0729])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([-0.030129723250865936, -0.9599840641021729, -1.4177138805389404]), dtype=torch.float32, device = 'cuda')


    T2 = T2 * 50 + offset
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")






#workroom 1n and 2high, relative good align:
from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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

#Utils:
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([1.2279915809631348], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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
    # safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    RI1tmp = np.array([[ 0.9265, -0.0727,  0.3691],
        [ 0.1440,  0.9749, -0.1695],
        [-0.3475,  0.2102,  0.9138]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([-1.1670,  0.7026,  0.2016])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[ 0.9732,  0.0346, -0.2274],
        [-0.0975,  0.9575, -0.2714],
        [ 0.2083,  0.2862,  0.9352]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([-3.9852,  0.6341, -3.3516])
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


# [[ 0.9935,  0.0489,  0.1027,  0.1224],
#          [-0.0364,  0.9921, -0.1204, -0.0255],
#          [-0.1078,  0.1159,  0.9874,  0.1481],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]])

    R2tmp = np.array([[0.9804311990737915, 0.0637773796916008, 0.186244934797287], [-0.0459735207259655, 0.9940845966339111, -0.0983986034989357], [-0.1914188116788864, 0.0879107192158699, 0.9775635600090027]]

)
    T2tmp = np.array([0.1224, -0.0255, 0.1481])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([1.5255544185638428, -0.5759915709495544, 2.789480686187744]), dtype=torch.float32, device = 'cuda')


    T2 = T2 * 10 + offset
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")






#workroom 1n and 2med:
from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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

#Utils:
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([1.03271484375], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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
    # safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    RI1tmp = np.array([[ 0.9990, -0.0436,  0.0114],
        [ 0.0404,  0.9789,  0.2004],
        [-0.0199, -0.1998,  0.9796]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([-4.2019, -0.8331,  1.3240])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[ 0.9265, -0.0727,  0.3691],
        [ 0.1440,  0.9749, -0.1695],
        [-0.3475,  0.2102,  0.9138]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([-1.1670,  0.7026,  0.2016])
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


# [[ 0.7531,  0.0573, -0.6554, -0.0816],
#          [-0.1673,  0.9801, -0.1065, -0.0421],
#          [ 0.6363,  0.1898,  0.7477,  0.0227],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]])


    R2tmp = np.array( [[0.7845010161399841, 0.012300902046263218, -0.6200054883956909], [-0.13984425365924835, 0.9775586724281311, -0.15755201876163483], [0.6041537523269653, 0.2103039175271988, 0.7686159610748291]]

)
    T2tmp = np.array([-0.0816,-0.0421, 0.0227])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([0.5571436882019043, 1.1806533336639404, -1.4584752321243286]), dtype=torch.float32, device = 'cuda')


    T2 = T2 * 10 + offset
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")




#dra 1-2low not good:
from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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

#Utils:
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([0.8], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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
    # safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    RI1tmp = np.array([[ 0.1083,  0.5066, -0.8554],
        [-0.2595,  0.8450,  0.4676],
        [ 0.9596,  0.1713,  0.2230]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([-3.5420, -0.9567,  1.9176])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[ 0.9814, -0.0662,  0.1803],
        [ 0.0638,  0.9978,  0.0191],
        [-0.1812, -0.0073,  0.9834]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([ 3.6508, -0.4224,  1.2554])
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



    R2tmp = np.array( [[-0.15321040153503418, 0.6831045746803284, -0.7140690088272095], [-0.4942459762096405, 0.5727567672729492, 0.6539652943611145], [0.8557145595550537, 0.4531201124191284, 0.2498694658279419]]

)
    T2tmp = np.array([0.3493,-0.3142,0.3241])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([-0.1948704719543457, -0.8653588891029358, -1.133754014968872]), dtype=torch.float32, device = 'cuda')


    T2 = T2 * 19 + offset
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")





#dra 1-2med: not good:
from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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

#Utils:
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([1.6255137920379639], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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
    # safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    RI1tmp = np.array([[ 0.9937,  0.0870, -0.0702],
        [-0.0809,  0.9930,  0.0864],
        [ 0.0772, -0.0801,  0.9938]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([1.5424, 0.3364, 0.3719])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[ 0.9443, -0.2563,  0.2064],
        [ 0.2622,  0.9650, -0.0014],
        [-0.1988,  0.0554,  0.9785]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([ 1.0226, -0.7055,  1.3249])
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




#  [[-0.9546,  0.2510, -0.1605,  0.0579],
#          [-0.0882,  0.2765,  0.9569, -0.4193],
#          [ 0.2845,  0.9277, -0.2418,  0.5381],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]

    R2tmp = np.array( [[-0.946298360824585, 0.2878122925758362, -0.1472524106502533], [-0.03758164867758751, 0.3544566035270691, 0.9343169927597046], [0.321102499961853, 0.8896766901016235, -0.3246053457260132]]

)
    T2tmp = np.array([0.0579,-0.4193,0.5381])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([0.19371956586837769, 0.5625572204589844, -0.20439930260181427]), dtype=torch.float32, device = 'cuda')


    T2 = T2 * 12 + offset
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")




#final with image comparison
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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([1.0505108833312988], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)

    fovx = 0.812657831303291
    fovy = 0.5579197285849142
    znear = 0.01
    zfar = 100.0






    scaling_modifer = 1.0
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    rotmat = np.array([[-0.8466, -0.0140,  0.5321],
                 [0.1371,  0.9602,  0.2434],
                 [-0.5143,  0.2790, -0.8110]])
    tvec = np.array([2.7114, -0.6500,  7.7526])

    
    rotmat = torch.tensor(rotmat, dtype=torch.float32, device="cuda")
    tvec = torch.tensor(tvec, dtype=torch.float32, device="cuda")


    cam_pos_right = myMiniCam3(900, 600, rotmat, tvec, fovx, fovy, znear, zfar)

    cam_R_new = (RI1).T @ rotmat 
    cam_T_new = tvec - (TI1) @ (RI1).T @ rotmat
    cam_pos_new = myMiniCam3(900, 600, cam_R_new, cam_T_new, fovx, fovy, znear, zfar)

    render_new = render(cam_pos_new, g2copy, pipe, background, scaling_modifer)["render"]
    render_ref = render(cam_pos_right, g1, pipe, background, scaling_modifer)["render"]
    render_aligned = render(cam_pos_new, gnew, pipe, background, scaling_modifer)["render"]

    image = render_new.permute(1, 2, 0).cpu().detach().numpy()
    image = cv2.cvtColor(image.astype('float32'), cv2.COLOR_RGB2BGR)
    cv2.imshow('Combined Image', image)
    cv2.waitKey(0)
    image = render_ref.permute(1, 2, 0).cpu().detach().numpy()
    image = cv2.cvtColor(image.astype('float32'), cv2.COLOR_RGB2BGR)
    cv2.imshow('Combined Image', image)
    cv2.waitKey(0)
    image = render_aligned.permute(1, 2, 0).cpu().detach().numpy()
    image = cv2.cvtColor(image.astype('float32'), cv2.COLOR_RGB2BGR)
    cv2.imshow('Combined Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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
    # safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    RI1tmp = np.array([[-0.1238, -0.2341,  0.9643],
                  [0.2545,  0.9318,  0.2589],
                  [-0.9591,  0.2775, -0.0558]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([2.9596, -0.9691,  3.5393])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[0.1761,  0.1369, -0.9748],
                 [0.1506,  0.9749,  0.1642],
                 [0.9728, -0.1757,  0.1510]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([-2.0454, -0.1671,  3.9528])
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




#  [[-0.9546,  0.2510, -0.1605,  0.0579],
#          [-0.0882,  0.2765,  0.9569, -0.4193],
#          [ 0.2845,  0.9277, -0.2418,  0.5381],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]

    R2tmp = np.array( [[0.9206722974777222, -0.2741420567035675, 0.27786436676979065], 
 [0.2996361553668976, 0.9525808095932007, -0.05299083888530731], 
 [-0.25016123056411743, 0.13204540312290192, 0.9591576457023621]]

)
    T2tmp = np.array([-0.1084, -0.0047, 0.0224])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([0.047141481190919876, -0.002285347320139408, 0.014713875949382782]), dtype=torch.float32, device = 'cuda')


    T2 = T2 * 12 + offset
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")




#colmap truck
from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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




# #write into class later
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


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
    gaus_transform(g1copy, RI1, TI1)
    gaus_transform(g2copy, RI2, TI2)
    
    
    g1scale = torch.tensor([1 / 1.207], dtype=torch.float32, device="cuda")
    rescale(g1copy, g1scale)
    g2scale = torch.tensor([1 / 1.149], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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

    RI1tmp = np.array([[ 0.9999,  0.0030,  0.0150],
        [-0.0053,  0.9881,  0.1539],
        [-0.0144, -0.1539,  0.9880]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([-0.1395,  0.1730, -2.4476])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")

    RI2tmp =  np.array([[-0.9595,  0.0311, -0.2799],
        [-0.0221,  0.9825,  0.1850],
        [ 0.2807,  0.1837, -0.9420]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([ 1.4802, -0.1773,  2.3070])
    TI2 = torch.tensor(TI2tmp, dtype=torch.float32, device="cuda")
    # TI1 = -torch.mm(RI1.t(), TI1.view(3,1)).view(1,3)
    # TI2 = -torch.mm(RI2.t(), TI2.view(3,1)).view(1,3)



    R0 = [[1,0,0],[0,1,0],[0,0,1]]
    T0 = [ 0, 0, 0]

    R1tmp = np.array(R0)
    T1tmp = np.array(T0)

    R2tmp = np.array( [[0.9833987355232239, -0.10865761339664459, -0.14532890915870667], 
                       [0.08402009308338165, 0.9825289845466614, -0.16606464982032776], 
                       [0.16083404421806335, 0.15109722316265106, 0.9753471612930298]]

)
    T2tmp = np.array([-0.00251219 -0.0069958  -0.0017439 ])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([0, 0, 0]), dtype=torch.float32, device = 'cuda')


    T2 = T2 + offset
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")




#workroom mid overlap almost perfect align:
from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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

#Utils:
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([0.861200213432312], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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
    # safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    RI1tmp = np.array([[ 0.5906, -0.5576,  0.5833],
        [ 0.1520,  0.7868,  0.5982],
        [-0.7925, -0.2646,  0.5495]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([ 3.8499,  3.3470, -3.9914])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[ 0.3082,  0.5335, -0.7876],
        [-0.2065,  0.8457,  0.4921],
        [ 0.9286,  0.0110,  0.3708]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([ 3.7501,  2.1876, -5.1780])
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


# [[ 0.9950, -0.0944, -0.0312,  0.0311],
#          [ 0.0861,  0.9751, -0.2045,  0.0272],
#          [ 0.0497,  0.2008,  0.9784,  0.0099],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]

    R2tmp = np.array([[0.9951345324516296, -0.09360852092504501, -0.030736181885004044], [0.08692999184131622, 0.9810299873352051, -0.17327234148979187], [0.04637288674712181, 0.1697573959827423, 0.9843941926956177]]

)
    T2tmp = np.array([0.0311, 0.0272, 0.0099])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array( [-0.11754917353391647, -0.023376677185297012, -0.07936595380306244]), dtype=torch.float32, device = 'cuda')


    T2 = T2 * 10 + offset
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")


# [[ 0.9876, -0.1211,  0.0997, -0.0218],
#          [ 0.1419,  0.9607, -0.2388,  0.0638],
#          [-0.0669,  0.2500,  0.9659,  0.0326],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]





#workroom high overlap, almost perfect:
from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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

#Utils:
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([0.8261324763298035], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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
    # safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    RI1tmp = np.array([[ 0.7860,  0.2260, -0.5754],
        [-0.1612,  0.9735,  0.1622],
        [ 0.5968, -0.0347,  0.8016]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([-1.6386, -0.1818,  0.6761])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[-0.7082,  0.3163, -0.6312],
        [-0.0842,  0.8498,  0.5203],
        [ 0.7010,  0.4216, -0.5752]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([-1.1165,  0.3818, -0.3991])
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


# [[ 0.9825, -0.1027,  0.1555, -0.0250],
#          [ 0.1074,  0.9940, -0.0217, -0.0418],
#          [-0.1524,  0.0380,  0.9876,  0.0582],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]

    R2tmp = np.array([[0.982968270778656, -0.10002736002206802, 0.15416844189167023], [0.10485421121120453, 0.9942103624343872, -0.02348168008029461], [-0.15092705190181732, 0.03924695774912834, 0.9877654910087585]]

)
    T2tmp = np.array([-0.0250,-0.0418, 0.0582])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array( [-0.24891099333763123, -0.1745477020740509, 0.6685101389884949]), dtype=torch.float32, device = 'cuda')


    T2 = T2 * 10 + offset
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")


# [[ 0.9876, -0.1211,  0.0997, -0.0218],
#          [ 0.1419,  0.9607, -0.2388,  0.0638],
#          [-0.0669,  0.2500,  0.9659,  0.0326],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]






#workroom ICP align, make it lower
from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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

#Utils:
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


#RI1, TI1, RI2, TI2, R1, T1, R2, T2
def merge(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, testing_iterations, saving_iterations, 
             checkpoint_iterations, checkpoint: str, checkpoint2: str, debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2, adjR, adjT):
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
    gaus_transform(g2copy, adjR, adjT)
    g2scale = torch.tensor([0.8], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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
    # safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    RI1tmp = np.array([[ 0.9845, -0.1054,  0.1402],
        [ 0.0968,  0.9930,  0.0670],
        [-0.1463, -0.0524,  0.9879]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([-0.9779,  0.1564, -0.0101])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[ 0.1627,  0.3500, -0.9225],
        [-0.1494,  0.9329,  0.3277],
        [ 0.9753,  0.0845,  0.2040]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([ 2.0282, -1.0095,  2.5417])
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


# [[[ 0.8233,  0.0921, -0.5601, -0.2117],
#          [-0.0508,  0.9947,  0.0889,  0.0261],
#          [ 0.5654, -0.0447,  0.8236, -0.0729],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]

    R2tmp = np.array( [[0.9560, -0.1253,  0.2654],
                        [0.1289,  0.9916,  0.0039],
                        [-0.2636,  0.0305,  0.9641]]

)
    T2tmp = np.array([-0.1618, 0.0286, -0.0588])


    adjRt = np.array([[ 1.00127146,  0.00362443, -0.00160164],
 [-0.00362138,  1.00127094  ,0.00190667],
 [ 0.00160853 ,-0.00190086 , 1.00127621]])
    adjTt = np.array([ 0.02231667, -0.07911994, -0.00583464])
    adjR = torch.tensor(adjRt, dtype=torch.float32, device = 'cuda') 
    adjT = torch.tensor(adjTt, dtype=torch.float32, device = 'cuda') 
    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([0, 0, 0]), dtype=torch.float32, device = 'cuda')


    T2 = T2 * 25 + offset
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2, adjR, adjT)

    print("\nTraining complete.")





#workroom 1 and high COLMAP alignment, not very good
from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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




# #write into class later
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


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
    gaus_transform(g1copy, RI1, TI1)
    gaus_transform(g2copy, RI2, TI2)
    
    
    g1scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    rescale(g1copy, g1scale)
    g2scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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

    RI1tmp = np.array([[ 0.8963,  0.1893, -0.4011],
        [-0.3612,  0.8365, -0.4122],
        [ 0.2575,  0.5143,  0.8180]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([ 0.0084,  0.3772, -0.1998])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")

    RI2tmp =  np.array([[ 0.1525, -0.1570,  0.9757],
        [ 0.7313,  0.6820, -0.0046],
        [-0.6648,  0.7143,  0.2189]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([ 0.1421, -0.4860,  0.4427])
    TI2 = torch.tensor(TI2tmp, dtype=torch.float32, device="cuda")
    # TI1 = -torch.mm(RI1.t(), TI1.view(3,1)).view(1,3)
    # TI2 = -torch.mm(RI2.t(), TI2.view(3,1)).view(1,3)



    R0 = [[1,0,0],[0,1,0],[0,0,1]]
    T0 = [ 0, 0, 0]

    R1tmp = np.array(R0)
    T1tmp = np.array(T0)

    R2tmp = np.array( [[0.9833987355232239, -0.10865761339664459, -0.14532890915870667], 
                       [0.08402009308338165, 0.9825289845466614, -0.16606464982032776], 
                       [0.16083404421806335, 0.15109722316265106, 0.9753471612930298]]

)
    T2tmp = np.array([-0.00251219 -0.0069958  -0.0017439 ])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([0, 0, 0]), dtype=torch.float32, device = 'cuda')


    T2 = T2 + offset
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")







#multi 2 and 1: good init
from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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



#utils:

# def qua_rotate(Q, R):
#     QM = quaternion_to_matrix(Q)
#     return torch.matmul(QM, R)
# def gaus_rotate(G, R):
#     G._xyz = torch.mm(G._xyz, R.t())
#     new_rotation = build_rotation(G._rotation)
#     new_rotation = R @ new_rotation
#     # G._rotation = extract_rotation_scipy(new_rotation)
#     G._rotation = matrix_to_quaternion(qua_rotate(G._rotation, R))
# def transform_shs(shs_feat, rotation_matrix):

#     ## rotate shs
#     P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) # switch axes: yzx -> xyz
#     rotation_matrix = rotation_matrix.cpu().numpy()
#     permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix @ P
#     rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotation_matrix))
    
#     # Construction coefficient
#     D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device='cuda', dtype=torch.float32)
#     D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device='cuda', dtype=torch.float32)
#     D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device='cuda', dtype=torch.float32)
#     #rotation of the shs features
#     one_degree_shs = shs_feat[:, 0:3]
#     one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     one_degree_shs = einsum(
#             D_1,
#             one_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat = torch.cat((one_degree_shs, shs_feat[:, 3:]), dim=1)

#     two_degree_shs = shs_feat[:, 3:8]
#     two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     two_degree_shs = einsum(
#             D_2,
#             two_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat[:, 3:8] = two_degree_shs

#     three_degree_shs = shs_feat[:, 8:15]
#     three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     three_degree_shs = einsum(
#             D_3,
#             three_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat[:, 8:15] = three_degree_shs

#     return shs_feat

# def gaus_translate(G, t):
#     G._xyz = G._xyz + t
# def gaus_append(g1, g2, gnew):
#     xyz_new = torch.cat((g1._xyz, g2._xyz), dim=0)
#     # print("this is xyz new", xyz_new)
#     features_dc_new = torch.cat((g1._features_dc, g2._features_dc), dim=0)
#     features_rest_new = torch.cat((g1._features_rest, g2._features_rest), dim=0)
#     scaling_new = torch.cat((g1._scaling, g2._scaling), dim=0)
#     rotation_new = torch.cat((g1._rotation, g2._rotation), dim=0)
#     opacity_new = torch.cat((g1._opacity, g2._opacity), dim=0)
#     gnew._xyz = xyz_new
#     gnew._features_dc = features_dc_new
#     gnew._features_rest = features_rest_new
#     gnew._scaling = scaling_new
#     gnew._rotation = rotation_new
#     gnew._opacity = opacity_new

# def gaus_transform(G, R, t):
#     rotate_by_matrix(G, R)
#     gaus_translate(G, t)


# def rotate_by_matrix(G, rotation_matrix, keep_sh_degree: bool = True):
#     # rotate xyz
#     G._xyz = torch.mm(G._xyz, rotation_matrix.t())
#     gaussian_rotation = build_rotation(G._rotation)
#     gaussian_rotation = rotation_matrix @ gaussian_rotation
#     # gaussian_rotation = torch.matmul(gaussian_rotation, rotation_matrix.t())
#     xyzw_quaternions = R.from_matrix(gaussian_rotation.cpu().detach().numpy()).as_quat()
#     wxyz_quaternions = xyzw_quaternions
#     wxyz_quaternions[:, [0, 1, 2, 3]] = wxyz_quaternions[:, [3, 0, 1, 2]]
#     rotations_from_matrix = torch.tensor(wxyz_quaternions, dtype=torch.float32, device="cuda")
#     G._rotation = rotations_from_matrix
#     if keep_sh_degree is False:
#         print("set sh_degree=0 when rotation transform enabled")
#         G.sh_degrees = 0
#     G._features_rest = transform_shs(G._features_rest, rotation_matrix)

# #write into class later
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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

    #trucktwin_right:
    # [[ 0.1714, -0.0919,  0.9809,  0.0000],
    #     [ 0.3924,  0.9196,  0.0176,  0.0000],
    #     [-0.9037,  0.3819,  0.1936,  0.0000],
    #     [ 0.2364, -0.1797,  4.4662,  1.0000]]

    #truckscale right:
    #    [[ 0.2751, -0.3128,  0.9091,  0.0000],
    #     [ 0.3723,  0.9065,  0.1993,  0.0000],
    #     [-0.8864,  0.2836,  0.3658,  0.0000],
    #     [ 1.1816, -0.8960,  3.5893,  1.0000]]

        # [[ 0.0202, -0.3514,  0.9360,  0.0000],
    #     [ 0.4272,  0.8495,  0.3097,  0.0000],
    #     [-0.9039,  0.3936,  0.1672,  0.0000],
    #     [ 2.0416, -1.3046,  2.8997,  1.0000]]
    RI1tmp = np.array([[-0.7120,  0.3921, -0.5825],
        [-0.3285,  0.5472,  0.7698],
        [ 0.6206,  0.7395, -0.2608]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([-1.7689,  1.3597, -2.1275])
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


# [[ 0.9971,  0.0166, -0.0741,  0.0279],
#          [-0.0089,  0.9947,  0.1024, -0.0073],
#          [ 0.0754, -0.1014,  0.9920,  0.0178],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]]

# [[ 0.9787, -0.0849,  0.1867,  0.0773],
#          [ 0.0887,  0.9960, -0.0120, -0.0175],
#          [-0.1849,  0.0283,  0.9823,  0.0295],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]

    R2tmp = np.array( [[0.9787, -0.0849,  0.1867], 
                       [ 0.0887,  0.9960, -0.0120], 
                       [-0.1849,  0.0283,  0.9823]])
    T2tmp = np.array([0.0773, -0.0175, 0.0295])
    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([0, 0, 0]), dtype=torch.float32, device = 'cuda')


#     R2delta = torch.tensor(np.array( [[ 9.99999998e-01, -5.50246651e-05,  3.05926189e-05],
#  [ 5.50255531e-05,  9.99999998e-01, -2.90237284e-05],
#  [-3.05910218e-05 , 2.90254117e-05  ,9.99999999e-01]]), dtype=torch.float32, device = 'cuda')
#     T2delta = torch.tensor(np.array([-0.00108903,  0.00056625 ,-0.00060884]), dtype=torch.float32, device = 'cuda')
#     #scale
    T2 = T2 * 15 + offset
    # T2 = torch.mm(T2.reshape(1,3), R2delta.T) + T2delta
    # R2 = torch.mm(R2, R2delta.T)
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")




#multi 2 and 3: PC viz:

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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([0.95], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
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

    RI1tmp = np.array([[ 0.9558, -0.2941, -0.0031],
        [ 0.2816,  0.9117,  0.2992],
        [-0.0852, -0.2868,  0.9542]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([ 2.1698,  4.7415, -2.3505])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")



    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[ 0.9548,  0.2966, -0.0211],
        [-0.2880,  0.9402,  0.1816],
        [ 0.0737, -0.1673,  0.9831]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([ 3.7291,  1.6035, -2.1273])
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

    R2tmp = np.array( [[0.9271,  0.3681, -0.0707], 
                       [-0.3719,  0.9268, -0.0522], 
                       [ 0.0463,  0.0747,  0.9961]])
    T2tmp = np.array([0.1440, 0.0096, 0.0250])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([0, 0, 0]), dtype=torch.float32, device = 'cuda')




#     R2delta = torch.tensor(np.array( [[ 9.99999998e-01, -5.50246651e-05,  3.05926189e-05],
#  [ 5.50255531e-05,  9.99999998e-01, -2.90237284e-05],
#  [-3.05910218e-05 , 2.90254117e-05  ,9.99999999e-01]]), dtype=torch.float32, device = 'cuda')
#     T2delta = torch.tensor(np.array([-0.00108903,  0.00056625 ,-0.00060884]), dtype=torch.float32, device = 'cuda')
#     #scale
    T2 = T2 * 18 + offset
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")




#gsmerge multi 2 3 best

from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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



#utils:

# def qua_rotate(Q, R):
#     QM = quaternion_to_matrix(Q)
#     return torch.matmul(QM, R)
# def gaus_rotate(G, R):
#     G._xyz = torch.mm(G._xyz, R.t())
#     new_rotation = build_rotation(G._rotation)
#     new_rotation = R @ new_rotation
#     # G._rotation = extract_rotation_scipy(new_rotation)
#     G._rotation = matrix_to_quaternion(qua_rotate(G._rotation, R))
# def transform_shs(shs_feat, rotation_matrix):

#     ## rotate shs
#     P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) # switch axes: yzx -> xyz
#     rotation_matrix = rotation_matrix.cpu().numpy()
#     permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix @ P
#     rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotation_matrix))
    
#     # Construction coefficient
#     D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device='cuda', dtype=torch.float32)
#     D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device='cuda', dtype=torch.float32)
#     D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device='cuda', dtype=torch.float32)
#     #rotation of the shs features
#     one_degree_shs = shs_feat[:, 0:3]
#     one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     one_degree_shs = einsum(
#             D_1,
#             one_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat = torch.cat((one_degree_shs, shs_feat[:, 3:]), dim=1)

#     two_degree_shs = shs_feat[:, 3:8]
#     two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     two_degree_shs = einsum(
#             D_2,
#             two_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat[:, 3:8] = two_degree_shs

#     three_degree_shs = shs_feat[:, 8:15]
#     three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     three_degree_shs = einsum(
#             D_3,
#             three_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat[:, 8:15] = three_degree_shs

#     return shs_feat

# def gaus_translate(G, t):
#     G._xyz = G._xyz + t
# def gaus_append(g1, g2, gnew):
#     xyz_new = torch.cat((g1._xyz, g2._xyz), dim=0)
#     # print("this is xyz new", xyz_new)
#     features_dc_new = torch.cat((g1._features_dc, g2._features_dc), dim=0)
#     features_rest_new = torch.cat((g1._features_rest, g2._features_rest), dim=0)
#     scaling_new = torch.cat((g1._scaling, g2._scaling), dim=0)
#     rotation_new = torch.cat((g1._rotation, g2._rotation), dim=0)
#     opacity_new = torch.cat((g1._opacity, g2._opacity), dim=0)
#     gnew._xyz = xyz_new
#     gnew._features_dc = features_dc_new
#     gnew._features_rest = features_rest_new
#     gnew._scaling = scaling_new
#     gnew._rotation = rotation_new
#     gnew._opacity = opacity_new

# def gaus_transform(G, R, t):
#     rotate_by_matrix(G, R)
#     gaus_translate(G, t)


# def rotate_by_matrix(G, rotation_matrix, keep_sh_degree: bool = True):
#     # rotate xyz
#     G._xyz = torch.mm(G._xyz, rotation_matrix.t())
#     gaussian_rotation = build_rotation(G._rotation)
#     gaussian_rotation = rotation_matrix @ gaussian_rotation
#     # gaussian_rotation = torch.matmul(gaussian_rotation, rotation_matrix.t())
#     xyzw_quaternions = R.from_matrix(gaussian_rotation.cpu().detach().numpy()).as_quat()
#     wxyz_quaternions = xyzw_quaternions
#     wxyz_quaternions[:, [0, 1, 2, 3]] = wxyz_quaternions[:, [3, 0, 1, 2]]
#     rotations_from_matrix = torch.tensor(wxyz_quaternions, dtype=torch.float32, device="cuda")
#     G._rotation = rotations_from_matrix
#     if keep_sh_degree is False:
#         print("set sh_degree=0 when rotation transform enabled")
#         G.sh_degrees = 0
#     G._features_rest = transform_shs(G._features_rest, rotation_matrix)

# #write into class later
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([1.0556305646896362], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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

    #trucktwin_right:
    # [[ 0.1714, -0.0919,  0.9809,  0.0000],
    #     [ 0.3924,  0.9196,  0.0176,  0.0000],
    #     [-0.9037,  0.3819,  0.1936,  0.0000],
    #     [ 0.2364, -0.1797,  4.4662,  1.0000]]

    #truckscale right:
    #    [[ 0.2751, -0.3128,  0.9091,  0.0000],
    #     [ 0.3723,  0.9065,  0.1993,  0.0000],
    #     [-0.8864,  0.2836,  0.3658,  0.0000],
    #     [ 1.1816, -0.8960,  3.5893,  1.0000]]

        # [[ 0.0202, -0.3514,  0.9360,  0.0000],
    #     [ 0.4272,  0.8495,  0.3097,  0.0000],
    #     [-0.9039,  0.3936,  0.1672,  0.0000],
    #     [ 2.0416, -1.3046,  2.8997,  1.0000]]
    RI1tmp = np.array([[ 0.9558, -0.2941, -0.0031],
        [ 0.2816,  0.9117,  0.2992],
        [-0.0852, -0.2868,  0.9542]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([ 2.1698,  4.7415, -2.3505])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[ 0.9548,  0.2966, -0.0211],
        [-0.2880,  0.9402,  0.1816],
        [ 0.0737, -0.1673,  0.9831]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([ 3.7291,  1.6035, -2.1273])
    TI2 = torch.tensor(TI2tmp, dtype=torch.float32, device="cuda")
    # TI1 = -torch.mm(RI1.t(), TI1.view(3,1)).view(1,3)
    # TI2 = -torch.mm(RI2.t(), TI2.view(3,1)).view(1,3)



    R0 = [[1,0,0],[0,1,0],[0,0,1]]
    T0 = [ 0, 0, 0]

    R1tmp = np.array(R0)
    T1tmp = np.array(T0)


    R2tmp = np.array( [[0.9254897236824036, 0.37496864795684814, -0.053547102957963943], [-0.37742307782173157, 0.9248538017272949, -0.046875108033418655], [0.03194654732942581, 0.0635923370718956, 0.9974644780158997]])
    T2tmp = np.array([0.1440, 0.0096, 0.0250])
    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([-0.327312171459198, -0.02915848046541214, -0.2542802691459656]), dtype=torch.float32, device = 'cuda')


#     R2delta = torch.tensor(np.array( [[ 9.99999998e-01, -5.50246651e-05,  3.05926189e-05],
#  [ 5.50255531e-05,  9.99999998e-01, -2.90237284e-05],
#  [-3.05910218e-05 , 2.90254117e-05  ,9.99999999e-01]]), dtype=torch.float32, device = 'cuda')
#     T2delta = torch.tensor(np.array([-0.00108903,  0.00056625 ,-0.00060884]), dtype=torch.float32, device = 'cuda')
#     #scale
    T2 = T2 * 18 + offset
    # T2 = torch.mm(T2.reshape(1,3), R2delta.T) + T2delta
    # R2 = torch.mm(R2, R2delta.T)
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")






#multi 23 best PC_viz
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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([1.0556305646896362], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
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

    RI1tmp = np.array([[ 0.9558, -0.2941, -0.0031],
        [ 0.2816,  0.9117,  0.2992],
        [-0.0852, -0.2868,  0.9542]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([ 2.1698,  4.7415, -2.3505])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")

    RI2tmp =  np.array([[ 0.9548,  0.2966, -0.0211],
        [-0.2880,  0.9402,  0.1816],
        [ 0.0737, -0.1673,  0.9831]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([ 3.7291,  1.6035, -2.1273])
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

    R2tmp = np.array( [[0.9254897236824036, 0.37496864795684814, -0.053547102957963943], [-0.37742307782173157, 0.9248538017272949, -0.046875108033418655], [0.03194654732942581, 0.0635923370718956, 0.9974644780158997]])
    T2tmp = np.array([0.1440, 0.0096, 0.0250])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([-0.327312171459198, -0.02915848046541214, -0.2542802691459656]), dtype=torch.float32, device = 'cuda')

    T2 = T2 * 18 + offset
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")





#multi 21 gsmerge  best
from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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



#utils:

# def qua_rotate(Q, R):
#     QM = quaternion_to_matrix(Q)
#     return torch.matmul(QM, R)
# def gaus_rotate(G, R):
#     G._xyz = torch.mm(G._xyz, R.t())
#     new_rotation = build_rotation(G._rotation)
#     new_rotation = R @ new_rotation
#     # G._rotation = extract_rotation_scipy(new_rotation)
#     G._rotation = matrix_to_quaternion(qua_rotate(G._rotation, R))
# def transform_shs(shs_feat, rotation_matrix):

#     ## rotate shs
#     P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) # switch axes: yzx -> xyz
#     rotation_matrix = rotation_matrix.cpu().numpy()
#     permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix @ P
#     rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotation_matrix))
    
#     # Construction coefficient
#     D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device='cuda', dtype=torch.float32)
#     D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device='cuda', dtype=torch.float32)
#     D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device='cuda', dtype=torch.float32)
#     #rotation of the shs features
#     one_degree_shs = shs_feat[:, 0:3]
#     one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     one_degree_shs = einsum(
#             D_1,
#             one_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat = torch.cat((one_degree_shs, shs_feat[:, 3:]), dim=1)

#     two_degree_shs = shs_feat[:, 3:8]
#     two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     two_degree_shs = einsum(
#             D_2,
#             two_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat[:, 3:8] = two_degree_shs

#     three_degree_shs = shs_feat[:, 8:15]
#     three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     three_degree_shs = einsum(
#             D_3,
#             three_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat[:, 8:15] = three_degree_shs

#     return shs_feat

# def gaus_translate(G, t):
#     G._xyz = G._xyz + t
# def gaus_append(g1, g2, gnew):
#     xyz_new = torch.cat((g1._xyz, g2._xyz), dim=0)
#     # print("this is xyz new", xyz_new)
#     features_dc_new = torch.cat((g1._features_dc, g2._features_dc), dim=0)
#     features_rest_new = torch.cat((g1._features_rest, g2._features_rest), dim=0)
#     scaling_new = torch.cat((g1._scaling, g2._scaling), dim=0)
#     rotation_new = torch.cat((g1._rotation, g2._rotation), dim=0)
#     opacity_new = torch.cat((g1._opacity, g2._opacity), dim=0)
#     gnew._xyz = xyz_new
#     gnew._features_dc = features_dc_new
#     gnew._features_rest = features_rest_new
#     gnew._scaling = scaling_new
#     gnew._rotation = rotation_new
#     gnew._opacity = opacity_new

# def gaus_transform(G, R, t):
#     rotate_by_matrix(G, R)
#     gaus_translate(G, t)


# def rotate_by_matrix(G, rotation_matrix, keep_sh_degree: bool = True):
#     # rotate xyz
#     G._xyz = torch.mm(G._xyz, rotation_matrix.t())
#     gaussian_rotation = build_rotation(G._rotation)
#     gaussian_rotation = rotation_matrix @ gaussian_rotation
#     # gaussian_rotation = torch.matmul(gaussian_rotation, rotation_matrix.t())
#     xyzw_quaternions = R.from_matrix(gaussian_rotation.cpu().detach().numpy()).as_quat()
#     wxyz_quaternions = xyzw_quaternions
#     wxyz_quaternions[:, [0, 1, 2, 3]] = wxyz_quaternions[:, [3, 0, 1, 2]]
#     rotations_from_matrix = torch.tensor(wxyz_quaternions, dtype=torch.float32, device="cuda")
#     G._rotation = rotations_from_matrix
#     if keep_sh_degree is False:
#         print("set sh_degree=0 when rotation transform enabled")
#         G.sh_degrees = 0
#     G._features_rest = transform_shs(G._features_rest, rotation_matrix)

# #write into class later
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([1.9350299835205078], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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

    #trucktwin_right:
    # [[ 0.1714, -0.0919,  0.9809,  0.0000],
    #     [ 0.3924,  0.9196,  0.0176,  0.0000],
    #     [-0.9037,  0.3819,  0.1936,  0.0000],
    #     [ 0.2364, -0.1797,  4.4662,  1.0000]]

    #truckscale right:
    #    [[ 0.2751, -0.3128,  0.9091,  0.0000],
    #     [ 0.3723,  0.9065,  0.1993,  0.0000],
    #     [-0.8864,  0.2836,  0.3658,  0.0000],
    #     [ 1.1816, -0.8960,  3.5893,  1.0000]]

        # [[ 0.0202, -0.3514,  0.9360,  0.0000],
    #     [ 0.4272,  0.8495,  0.3097,  0.0000],
    #     [-0.9039,  0.3936,  0.1672,  0.0000],
    #     [ 2.0416, -1.3046,  2.8997,  1.0000]]
    RI1tmp = np.array([[ 0.1742,  0.7024, -0.6901],
        [-0.7600,  0.5415,  0.3594],
        [ 0.6261,  0.4619,  0.6282]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([ 6.8444,  1.8075, -1.9562])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp =  np.array([[ 0.6059, -0.6049,  0.5167],
        [ 0.6092,  0.7705,  0.1877],
        [-0.5117,  0.2011,  0.8353]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([ 1.7184,  0.3150, -1.2049])
    TI2 = torch.tensor(TI2tmp, dtype=torch.float32, device="cuda")
    # TI1 = -torch.mm(RI1.t(), TI1.view(3,1)).view(1,3)
    # TI2 = -torch.mm(RI2.t(), TI2.view(3,1)).view(1,3)



    R0 = [[1,0,0],[0,1,0],[0,0,1]]
    T0 = [ 0, 0, 0]

    R1tmp = np.array(R0)
    T1tmp = np.array(T0)

# [[ 0.8213, -0.4876,  0.2963, -0.1260],
#          [ 0.5097,  0.8603,  0.0028, -0.0339],
#          [-0.2563,  0.1488,  0.9551, -0.0138],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]

    R2tmp = np.array( [[0.8278205394744873, -0.4885037839412689, 0.2758210301399231], [0.5057887434959412, 0.8626025319099426, 0.009724809788167477], [-0.2426745444536209, 0.13145679235458374, 0.9611598253250122]])
    
    T2tmp = np.array([-0.1260, -0.0339,-0.0138])
    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([-0.02491196058690548, -0.02893947809934616, -0.03709326311945915]), dtype=torch.float32, device = 'cuda')


#     R2delta = torch.tensor(np.array( [[ 9.99999998e-01, -5.50246651e-05,  3.05926189e-05],
#  [ 5.50255531e-05,  9.99999998e-01, -2.90237284e-05],
#  [-3.05910218e-05 , 2.90254117e-05  ,9.99999999e-01]]), dtype=torch.float32, device = 'cuda')
#     T2delta = torch.tensor(np.array([-0.00108903,  0.00056625 ,-0.00060884]), dtype=torch.float32, device = 'cuda')
#     #scale
    T2 = T2 * 10 + offset
    # T2 = torch.mm(T2.reshape(1,3), R2delta.T) + T2delta
    # R2 = torch.mm(R2, R2delta.T)
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")






# command line for multi merge: python multi_gsMerge.py -s playground/train_right/ --start_checkpoint m2_21_model/gnew.pth --start_checkpoint2 output/m2_3_model/chkpnt10000.pth





cs version1 rough align:
from e3nn import o3
import einops
from einops import einsum
from myUtils import gaus_transform, gaus_append, gaus_copy, rescale


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

#Utils:
def gaus_copy(g, gnew):
    gnew.active_sh_degree = g.active_sh_degree
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity


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
    gaus_transform(g1copy, RI1.t(), TI1)
    gaus_transform(g1copy, R1, T1)
    gaus_transform(g2copy, RI2.t(), TI2)
    gaus_transform(g2copy, R2, T2)
    
    g2scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    rescale(g2copy, g2scale)
    gaus_append(g1copy, g2copy, gnew)
    print(g1.active_sh_degree, g2.active_sh_degree)
    print(gnew.active_sh_degree)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_iteration = 10000
    torch.save((gnew.capture(), test_iteration), "gnew.pth")
    print("save complete!")
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
                print(e)
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
    # safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)



# [[ 0.9791,  0.0492, -0.1973, -0.0000],
#         [-0.0780,  0.9869, -0.1414,  0.0000],
#         [ 0.1878,  0.1538,  0.9701,  0.0000],
#         [ 0.7222,  0.4061, -3.8769,  1.0000]]


    RI1tmp = np.array([[ 9.6280e-01,  5.2044e-02, -2.6517e-01],
        [-5.0157e-02,  9.9864e-01,  1.3887e-02],
        [ 2.6553e-01, -7.0159e-05,  9.6410e-01]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([ 3.0873,  0.3105, -2.3074])
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")


# [[[ 0.9822, -0.0210,  0.1869,  0.0027],
#          [ 0.0264,  0.9993, -0.0268, -0.0015],
#          [-0.1862,  0.0313,  0.9820, -0.0182],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]

    RI2tmp =  np.array([[ 0.9888,  0.1016,  0.1093],
        [-0.1280,  0.9540,  0.2710],
        [-0.0767, -0.2819,  0.9564]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([ 3.6159,  0.2146, -2.3983])
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


# [[ 0.9796, -0.0605, -0.1916,  0.0482],
#          [ 0.0544,  0.9978, -0.0368,  0.0040],
#          [ 0.1934,  0.0256,  0.9808, -0.0054],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]])
    R2tmp = np.array([[0.9822, -0.0210,  0.1869],
        [0.0264,  0.9993, -0.0268],
        [-0.1862,  0.0313,  0.9820]])


    T2tmp = np.array([0.0027, -0.0015, -0.0182])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([0 ,0, 0]), dtype=torch.float32, device = 'cuda')


    T2 = T2 * 14 + offset
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")




