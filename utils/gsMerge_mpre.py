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

    gaus_transform(gnew, RI1, - RI1 @ TI1)
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