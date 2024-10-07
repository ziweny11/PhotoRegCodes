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
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

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



    fovx = 1.3973215818405151 
    fovy = 0.872002899646759
    znear = 0.01
    zfar = 100.0
    qvec = np.array([0.982584, 0.0172863, 0.177767, 0.0512692])
    tvec1 = np.array([-3.0538, 3.07277, -4.15644])
    tvec2 = np.array([2.75374, -0.650844, 1.24358])
    tvec = np.array([-8.3768, -4.3394,  4.6934])
    tvec = torch.tensor(tvec, dtype=torch.float32, device="cuda")


    # Rc = qvec2rotmat(qvec)
    # Rc = torch.tensor(Rc, dtype=torch.float32, device="cuda")
# [[  0.6911,   0.3265,  -0.6448,  -0.0000],
#         [ -0.5080,   0.8541,  -0.1120,   0.0000],
#         [  0.5141,   0.4049,   0.7561,   0.0000],
#         [-10.4744,  -3.4600,  10.0819,   1.0000]]

    Rc = torch.tensor(np.array([[0.6981,  0.3828, -0.6051],
                                [-0.6052,  0.7671, -0.2129],
                                [0.3827,  0.5148,  0.7671]]), dtype=torch.float32, device="cuda")

    Rc = RI1.T @ Rc
    tvec = tvec - TI1 @ Rc
    T_opposite = -torch.matmul(Rc, tvec)
    custom_cam = myMiniCam3(979,543, Rc, tvec, fovx, fovy, znear, zfar)
    scaling_modifer = 1.0
    g1_img = render(custom_cam, g1copy, pipe, background, scaling_modifer)["render"]
    img = g1_img.permute(1,2,0).cpu().detach().numpy()
    image_g1 = cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2BGR)
    cv2.imshow('Image', image_g1)
    cv2.waitKey(0)
    g2_img = render(custom_cam, g2copy, pipe, background, scaling_modifer)["render"]
    img = g2_img.permute(1,2,0).cpu().detach().numpy()
    image_g2 = cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2BGR)
    cv2.imshow('Image', image_g2)
    cv2.waitKey(0)
    g12_img = render(custom_cam, gnew, pipe, background, scaling_modifer)["render"]
    img = g12_img.permute(1,2,0).cpu().detach().numpy()
    image_g3 = cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2BGR)
    cv2.imshow('Image', image_g3)
    cv2.waitKey(0)



    cv2.destroyAllWindows()


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


# [[ 0.9825, -0.1027,  0.1555, -0.0250],
#          [ 0.1074,  0.9940, -0.0217, -0.0418],
#          [-0.1524,  0.0380,  0.9876,  0.0582],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]


    R2tmp = np.array([[0.9951345324516296, -0.09360852092504501, -0.030736181885004044], [0.08692999184131622, 0.9810299873352051, -0.17327234148979187], [0.04637288674712181, 0.1697573959827423, 0.9843941926956177]]

)
    T2tmp = np.array([0.0311, 0.0272, 0.0099])

    R1 = torch.tensor(R0, dtype=torch.float32, device = 'cuda') 
    T1 = torch.tensor(T0, dtype=torch.float32, device = 'cuda')
    R2 = torch.tensor(R2tmp, dtype=torch.float32, device = 'cuda') 
    T2 = torch.tensor(T2tmp, dtype=torch.float32, device = 'cuda')
    offset = torch.tensor(np.array([-0.327312171459198, -0.02915848046541214, -0.2542802691459656]), dtype=torch.float32, device = 'cuda')

    T2 = T2 * 10 + offset
    merge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, RI1, TI1, RI2, TI2, R1, T1, R2, T2)

    print("\nTraining complete.")

0.865816, 0.169131, -0.413981, -0.224448
4.21953, 2.02385, -4.59366