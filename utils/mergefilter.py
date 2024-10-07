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
    g1copy.filter_xyz(nb = 30, r = 0.1)
    g2copy.filter_xyz(nb = 30, r = 0.1)
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
