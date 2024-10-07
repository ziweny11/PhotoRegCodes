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