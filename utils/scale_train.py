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

from utils.general_utils import build_rotation
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.optim import Adam
import torch.nn.functional as F
import os
from scene.cameras import MiniCam, Camera, myMiniCam2
from gsMergev import gaus_transform, gaus_append, gaus_copy
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


def myloss(img1, img2):
    # Convert images to grayscale using the luminosity method
    # Since the image format is [channels, height, width], index with 0, 1, 2 directly
    gray1 = 0.299 * img1[0, :, :] + 0.587 * img1[1, :, :] + 0.114 * img1[2, :, :]
    gray2 = 0.299 * img2[0, :, :] + 0.587 * img2[1, :, :] + 0.114 * img2[2, :, :]

    # Compute MSE loss
    loss = F.mse_loss(gray1, gray2)
    return loss


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint: str, checkpoint2: str, debug_from, 
             RI1, TI1, RI2, TI2, R1, T1, R2, T2):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)


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

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)


    scale = torch.tensor([16.9], requires_grad=True, device='cuda')
    offset = torch.tensor([0.0, 0.0, 0.0], requires_grad=True, device='cuda')
    #rotation = ???
    optimizer = Adam([offset], lr=0.001)
    # optimizer = torch.optim.RMSprop([scale], lr=0.01, alpha=0.99)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    for iter in range(1000):


        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         # custom_cam.printinfo()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gnew, pipe, background, scaling_modifer)["render"]
        #             # #check type
        #             # print("type of custom_cam:")
        #             # print(type(custom_cam), end="\n")
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #     except Exception as e:
        #         network_gui.conn = None




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


        gaus_transform(g1copy, RI1.t(), TI1)
        gaus_transform(g1copy, R1, T1)
        gaus_transform(g2copy, RI2.t(), TI2)
        gaus_transform(g2copy, R2, scale * T2 + offset)
        gaus_append(g1copy, g2copy, gnew)

        #gnew is the new gaussian model

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        #set cam pos 
        fovy = 0.8733320236206055
        fovx = 1.398024082183838
        znear = 0.008999999612569809
        zfar = 1100.0

            # [[ 0.5173, -0.1252,  0.8466, -0.0000],
            #         [ 0.2749,  0.9611, -0.0258, -0.0000],
            #         [-0.8105,  0.2461,  0.5316,  0.0000],
            #         [ 1.5961, -0.4203,  3.5440,  1.0000]]
            #     [[ 0.1409, -0.4577,  0.8779,  0.0000],
            # [ 0.5525,  0.7722,  0.3138,  0.0000],
            # [-0.8215,  0.4408,  0.3617,  0.0000],
            # [ 0.1486, -1.2944,  3.3002,  1.0000]]
        scaling_modifer = 1.0
        pipe.convert_SHs_python = False 
        pipe.compute_cov3D_python = False




        # cam_R_right2 = np.array([[0.1409, -0.4577,  0.8779],
        #             [0.5525,  0.7722,  0.3138],
        #             [-0.8215,  0.4408,  0.3617]])
        # cam_T_right2 = np.array([0.1486, -1.2944,  3.3002]) 
        # cam_pos_right2 = myMiniCam2(978, 543, cam_R_right2, cam_T_right2, fovx, fovy, znear, zfar)
        # cam_R_new2 = (RI1.cpu().numpy()).T @ cam_R_right2
        # cam_T_new2 = cam_T_right2 - (TI1.cpu().numpy()) @ (RI1.cpu().numpy()).T @ cam_R_right2
        # cam_pos_new2 = myMiniCam2(978, 543, cam_R_new2, cam_T_new2, fovx, fovy, znear, zfar)
        # img_new2 = render(cam_pos_new2, gnew, pipe, background, scaling_modifer)["render"]
        # img_ref2 = render(cam_pos_right2, g1, pipe, background, scaling_modifer)["render"]


        # Ll12 = l1_loss(img_new2, img_ref2)
        # loss2 = (1.0 - opt.lambda_dssim) * Ll12 + opt.lambda_dssim * (1.0 - ssim(img_new2, img_ref2))

        cam_R_right = np.array([[0.5173, -0.1252,  0.8466],
                    [0.2749,  0.9611, -0.0258],
                    [-0.8105,  0.2461,  0.5316]])
        cam_T_right = np.array([1.5961, -0.4203,  3.5440])
        cam_pos_right = myMiniCam2(978, 543, cam_R_right, cam_T_right, fovx, fovy, znear, zfar)

        cam_R_new = (RI1.cpu().numpy()).T @ cam_R_right 
        cam_T_new = cam_T_right - (TI1.cpu().numpy()) @ (RI1.cpu().numpy()).T @ cam_R_right
        cam_pos_new = myMiniCam2(978, 543, cam_R_new, cam_T_new, fovx, fovy, znear, zfar)
        img_new = render(cam_pos_new, gnew, pipe, background, scaling_modifer)["render"]
        img_ref = render(cam_pos_right, g1, pipe, background, scaling_modifer)["render"]



        # img2 = img_ref2.permute(1,2,0).cpu().detach().numpy()
        # image = cv2.cvtColor(img2.astype('float32'), cv2.COLOR_RGB2BGR)
        # cv2.imshow('Image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



        # img_new = rgb_to_grayscale(img_new)
        # img_ref = rgb_to_grayscale(img_ref)
        Ll1 = l1_loss(img_new, img_ref)
        loss1 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(img_new, img_ref))


        loss = loss1
        # loss = myloss(img_new, img_ref)
        loss.backward()        # Compute gradients
        iter_end.record()

        optimizer.step()       # Update the parameter
        if iter % 20 == 0:  # Update learning rate every 20 iterations
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
            print(f'Iteration {iter}: Loss = {loss.item()}, Learning Rate = {current_lr}')




        update_scale = scale.item()
        update_offset = offset.tolist()
        print(f'Iteration {iter}: Loss = {loss.item()}, Updated scale = {update_offset}')


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


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.g1, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.g1.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.g1.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()






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
    RI1tmp = np.array([[0.1714, -0.0919,  0.9809],
                 [0.3924,  0.9196,  0.0176],
                 [-0.9037,  0.3819,  0.1936]])
    RI1 = torch.tensor(RI1tmp, dtype=torch.float32, device="cuda")
    TI1tmp = np.array([0.2364, -0.1797,  4.4662]) 
    TI1 = torch.tensor(TI1tmp, dtype=torch.float32, device="cuda")



    # -0.7488,  0.0608,  0.6600,  0.0000],
    #     [ 0.4406,  0.7896,  0.4272, -0.0000],
    #     [-0.4951,  0.6107, -0.6180, -0.0000],
    #     [-0.7020, -0.8442,  4.5460,  1.0000]]
    RI2tmp = np.array([[-0.7488,  0.0608,  0.6600],
                 [ 0.4406,  0.7896,  0.4272],
                 [-0.4951,  0.6107, -0.6180]])
    RI2 = torch.tensor(RI2tmp, dtype=torch.float32, device="cuda")
    TI2tmp = np.array([-0.7020, -0.8442,  4.5460])
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
    R2tmp = np.array([[0.4986, -0.2385,  0.8334],
                    [0.2861,  0.9528,  0.1016],
                    [ -0.8183,  0.1878,  0.5433]])
    T2tmp = np.array([-0.1995, 0.0221, 0.0976])

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