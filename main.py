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

import torch
from gaussian_renderer import network_gui
import sys
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

from findpair import findpair
from coarseReg import coarseReg
from finetuning import finetuning
from gsMerge import gsMerge
from pytorch3d.transforms import quaternion_to_matrix

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
    parser.add_argument("--start_checkpoint2", type=str, default=None)
    parser.add_argument("--campose1", type = str, default = None)
    parser.add_argument("--campose2", type = str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    #pick choose two cam poses, and store respective images as "img1.png", "img2.png"
    cam1R, cam1T, cam2R, cam2T = findpair(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint, args.start_checkpoint2, args.campose1, args.campose2)
    iniR, iniT = coarseReg("img1.png", "img2.png")
    cam1Rt = torch.tensor(cam1R, dtype=torch.float32, device="cuda")
    cam1Tt = torch.tensor(cam1T, dtype=torch.float32, device="cuda")
    cam2Rt = torch.tensor(cam2R, dtype=torch.float32, device="cuda")
    cam2Tt = torch.tensor(cam2T, dtype=torch.float32, device="cuda")
    iniRt = torch.tensor(iniR, dtype=torch.float32, device="cuda")
    iniTt = torch.tensor(iniT, dtype=torch.float32, device="cuda")
    idRt = torch.eye(3).device("cuda")
    idTt = torch.tensor([0, 0, 0]).device("cuda")
    best_loss, offset, rot, Sscale, Tscale = finetuning(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, 
                                                        cam1Rt, cam1Tt, cam2Rt, cam2Tt, idRt, idTt, iniRt, iniTt, args.campose1, args.campose2)
    finetunedR = quaternion_to_matrix(rot)
    #final viz of fused gs
    gsMerge(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint2, args.debug_from, 
            cam1Rt, cam1Tt, cam2Rt, cam2Tt, idRt, idTt, finetunedR, Tscale * iniTt + offset, Sscale)