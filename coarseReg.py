import argparse
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import os
import numpy as np

#return numpy R,T from trans matrix
def extract_RT(m):
    if m.shape[0] < 2 or m.shape[1] < 4 or m.shape[2] < 4:
        raise ValueError("invalid matrix obtained from dust3r")

    identity = np.eye(4)

    if not np.allclose(m[0], identity):
        inv_m0 = np.linalg.inv(m[0])
        m[1] = np.dot(inv_m0, m[1])
    R = m[1][:3, :3]
    T = m[1][:3, 3]

    return R, T

def get_args_parser():
    parser = argparse.ArgumentParser(description='DUSt3R Camera Alignment Tool')
    parser.add_argument('--image_paths', nargs=2, required=True, help='Paths to the two images for alignment')
    parser.add_argument('--weights', type=str, help='Path to the model weights')
    parser.add_argument('--model_name', type=str, help='Name of the model if using predefined weights',
                        choices=["DUSt3R_ViTLarge_BaseDecoder_512_dpt",
                                 "DUSt3R_ViTLarge_BaseDecoder_512_linear",
                                 "DUSt3R_ViTLarge_BaseDecoder_224_linear"])
    parser.add_argument('--device', type=str, default='cuda', help='PyTorch device')
    parser.add_argument('--image_size', type=int, default=512, help='Input image size')
    return parser

def coarseReg(image1_path, image2_path, device='cuda', image_size=512):
    model_path = os.path.join(os.path.dirname(image1_path), "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
    
    # Load images
    imgs = load_images([image1_path, image2_path], size=image_size)
    pairs = make_pairs(imgs, scene_graph='complete')

    # Run inference
    output = inference(pairs, model, device, batch_size=1)

    # Run global alignment
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)
    cams2world = scene.get_im_poses().cpu()

    return extract_RT(cams2world.numpy())