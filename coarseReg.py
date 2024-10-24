import argparse
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


def extract_RT(m):
    if m.shape[0] < 2:
        raise ValueError("The input tensor must contain at least two matrices.")

    matrix = m[1]
    R = matrix[:3, :3].numpy()
    T = matrix[:3, 3].numpy()

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

def coarseReg(image_paths, weights, model_name, device, image_size):
    # Load the model based on either direct weights or model name
    if weights:
        model = AsymmetricCroCo3DStereo.from_pretrained(weights).to(device)
    elif model_name:
        model = AsymmetricCroCo3DStereo.from_pretrained(f"naver/{model_name}").to(device)
    else:
        raise ValueError("Either --weights or --model_name must be provided.")
    
    # Load images
    imgs = load_images(image_paths, size=image_size)
    pairs = make_pairs(imgs, scene_graph='complete')

    # Run inference
    output = inference(pairs, model, device, batch_size=1)

    # Run global alignment
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)
    cams2world = scene.get_im_poses().cpu()

    return extract_RT(cams2world.numpy())
