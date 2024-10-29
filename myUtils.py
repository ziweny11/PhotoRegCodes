import torch
from pytorch3d.transforms import matrix_to_quaternion
from e3nn import o3
import einops
from einops import einsum
import numpy as np
from utils.general_utils import build_rotation


def transform_shs(shs_feat, rotation_matrix):
    # Ensure all inputs are on the same device, preferably at the start of your function
    device = rotation_matrix.device  # Use the device from rotation_matrix
    rotation_matrix = rotation_matrix.to(device)
    new_shs_feat = shs_feat.clone().to(device)

    # Permutation matrix directly on the correct device
    P = torch.tensor([[0, 0, 1], 
                      [1, 0, 0], 
                      [0, 1, 0]], dtype=torch.float32, device=device)

    permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)

    a = rot_angles[0].to(device)
    b = rot_angles[1].to(device)
    c = rot_angles[2].to(device)

    # Call Wigner D functions
    D_1 = wigner_D(1, a, -b, c)
    D_2 = wigner_D(2, a, -b, c)
    D_3 = wigner_D(3, a, -b, c)
    # rotation of the shs features
    one_degree_shs = new_shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
        D_1,
        one_degree_shs,
        "... i j, ... j -> ... i",
    )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    new_shs_feat[:, 0:3] = one_degree_shs

    if new_shs_feat.shape[1] >= 4:
        two_degree_shs = new_shs_feat[:, 3:8]
        two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
        two_degree_shs = einsum(
            D_2,
            two_degree_shs,
            "... i j, ... j -> ... i",
        )
        two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
        new_shs_feat[:, 3:8] = two_degree_shs

        if new_shs_feat.shape[1] >= 9:
            three_degree_shs = new_shs_feat[:, 8:15]
            three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
            three_degree_shs = einsum(
                D_3,
                three_degree_shs,
                "... i j, ... j -> ... i",
            )
            three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
            new_shs_feat[:, 8:15] = three_degree_shs
    return new_shs_feat

def gaus_translate(G, t):
    G._xyz = G._xyz + t

def gaus_append(g1, g2, gnew):
    (active_sh_degree,
    xyz,
    features_dc,
    features_rest,
    scaling,
    rotation,
    opacity,
    max_radii2D,
    xyz_gradient_accum,
    denom, opt_dict,
    spatial_lr_scale) = g1.capture()
    (active_sh_degree2,
    xyz2,
    features_dc2,
    features_rest2,
    scaling2,
    rotation2,
    opacity2,
    max_radii2D2,
    xyz_gradient_accum2,
    denom2, opt_dict2,
    spatial_lr_scale2) = g2.capture()
    xyz_new = torch.cat((xyz, xyz2), dim=0)
    features_dc_new = torch.cat((features_dc, features_dc2), dim=0)
    features_rest_new = torch.cat((features_rest, features_rest2), dim=0)
    scaling_new = torch.cat((scaling, scaling2), dim=0)
    rotation_new = torch.cat((rotation, rotation2), dim=0)
    opacity_new = torch.cat((opacity, opacity2), dim=0)
    gnew.active_sh_degree = max(active_sh_degree, active_sh_degree2)
    gnew._xyz = xyz_new
    gnew._features_dc = features_dc_new
    gnew._features_rest = features_rest_new
    gnew._scaling = scaling_new
    gnew._rotation = rotation_new
    gnew._opacity = opacity_new

def gaus_transform(G, R, t):
    rotate_by_matrix(G, R)
    gaus_translate(G, t)

def rotate_by_matrix(G, rotation_matrix, keep_sh_degree: bool = True):
    G._xyz = torch.mm(G._xyz, rotation_matrix.t())

    gaussian_rotation_matrix = build_rotation(G._rotation)
    gaussian_rotation_matrix = rotation_matrix @ gaussian_rotation_matrix
    new_quaternions = matrix_to_quaternion(gaussian_rotation_matrix)
    G._rotation = new_quaternions
    if not keep_sh_degree:
        print("set sh_degree=0 when rotation transform enabled")
        G.sh_degrees = 0
    G._features_rest = transform_shs(G._features_rest, rotation_matrix)

def gaus_copy(g, gnew):
    gnew._xyz = g._xyz
    gnew._features_dc = g._features_dc
    gnew._features_rest = g._features_rest
    gnew._scaling = g._scaling
    gnew._rotation = g._rotation
    gnew._opacity = g._opacity

def rescale(g, scale):
    sc = scale.item()
    g._xyz = g._xyz * sc
    g._scaling = g._scaling + torch.log(scale)

def vec2rot(A, B):
    direction = A - B
    direction_norm = direction / np.linalg.norm(direction)
    up = np.array([0, 1, 0])
    right = np.cross(up, direction_norm)
    right_norm = right / np.linalg.norm(right)
    up_new = np.cross(direction_norm, right_norm)
    rotation_matrix = np.vstack([right_norm, up_new, direction_norm]).T
    return rotation_matrix


def maxdens(g):
    coor = g._xyz
    voxel_size = 10
    min_coords, _ = torch.min(coor, dim=0)
    max_coords, _ = torch.max(coor, dim=0)
    voxel_indices = ((coor - min_coords) / voxel_size).long()
    unique_voxels, counts = torch.unique(voxel_indices, return_counts=True, dim=0)
    max_count_index = torch.argmax(counts)
    densest_voxel = unique_voxels[max_count_index]
    densest_point = densest_voxel.float() * voxel_size + min_coords + (voxel_size / 2)
    return densest_point



def measure_blurriness(img):
    img_np = img.detach().cpu().numpy()
    grayscale_image_np = np.mean(img_np, axis=0, keepdims=True)
    laplacian_kernel_np = np.array([[-1, -1, -1],
                                    [-1,  8, -1],
                                    [-1, -1, -1]], dtype=np.float32)
    laplacian_np = np.zeros_like(grayscale_image_np)
    for i in range(grayscale_image_np.shape[0]):
        laplacian_np[i] = np.convolve(grayscale_image_np[i].flatten(), laplacian_kernel_np.flatten(), mode='same').reshape(grayscale_image_np[i].shape)
    variance_np = np.var(laplacian_np)
    
    return variance_np



#adapted from o3 library
def wigner_D(l, alpha, beta, gamma):
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    alpha = alpha[..., None, None] % (2 * torch.pi)
    beta = beta[..., None, None] % (2 * torch.pi)
    gamma = gamma[..., None, None] % (2 * torch.pi)
    X = so3_generators(l).to('cuda')
    return torch.matrix_exp(alpha * X[1]) @ torch.matrix_exp(beta * X[0]) @ torch.matrix_exp(gamma * X[1])

def so3_generators(l) -> torch.Tensor:
    X = su2_generators(l)
    Q = change_basis_real_to_complex(l)
    X = torch.conj(Q.T) @ X @ Q
    assert torch.all(torch.abs(torch.imag(X)) < 1e-5)
    return torch.real(X)

def su2_generators(j) -> torch.Tensor:
    m = torch.arange(-j, j)
    raising = torch.diag(-torch.sqrt(j * (j + 1) - m * (m + 1)), diagonal=-1)

    m = torch.arange(-j + 1, j + 1)
    lowering = torch.diag(torch.sqrt(j * (j + 1) - m * (m - 1)), diagonal=1)

    m = torch.arange(-j, j + 1)
    return torch.stack([
        0.5 * (raising + lowering),  # x (usually)
        torch.diag(1j * m),  # z (usually)
        -0.5j * (raising - lowering),  # -y (usually)
    ], dim=0)


def change_basis_real_to_complex(l: int, dtype=None, device=None) -> torch.Tensor:
    # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    q = torch.zeros((2 * l + 1, 2 * l + 1), dtype=torch.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / 2**0.5
        q[l + m, l - abs(m)] = -1j / 2**0.5
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1)**m / 2**0.5
        q[l + m, l - abs(m)] = 1j * (-1)**m / 2**0.5
    q = (-1j)**l * q  # Added factor of 1j**l to make the Clebsch-Gordan coefficients real

    dtype, device = explicit_default_types(dtype, device)
    dtype = {
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
    }[dtype]
    # make sure we always get:
    # 1. a copy so mutation doesn't ruin the stored tensors
    # 2. a contiguous tensor, regardless of what transpositions happened above
    return q.to(dtype=dtype, device=device, copy=True, memory_format=torch.contiguous_format)


def explicit_default_types(dtype, device):
    """A torchscript-compatible type resolver"""
    if dtype is None:
        dtype = _torch_get_default_dtype()
    if device is None:
        device = torch_get_default_device()
    return dtype, device

def _torch_get_default_dtype() -> torch.dtype:
    """A torchscript-compatible version of torch.get_default_dtype()"""
    return torch.empty(0).dtype


def torch_get_default_device() -> torch.device:
    return torch.empty(0).device