import torch
from pytorch3d.transforms import matrix_to_quaternion
from e3nn import o3
import einops
from einops import einsum
import numpy as np
from utils.general_utils import build_rotation

def transform_shs(shs_feat, rotation_matrix):
    # print("old", shs_feat)
    new_shs_feat = shs_feat.clone()
    # new_shs_feat = new_shs_feat.transpose(1, 2)

    ## rotate shs
    # switch axes: yzx -> xyz
    P = torch.tensor(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), dtype = torch.float32, device="cuda")
    permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)

    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2])

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
    # print("new", new_shs_feat)
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
    # print("this is xyz new", xyz_new)
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


def rotate_by_matrix(G, rotation_matrix, keep_sh_degree: bool = True):
    G._xyz = torch.mm(G._xyz, rotation_matrix.t())

    gaussian_rotation_matrix = build_rotation(G._rotation)
    # print(gaussian_rotation_matrix)
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



#rubbish
def pos_gen(g):
    coor = g._xyz
    # com = coor.mean(dim = 0)
    com = maxdens(g)
    min_vals, _ = torch.min(coor, dim=0)
    max_vals, _ = torch.max(coor, dim=0)
    mid = (min_vals + max_vals) / 2
    rot = vec2rot(com.cpu().detach().numpy(), mid.cpu().detach().numpy())
    print(rot, mid.cpu().detach().numpy())
    return rot, mid.cpu().detach().numpy()/7


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