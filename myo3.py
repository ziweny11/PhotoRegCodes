#adapted from e3nn package, for solving clamp(-1.1) bug

import torch
import math

def matrix_to_angles(R):
    r"""conversion from matrix to angles

    Parameters
    ----------
    R : `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    assert torch.allclose(torch.det(R), R.new_tensor(1))
    x = R @ R.new_tensor([0.0, 1.0, 0.0])
    a, b = xyz_to_angles(x)
    R = angles_to_matrix(a, b, torch.zeros_like(a)).transpose(-1, -2) @ R
    c = torch.atan2(R[..., 0, 2], R[..., 0, 0])
    return a, b, c



def xyz_to_angles(xyz):
    r"""convert a point :math:`\vec r = (x, y, z)` on the sphere into angles :math:`(\alpha, \beta)`

    .. math::

        \vec r = R(\alpha, \beta, 0) \vec e_z


    Parameters
    ----------
    xyz : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    magnitudes = torch.linalg.norm(xyz, dim=-1)
    epsilon = 1e-8
    is_near_zero_vector = magnitudes < epsilon

    if is_near_zero_vector.any():
        print("Near-zero vectors found before normalization:", xyz[is_near_zero_vector])
    xyz = torch.nn.functional.normalize(xyz, p=2, dim=-1)
    if not torch.isfinite(xyz).all():
        print("Invalid xyz values:", xyz)
    xyz = xyz.clamp(-0.999999, 0.999999)
    # xyz = xyz.clamp(-1, 1)
    assert (xyz >= -1).all() and (xyz <= 1).all(), "xyz out of range after clamping"
    beta = torch.acos(xyz[..., 1])
    alpha = torch.atan2(xyz[..., 0], xyz[..., 2])
    return alpha, beta


def angles_to_matrix(alpha, beta, gamma) -> torch.Tensor:
    r"""conversion from angles to matrix

    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    return matrix_y(alpha) @ matrix_x(beta) @ matrix_y(gamma)


def matrix_y(angle: torch.Tensor) -> torch.Tensor:
    r"""matrix of rotation around Y axis

    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack(
        [
            torch.stack([c, z, s], dim=-1),
            torch.stack([z, o, z], dim=-1),
            torch.stack([-s, z, c], dim=-1),
        ],
        dim=-2,
    )
    
    
def matrix_x(angle: torch.Tensor) -> torch.Tensor:
    r"""matrix of rotation around X axis

    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack(
        [
            torch.stack([o, z, z], dim=-1),
            torch.stack([z, c, -s], dim=-1),
            torch.stack([z, s, c], dim=-1),
        ],
        dim=-2,
    )
