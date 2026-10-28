"""Batched quaternion helpers in pure torch.

Convention (same as Genesis and Isaac Lab): quaternions are (w, x, y, z), Hamilton product, rotating body -> world.
"""

import torch


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors `v` from the body frame to the world frame."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors `v` from the world frame to the body frame."""
    return quat_rotate(quat_conjugate(q), v)


def quat_from_euler(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Intrinsic Z-Y-X (yaw, pitch, roll) Euler angles to quaternion."""
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dim=-1,
    )
