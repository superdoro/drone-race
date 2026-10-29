"""Simulator-agnostic backend interface.

The task layer (task/, control/) only talks to this interface with batched torch tensors of shape (num_envs, ...).
Porting to another simulator (e.g. Isaac Lab) means writing one new subclass; nothing above this layer changes.

Conventions every backend must follow:
  * world frame is z-up, quaternions are (w, x, y, z) rotating body -> world
  * `lin_vel` is expressed in the world frame, `ang_vel` in the body frame
  * environments share local coordinates (each env is its own world starting at the origin)
  * rotor thrusts act along body +z at the rotor positions in `DroneParams.rotor_xy`, and each rotor adds a yaw
    torque of `rotor_spin * km / kf * thrust` about body +z
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from config.drone_params import DroneParams


@dataclass
class DroneState:
    pos: torch.Tensor  # (N, 3) world
    quat: torch.Tensor  # (N, 4) wxyz, body -> world
    lin_vel: torch.Tensor  # (N, 3) world
    ang_vel: torch.Tensor  # (N, 3) body


class SimBackend(ABC):
    num_envs: int
    device: torch.device
    dt: float  # physics step (s)

    def __init__(self, drone: DroneParams, dt: float):
        self.drone = drone
        self.dt = dt

    @abstractmethod
    def build(
        self,
        num_envs: int,
        obstacle_radii: list[float],
        obstacle_height: float,
        wall_thickness: float,
        wall_height: float,
    ) -> None:
        """Create the scene: ground, one drone per env, one vertical cylinder per obstacle slot, and walls.

        Collisions and the state observations are computed analytically by the task, so obstacles and walls only need
        to exist where something looks at them (viewer, video and, later, depth cameras).
        """

    @abstractmethod
    def reset_drones(self, env_ids: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor) -> None:
        """Teleport the drones of `env_ids` to `pos` / `quat` with zero velocity."""

    @abstractmethod
    def set_obstacles(self, env_ids: torch.Tensor, xy: torch.Tensor, active: torch.Tensor) -> None:
        """Place obstacle slots of `env_ids` at `xy` (n, K, 2); inactive slots (n, K) must be hidden from sensors."""

    @abstractmethod
    def set_walls(self, env_ids: torch.Tensor, segments: torch.Tensor, active: torch.Tensor) -> None:
        """Lay out the walls of `env_ids`: center lines (n, S, 4) as x0, y0, x1, y1, standing on the ground, with the
        thickness given at build time; inactive slots (n, S) must be hidden. Wall lengths differ per env and episode."""

    @abstractmethod
    def apply_rotor_thrusts(self, thrusts: torch.Tensor) -> None:
        """Command per-rotor thrusts (N, num_rotors) in Newtons for the next physics step."""

    @abstractmethod
    def step(self) -> None:
        """Advance the physics by `dt`."""

    @abstractmethod
    def get_drone_state(self) -> DroneState:
        """Return the current drone state of every env."""

    def get_depth(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Depth images (n, H, W) from the drone-mounted camera; needed only for obs_mode='depth'."""
        raise NotImplementedError("Depth camera is not implemented for this backend yet.")

    # ---- optional visualization helpers (no-ops unless a backend supports them) ----

    def set_goal_markers(self, env_ids: torch.Tensor, pos: torch.Tensor) -> None:
        pass

    def follow_camera(self, env_idx: int, target_pos: torch.Tensor) -> None:
        pass

    def start_recording(self, path: str, fps: int) -> None:
        raise NotImplementedError("Video recording is not supported by this backend.")

    def stop_recording(self) -> None:
        pass


def make_backend(name: str, drone: DroneParams, dt: float, **kwargs) -> SimBackend:
    """Import backends lazily so that only the selected simulator needs to be installed."""
    if name == "genesis":
        from backends.genesis_backend import GenesisBackend

        return GenesisBackend(drone, dt, **kwargs)
    raise ValueError(f"Unknown backend '{name}'. Available: genesis")
