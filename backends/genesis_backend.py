"""Genesis implementation of SimBackend. This is the only module that imports genesis."""

import math
import warnings

import torch
import genesis as gs

from backends.base import DroneState, SimBackend
from config.drone_params import DroneParams
from utils.math_utils import quat_rotate_inverse

_HIDDEN_Z = -100.0  # unused obstacles / wall tiles are parked far below the ground plane
# Genesis cannot resize a box after the scene is built, so a wall of any length is drawn with overlapping fixed-length
# tiles. The room generator never makes a wall piece shorter than 0.5 m.
_TILE_LENGTHS = (2.0, 0.5)
_TILE_COUNTS = (200, 160)


class GenesisBackend(SimBackend):
    def __init__(
        self,
        drone: DroneParams,
        dt: float,
        device: str = "cuda",
        show_viewer: bool = False,
        render_camera: bool = False,
        rendered_env: int = 0,
        seed: int | None = None,
    ):
        super().__init__(drone, dt)
        if not gs._initialized:
            gs.init(backend=gs.cpu if device == "cpu" else gs.gpu, logging_level="warning", seed=seed)
        self.device = gs.device
        self.show_viewer = show_viewer
        self.render_camera = render_camera
        # Nothing looks at the obstacles during training (collisions are analytic), so they are only created for the
        # viewer / video, and only the rendered env is kept up to date.
        self.visual = show_viewer or render_camera
        self.rendered_env = rendered_env
        self.camera = None

    def build(self, num_envs, obstacle_radii, obstacle_height, wall_thickness, wall_height) -> None:
        self.num_envs = num_envs
        self.obstacle_height = obstacle_height
        self.wall_thickness = wall_thickness
        self.wall_height = wall_height
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=1),
            rigid_options=gs.options.RigidOptions(enable_collision=False, enable_joint_limit=False),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-3.0, -3.0, 10.0), camera_lookat=(3.0, 3.0, 0.0), camera_fov=60
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[self.rendered_env]),
            show_viewer=self.show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.drone_entity = self.scene.add_entity(gs.morphs.Drone(file=self.drone.urdf, pos=(0.0, 0.0, 1.0)))
        self.obstacles, self.wall_tiles, self.goal_marker = [], {}, None
        if self.visual:
            self.obstacles = [self._fixed(gs.morphs.Cylinder, (0.85, 0.35, 0.15), radius=r, height=obstacle_height)
                              for r in obstacle_radii]  # fmt: skip
            self.wall_tiles = {
                length: [self._fixed(gs.morphs.Box, (0.6, 0.62, 0.68), size=(length, wall_thickness, wall_height))
                         for _ in range(count)]
                for length, count in zip(_TILE_LENGTHS, _TILE_COUNTS)
            }  # fmt: skip
            self.goal_marker = self._fixed(gs.morphs.Sphere, (0.1, 0.8, 0.2), radius=0.15)
        if self.render_camera:
            self.camera = self.scene.add_camera(res=(960, 540), fov=40, GUI=False)
        self.scene.build(n_envs=num_envs)

        self._kf = self.drone.kf
        self._rendered_idx = torch.tensor([self.rendered_env], device=self.device)
        self._warned_tiles = False

    def _fixed(self, morph_cls, color, **kwargs):
        """A visual-only fixed entity whose pose can differ per env."""
        morph = morph_cls(pos=(0.0, 0.0, _HIDDEN_Z), fixed=True, collision=False, batch_fixed_verts=True, **kwargs)
        return self.scene.add_entity(morph, surface=gs.surfaces.Default(color=color))

    def _rendered_row(self, env_ids: torch.Tensor) -> int | None:
        """Row of the rendered env inside `env_ids`, or None when it is not being reset."""
        if not self.visual:
            return None
        hit = (env_ids == self.rendered_env).nonzero()
        return int(hit[0, 0]) if hit.numel() > 0 else None

    def _place(self, entity, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        entity.set_pos(torch.tensor([[x, y, z]], device=self.device), envs_idx=self._rendered_idx)
        quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        entity.set_quat(torch.tensor([quat], device=self.device), envs_idx=self._rendered_idx)

    def reset_drones(self, env_ids, pos, quat):
        self.drone_entity.set_pos(pos, envs_idx=env_ids, zero_velocity=True)
        self.drone_entity.set_quat(quat, envs_idx=env_ids, zero_velocity=True)

    def set_obstacles(self, env_ids, xy, active):
        row = self._rendered_row(env_ids)
        if row is None:
            return
        for k, entity in enumerate(self.obstacles):
            z = self.obstacle_height / 2 if bool(active[row, k]) else _HIDDEN_Z
            self._place(entity, float(xy[row, k, 0]), float(xy[row, k, 1]), z)

    def set_walls(self, env_ids, segments, active):
        row = self._rendered_row(env_ids)
        if row is None:
            return
        t = self.wall_thickness
        placements = {length: [] for length in _TILE_LENGTHS}
        for x0, y0, x1, y1 in segments[row][active[row]].tolist():
            length = math.hypot(x1 - x0, y1 - y0)
            ux, uy = (x1 - x0) / length, (y1 - y0) / length
            length += t  # extend by half the thickness at both ends so that corners are closed
            tile = next((size for size in _TILE_LENGTHS if length >= size), _TILE_LENGTHS[-1])
            count = max(1, math.ceil(length / tile - 1e-6))
            for i in range(count):
                s = 0.5 * tile + (i * (length - tile) / (count - 1) if count > 1 else 0.5 * (length - tile))
                placements[tile].append((x0 + ux * (s - 0.5 * t), y0 + uy * (s - 0.5 * t), math.atan2(uy, ux)))
        for size, entities in self.wall_tiles.items():
            if len(placements[size]) > len(entities) and not self._warned_tiles:
                warnings.warn(f"Layout needs {len(placements[size])} wall tiles of {size} m, only {len(entities)} exist.")
                self._warned_tiles = True
            for j, entity in enumerate(entities):
                if j < len(placements[size]):
                    x, y, yaw = placements[size][j]
                    self._place(entity, x, y, self.wall_height / 2, yaw)
                else:
                    self._place(entity, 0.0, 0.0, _HIDDEN_Z)

    def apply_rotor_thrusts(self, thrusts):
        self.drone_entity.set_propellers_rpm(torch.sqrt(thrusts.clamp_min(0.0) / self._kf))

    def step(self):
        self.scene.step()

    def get_drone_state(self) -> DroneState:
        quat = self.drone_entity.get_quat()
        return DroneState(
            pos=self.drone_entity.get_pos(),
            quat=quat,
            lin_vel=self.drone_entity.get_vel(),
            ang_vel=quat_rotate_inverse(quat, self.drone_entity.get_ang()),  # Genesis reports it in world frame
        )

    # ---- visualization ----

    def set_goal_markers(self, env_ids, pos):
        row = self._rendered_row(env_ids)
        if row is not None:
            self._place(self.goal_marker, *pos[row].tolist())

    def follow_camera(self, env_idx, target_pos):
        if self.camera is not None:
            target = target_pos.detach().cpu().numpy()
            # High, narrow, slightly tilted view from above: little perspective lean on the 4 m walls, and no wall
            # blocks the drone. Pass `up` every time: Genesis otherwise reuses the previous frame's tilted up vector
            # and the view rolls.
            self.camera.set_pose(pos=target + [0.0, -2.0, 16.0], lookat=target, up=(0.0, 0.0, 1.0))

    def start_recording(self, path, fps):
        if self.camera is None:
            raise RuntimeError("Build the backend with render_camera=True to record videos.")
        self.camera.start_recording(save_to_filename=path, fps=fps)

    def stop_recording(self):
        if self.camera is not None:
            self.camera.stop_recording()
