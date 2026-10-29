"""Genesis implementation of SimBackend. This is the only module that imports genesis."""

import numpy as np
import torch
import genesis as gs

from backends.base import DroneState, SimBackend
from config.drone_params import DroneParams
from utils.math_utils import quat_rotate_inverse

_HIDDEN_Z = -100.0  # obstacles and walls rest far below the ground plane until they are laid out
_TRAIL_LENGTH = 90  # breadcrumbs behind the rendered drone, one every _TRAIL_EVERY camera updates
_TRAIL_EVERY = 2


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
        self.view_camera = None

    def build(self, num_envs, obstacle_radii, num_walls, wall_thickness) -> None:
        self.num_envs = num_envs
        self.wall_thickness = wall_thickness
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=1),
            rigid_options=gs.options.RigidOptions(enable_collision=False, enable_joint_limit=False),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-3.0, -3.0, 10.0), camera_lookat=(3.0, 3.0, 0.0), camera_fov=60
            ),
            # No shadows: untextured walls lit by one sun cast large slanted shadows on each other, which made
            # close-up (first-person) views unreadable. Brighter ambient light and a sky color keep faces distinct.
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[self.rendered_env], shadow=False, ambient_light=(0.35, 0.35, 0.35),
                background_color=(0.62, 0.72, 0.82),
            ),
            show_viewer=self.show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.drone_entity = self.scene.add_entity(gs.morphs.Drone(file=self.drone.urdf, pos=(0.0, 0.0, 1.0)))
        # Unit shapes whose visual vertices are rewritten per env, so each gets its own size (Genesis cannot resize a
        # built entity otherwise).
        self.obstacles, self.walls, self.goal_marker, self.drone_marker, self.trail = [], [], None, None, []
        if self.visual:
            self.obstacles = [self._shape(gs.morphs.Cylinder, (0.85, 0.35, 0.15), radius=1.0, height=1.0)
                              for _ in obstacle_radii]  # fmt: skip
            self.walls = [self._shape(gs.morphs.Box, (0.6, 0.62, 0.68), size=(1.0, 1.0, 1.0)) for _ in range(num_walls)]
            self.goal_marker = self.scene.add_entity(
                gs.morphs.Sphere(radius=0.15, pos=(0.0, 0.0, _HIDDEN_Z), fixed=True, collision=False, batch_fixed_verts=True),
                surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2)),
            )
            # The Crazyflie mesh is a few pixels wide on screen: mark its collision sphere and a short trail.
            self.drone_marker = self._marker(self.drone.body_radius, (1.0, 0.82, 0.1))
            self.trail = [self._marker(0.025, (0.35, 0.8, 1.0)) for _ in range(_TRAIL_LENGTH)]
        if self.render_camera:
            # Far away with a narrow field of view: close to a top-down orthographic view, so tall walls do not lean
            # over the rooms the way they do from a camera a few meters above them.
            self.camera = self.scene.add_camera(res=(1280, 720), fov=9, near=30.0, far=60.0, GUI=False)
            self.view_camera = self.scene.add_camera(res=(1280, 720), fov=75, near=0.05, far=40.0, GUI=False)
        self.scene.build(n_envs=num_envs)

        self._kf = self.drone.kf
        self._rendered_idx = torch.tensor([self.rendered_env], device=self.device)
        self._camera_updates = 0
        if self.visual:
            self._radii = torch.tensor(obstacle_radii, device=self.device)
            parked = torch.tensor([0.0, 0.0, _HIDDEN_Z], device=self.device)
            self._unit_cylinder = self.obstacles[0].get_vverts(envs_idx=self._rendered_idx)[0] - parked
            self._unit_box = self.walls[0].get_vverts(envs_idx=self._rendered_idx)[0] - parked

    def _marker(self, radius, color):
        morph = gs.morphs.Sphere(radius=radius, pos=(0.0, 0.0, _HIDDEN_Z), fixed=True, collision=False, batch_fixed_verts=True)
        return self.scene.add_entity(morph, surface=gs.surfaces.Default(color=color))

    def _shape(self, morph_cls, color, **kwargs):
        morph = morph_cls(pos=(0.0, 0.0, _HIDDEN_Z), fixed=True, collision=False, enable_custom_vverts=True, **kwargs)
        return self.scene.add_entity(morph, surface=gs.surfaces.Default(color=color))

    def _rendered_row(self, env_ids: torch.Tensor) -> int | None:
        """Row of the rendered env inside `env_ids`, or None when it is not being reset."""
        if not self.visual:
            return None
        hit = (env_ids == self.rendered_env).nonzero()
        return int(hit[0, 0]) if hit.numel() > 0 else None

    def _draw(self, entities, unit_verts, size, center, yaw, active) -> None:
        """Scale the unit shape to `size` (K, 3), rotate it by `yaw` (K,) about z and move it to `center` (K, 3)."""
        cos, sin = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        v = unit_verts[None] * size[:, None]
        v = torch.stack([cos * v[..., 0] - sin * v[..., 1], sin * v[..., 0] + cos * v[..., 1], v[..., 2]], dim=-1)
        v = v + torch.where(active[:, None], center, torch.tensor([0.0, 0.0, _HIDDEN_Z], device=self.device))[:, None]
        for entity, verts in zip(entities, v):
            entity.set_vverts(verts[None], envs_idx=self._rendered_idx)

    def reset_drones(self, env_ids, pos, quat):
        self.drone_entity.set_pos(pos, envs_idx=env_ids, zero_velocity=True)
        self.drone_entity.set_quat(quat, envs_idx=env_ids, zero_velocity=True)
        if self._rendered_row(env_ids) is not None:  # new episode: clear the trail
            hidden = torch.tensor([[0.0, 0.0, _HIDDEN_Z]], device=self.device)
            for crumb in self.trail:
                crumb.set_pos(hidden, envs_idx=self._rendered_idx)

    def set_obstacles(self, env_ids, xy, z_range, active):
        row = self._rendered_row(env_ids)
        if row is None:
            return
        z_lo, z_hi = z_range[row, :, 0], z_range[row, :, 1]
        size = torch.stack([self._radii, self._radii, z_hi - z_lo], dim=-1)
        center = torch.cat([xy[row], (0.5 * (z_lo + z_hi))[:, None]], dim=-1)
        self._draw(self.obstacles, self._unit_cylinder, size, center, torch.zeros_like(z_lo), active[row])

    def set_walls(self, env_ids, segments, z_range, active):
        row = self._rendered_row(env_ids)
        if row is None:
            return
        a, b = segments[row, :, :2], segments[row, :, 2:]
        z_lo, z_hi = z_range[row, :, 0], z_range[row, :, 1]
        t = torch.full_like(z_lo, self.wall_thickness)
        # extend by half the thickness at both ends so that corners are closed
        size = torch.stack([(b - a).norm(dim=-1) + t, t, z_hi - z_lo], dim=-1)
        center = torch.cat([0.5 * (a + b), (0.5 * (z_lo + z_hi))[:, None]], dim=-1)
        yaw = torch.atan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0])
        self._draw(self.walls, self._unit_box, size, center, yaw, active[row])

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
            self.goal_marker.set_pos(pos[row][None], envs_idx=self._rendered_idx)

    def follow_camera(self, env_idx, target_pos):
        if self.drone_marker is not None:
            pos = target_pos.detach()[None]
            self.drone_marker.set_pos(pos, envs_idx=self._rendered_idx)
            if self._camera_updates % _TRAIL_EVERY == 0:
                self.trail[(self._camera_updates // _TRAIL_EVERY) % _TRAIL_LENGTH].set_pos(pos, envs_idx=self._rendered_idx)
            self._camera_updates += 1
        if self.camera is not None:
            target = target_pos.detach().cpu().numpy()
            # Pass `up` every time: Genesis otherwise reuses the previous frame's tilted up vector and the view rolls.
            self.camera.set_pose(pos=target + [0.0, -3.0, 45.0], lookat=target, up=(0.0, 0.0, 1.0))

    def render_view(self, eye, lookat):
        if self.view_camera is None:
            raise RuntimeError("Build the backend with render_camera=True to render views.")
        self.view_camera.set_pose(pos=eye.detach().cpu().numpy(), lookat=lookat.detach().cpu().numpy(), up=(0.0, 0.0, 1.0))
        return np.asarray(self.view_camera.render()[0])

    def start_recording(self, path, fps):
        if self.camera is None:
            raise RuntimeError("Build the backend with render_camera=True to record videos.")
        self.camera.start_recording(save_to_filename=path, fps=fps)

    def stop_recording(self):
        if self.camera is not None:
            self.camera.stop_recording()
