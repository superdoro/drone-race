"""Genesis implementation of SimBackend. This is the only module that imports genesis."""

import torch
import genesis as gs

from backends.base import DroneState, SimBackend
from config.drone_params import DroneParams
from utils.math_utils import quat_rotate_inverse

_HIDDEN_Z = -100.0  # inactive obstacles / markers are parked far below the ground plane


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
        self.rendered_env = rendered_env
        self.camera = None

    def build(self, num_envs: int, obstacle_radii: list[float], obstacle_height: float) -> None:
        self.num_envs = num_envs
        self.obstacle_height = obstacle_height
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=1),
            rigid_options=gs.options.RigidOptions(enable_collision=False, enable_joint_limit=False),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-3.0, -3.0, 4.0), camera_lookat=(3.0, 3.0, 1.0), camera_fov=50
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[self.rendered_env]),
            show_viewer=self.show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.drone_entity = self.scene.add_entity(gs.morphs.Drone(file=self.drone.urdf, pos=(0.0, 0.0, 1.0)))
        self.obstacles = [
            self.scene.add_entity(
                gs.morphs.Cylinder(
                    radius=r,
                    height=obstacle_height,
                    pos=(0.0, 0.0, _HIDDEN_Z),
                    fixed=True,
                    collision=False,
                    batch_fixed_verts=True,
                ),
                surface=gs.surfaces.Default(color=(0.85, 0.35, 0.15)),
            )
            for r in obstacle_radii
        ]
        self.goal_marker = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.15, pos=(0.0, 0.0, _HIDDEN_Z), fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2)),
        )
        if self.render_camera:
            self.camera = self.scene.add_camera(res=(960, 540), fov=60, GUI=False)
        self.scene.build(n_envs=num_envs)

        self._kf = self.drone.kf
        self._parked = torch.tensor([0.0, 0.0, _HIDDEN_Z], device=self.device)

    def reset_drones(self, env_ids, pos, quat):
        self.drone_entity.set_pos(pos, envs_idx=env_ids, zero_velocity=True)
        self.drone_entity.set_quat(quat, envs_idx=env_ids, zero_velocity=True)

    def set_obstacles(self, env_ids, xy, active):
        z = torch.where(active, self.obstacle_height / 2, _HIDDEN_Z)
        pos = torch.cat([xy, z[..., None]], dim=-1)
        for k, entity in enumerate(self.obstacles):
            entity.set_pos(pos[:, k], envs_idx=env_ids)

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
        self.goal_marker.set_pos(pos, envs_idx=env_ids)

    def follow_camera(self, env_idx, target_pos):
        if self.camera is not None:
            target = target_pos.detach().cpu().numpy()
            # Pass `up` every time: Genesis otherwise reuses the previous frame's tilted up vector and the view rolls.
            self.camera.set_pose(pos=target + [-1.0, -1.0, 2.5], lookat=target, up=(0.0, 0.0, 1.0))

    def start_recording(self, path, fps):
        if self.camera is None:
            raise RuntimeError("Build the backend with render_camera=True to record videos.")
        self.camera.start_recording(save_to_filename=path, fps=fps)

    def stop_recording(self):
        if self.camera is not None:
            self.camera.stop_recording()
