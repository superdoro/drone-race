"""Task and PPO configuration. Units: meters, seconds, radians unless stated otherwise."""


def get_cfgs() -> tuple[dict, dict]:
    env_cfg = {
        "drone": "cf2x",
        "sim_dt": 0.01,  # physics + inner rate loop at 100 Hz
        "decimation": 2,  # policy at 50 Hz
        "episode_length_s": 15.0,
        "arena": {
            "spawn_z": (1.0, 2.0),  # start and goal altitude range
            "ground_z": 0.1,  # below this counts as a crash into the ground
            "ceiling_z": 3.5,  # below the obstacle tops, so obstacles must be passed around, not over
            "horizontal_margin": 3.0,  # allowed detour beyond the start-goal circle before out-of-bounds
        },
        "goal": {"reach_radius": 0.3},
        "init": {"max_tilt_deg": 10.0},
        "obstacles": {
            "radius_range": (0.1, 0.3),
            "height": 4.0,
            "along_range": (0.15, 0.85),  # fraction of the start->goal segment where obstacles are scattered
            "lateral_spread": 1.5,  # max sideways offset from the segment
            "endpoint_clearance": 0.6,  # min gap between an obstacle surface and the start / goal
        },
        "control": {
            "max_body_rates": (8.0, 8.0, 3.0),  # roll, pitch, yaw rate at |action| = 1
            "rate_gain": (20.0, 20.0, 10.0),  # inner P loop bandwidth (1/s)
        },
        "obs": {
            "mode": "state",  # "state": obstacle positions are known; "depth": reserved for the camera stage
            "num_sensed_obstacles": 4,
            "sense_range": 4.0,
            "goal_dist_clip": 10.0,
            "lin_vel_scale": 5.0,
            "ang_vel_scale": 10.0,
        },
        "rewards": {
            "progress": 5.0,
            "time": 0.02,
            "ang_vel": 0.001,
            "yaw_rate": 0.01,
            "action_rate": 0.01,
            "action_bound": 1.0,
            "proximity": 0.5,
            "proximity_margin": 0.3,
            "crash": 20.0,
            "goal": 10.0,
        },
        "curriculum": {
            # Each level widens the (min, max) ranges instead of shifting them, so short / open episodes stay in the
            # mix: training only on far goals made the policy forget how to brake for near ones.
            "levels": [
                {"num_obstacles": (0, 0), "goal_dist": (2.0, 4.0)},
                {"num_obstacles": (0, 2), "goal_dist": (2.0, 6.0)},
                {"num_obstacles": (0, 4), "goal_dist": (2.0, 8.0)},
                {"num_obstacles": (0, 6), "goal_dist": (2.0, 10.0)},
                {"num_obstacles": (0, 8), "goal_dist": (2.0, 12.0)},
            ],
            "promote_success_rate": 0.8,
            "window": 4096,  # finished episodes per promotion check
        },
        # Reserved for sim-to-real; not implemented yet.
        "domain_rand": {"enabled": False, "mass_scale": (0.9, 1.1), "thrust_scale": (0.9, 1.1), "latency_steps": (0, 2)},
    }

    train_cfg = {
        "num_steps_per_env": 24,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
            # std capped at 1: beyond that the env clipping hides it and the entropy bonus inflates it without bound
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
                "std_range": (0.01, 1.0),
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.005,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "schedule": "adaptive",
            "desired_kl": 0.01,
        },
        "logger": "tensorboard",
    }
    return env_cfg, train_cfg
