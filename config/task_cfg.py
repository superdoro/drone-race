"""Task and PPO configuration. Units: meters, seconds, radians unless stated otherwise."""


def get_cfgs() -> tuple[dict, dict]:
    env_cfg = {
        "drone": "cf2x",
        "sim_dt": 0.01,  # physics + inner rate loop at 100 Hz
        "decimation": 2,  # policy at 50 Hz
        "episode_length_s": 60.0,  # hard cap; each episode's own limit comes from `time_budget`
        # Minimum speed requirement: an episode times out after base_s + route length / min_avg_speed.
        "time_budget": {"base_s": 5.0, "min_avg_speed": 1.0},
        "arena": {
            "spawn_z": (1.0, 2.0),  # start and goal altitude range
            "ground_z": 0.1,  # below this counts as a crash into the ground
            "ceiling_z": 3.5,  # below the wall and pillar tops, so they must be passed around, not over
        },
        # A random chain of rectangular rooms, joined by one door each (see task/rooms.py).
        "rooms": {
            "size_range": (4.0, 8.0),  # room side length
            "grid": 0.5,  # room sizes and offsets are multiples of this, so no wall piece is a sliver
            "wall_thickness": 0.1,
            "wall_height": 4.0,
            "door_margin": 0.5,  # min wall length between a door / partition gap and a corner
            "start_inset": 0.8,  # start distance from its wall and from the corners
            "goal_inset": 0.6,
            "partition_clearance": 1.0,  # min distance from a partition to the room's entry and exit
            "partition_spacing": 1.2,  # min distance between parallel partitions
            "zigzag_prob": 0.7,  # chance that the next partition's gap is on the other half of the room
            "free_wall_length": (0.8, 2.0),
            "pass_gap": 0.5,  # min free gap between any two obstacles / walls, keeps every layout passable
            "port_clearance": 1.0,  # no pillar / free wall within this distance of a door, gap, start or goal
            "on_path_fraction": 0.8,  # share of obstacles sampled near the straight route (the rest anywhere)
            "on_path_sigma": 0.4,  # lateral spread of those
            "place_tries": 30,
            "pool_size": 8192,  # pre-generated layouts on the GPU
            "pool_reuse": 8,  # regenerate the pool after this many draws per layout on average
        },
        "goal": {"reach_radius": 0.4},
        "init": {"max_tilt_deg": 10.0},
        "obstacles": {"radius_range": (0.1, 0.3), "height": 4.0},
        "control": {
            "max_body_rates": (8.0, 8.0, 3.0),  # roll, pitch, yaw rate at |action| = 1
            "rate_gain": (20.0, 20.0, 10.0),  # inner P loop bandwidth (1/s)
        },
        "obs": {
            "mode": "state",  # "state": obstacle positions are known; "depth": reserved for the camera stage
            "num_sensed_pillars": 8,
            "num_sensed_walls": 12,
            "sense_range": 6.0,
            "goal_dist_clip": 10.0,
            "lin_vel_scale": 5.0,
            "ang_vel_scale": 10.0,
            "route_len_scale": 50.0,
        },
        "rewards": {
            "progress": 5.0,
            "time": 0.01,
            "speed_limit": 0.2,
            "max_speed": 3.0,  # soft limit; with progress 5 / m the best open-space speed is ~3.3 m/s
            "ang_vel": 0.001,
            "yaw_rate": 0.01,
            "action_rate": 0.01,
            "action_bound": 1.0,
            "proximity": 0.5,
            "proximity_margin": 0.3,
            # flying straight at a pillar from 1 m away at any speed costs ~0.5 * 25 = 12.5 in total before the crash
            "approach": 0.5,
            "approach_radius": 1.0,
            "crash": 40.0,
            "goal": 10.0,
        },
        "curriculum": {
            # Counts are (min, max) per layout ("rooms") or per room (the rest). Each level widens the ranges instead
            # of shifting them, so easy layouts stay in the mix.
            "levels": [
                {"rooms": (1, 2), "partitions": (0, 0), "free_walls": (0, 0), "pillars": (0, 1), "door_width": (1.2, 1.6)},
                {"rooms": (1, 3), "partitions": (0, 1), "free_walls": (0, 1), "pillars": (0, 2), "door_width": (1.0, 1.6)},
                {"rooms": (1, 4), "partitions": (0, 1), "free_walls": (0, 1), "pillars": (0, 3), "door_width": (1.0, 1.6)},
                {"rooms": (1, 5), "partitions": (0, 2), "free_walls": (0, 2), "pillars": (0, 4), "door_width": (0.9, 1.6)},
                {"rooms": (1, 6), "partitions": (0, 2), "free_walls": (0, 2), "pillars": (0, 5), "door_width": (0.8, 1.6)},
            ],
            "initial_level": 0,
            "promote_success_rate": 0.8,
            "window": 4096,  # finished episodes per promotion check
        },
        # Reserved for sim-to-real; not implemented yet.
        "domain_rand": {"enabled": False, "mass_scale": (0.9, 1.1), "thrust_scale": (0.9, 1.1), "latency_steps": (0, 2)},
    }

    train_cfg = {
        "num_steps_per_env": 24,
        "save_interval": 100,
        # the critic also sees privileged route information (the next partition gap, remaining route length)
        "obs_groups": {"actor": ["policy"], "critic": ["policy", "privileged"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
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
            "hidden_dims": [512, 256, 128],
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
