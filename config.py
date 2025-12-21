"""
AutoNavX Configuration Settings
"""

import numpy as np

# =============================================================================
# CARLA Connection Settings
# =============================================================================
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

# =============================================================================
# Vehicle Settings
# =============================================================================
VEHICLE_BLUEPRINT = 'vehicle.lincoln.mkz_2020'
MAX_SPEED = 50.0  # km/h
MIN_SPEED = 5.0   # km/h

# =============================================================================
# Sensor Configuration
# =============================================================================
LIDAR_CONFIG = {
    'channels': 64,
    'range': 100.0,  # meters
    'points_per_second': 1200000,
    'rotation_frequency': 20.0,
    'upper_fov': 15.0,
    'lower_fov': -25.0,
    'horizontal_fov': 360.0,
    'sensor_tick': 0.05,
    'position': {'x': 0.0, 'y': 0.0, 'z': 2.4}
}

RADAR_CONFIG = {
    'horizontal_fov': 30.0,
    'vertical_fov': 30.0,
    'range': 100.0,
    'points_per_second': 1500,
    'sensor_tick': 0.05,
    'position': {'x': 2.0, 'y': 0.0, 'z': 1.0}
}

GNSS_CONFIG = {
    'noise_alt_stddev': 0.0,
    'noise_lat_stddev': 0.0,
    'noise_lon_stddev': 0.0,
    'sensor_tick': 0.1
}

IMU_CONFIG = {
    'noise_accel_stddev_x': 0.0,
    'noise_accel_stddev_y': 0.0,
    'noise_accel_stddev_z': 0.0,
    'noise_gyro_stddev_x': 0.0,
    'noise_gyro_stddev_y': 0.0,
    'noise_gyro_stddev_z': 0.0,
    'sensor_tick': 0.05
}

CAMERA_CONFIG = {
    'image_size_x': 800,
    'image_size_y': 600,
    'fov': 110,
    'sensor_tick': 0.05
}

# =============================================================================
# LiDAR Processing (MVF - Multi-View Fusion)
# =============================================================================
VOXEL_SIZE = [0.1, 0.1, 0.2]  # x, y, z in meters
POINT_CLOUD_RANGE = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
MAX_POINTS_PER_VOXEL = 32
MAX_VOXELS = 16000

# =============================================================================
# Radar Processing (EKF)
# =============================================================================
EKF_PROCESS_NOISE = np.diag([0.5, 0.5, 0.1, 0.1])  # [x, y, vx, vy]
EKF_MEASUREMENT_NOISE = np.diag([1.0, 1.0, 0.5, 0.5])
EKF_INITIAL_COVARIANCE = np.eye(4) * 10

# =============================================================================
# TD3 Reinforcement Learning Settings
# =============================================================================
TD3_CONFIG = {
    # Network architecture
    'state_dim': 64,
    'action_dim': 2,  # [steering, throttle/brake]
    'hidden_dim': 256,
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,
    
    # TD3 specific
    'tau': 0.005,
    'gamma': 0.99,
    'policy_noise': 0.2,
    'noise_clip': 0.5,
    'policy_freq': 2,
    
    # Action bounds
    'max_action': 1.0,
    
    # Training
    'batch_size': 256,
    'buffer_size': 1000000,
    'warmup_steps': 10000,
    'max_episodes': 5000,
    'max_steps_per_episode': 1000,
    
    # Exploration
    'expl_noise': 0.1,
    'expl_noise_decay': 0.9999,
    'min_expl_noise': 0.01
}

# =============================================================================
# RAIM (LSTM) Settings
# =============================================================================
RAIM_CONFIG = {
    'input_dim': 32,
    'hidden_dim': 128,
    'num_layers': 2,
    'output_dim': 3,  # [straight, left_lane_change, right_lane_change]
    'sequence_length': 10,
    'dropout': 0.2,
    'learning_rate': 1e-3
}

# =============================================================================
# Motion Planning Settings
# =============================================================================
MOTION_PLANNER_CONFIG = {
    'lookahead_distance': 20.0,
    'lateral_pid': {'Kp': 1.0, 'Ki': 0.1, 'Kd': 0.5},
    'longitudinal_pid': {'Kp': 1.0, 'Ki': 0.1, 'Kd': 0.1},
    'safety_margin': 2.0,
    'time_to_collision_threshold': 3.0,
    'lane_change_duration': 3.0
}

# =============================================================================
# Finite State Machine States
# =============================================================================
class VehicleState:
    IDLE = 'IDLE'
    LANE_FOLLOWING = 'LANE_FOLLOWING'
    APPROACHING_INTERSECTION = 'APPROACHING_INTERSECTION'
    NEGOTIATING_INTERSECTION = 'NEGOTIATING_INTERSECTION'
    LANE_CHANGE_LEFT = 'LANE_CHANGE_LEFT'
    LANE_CHANGE_RIGHT = 'LANE_CHANGE_RIGHT'
    EMERGENCY_STOP = 'EMERGENCY_STOP'
    CUT_IN_MANEUVER = 'CUT_IN_MANEUVER'

# =============================================================================
# Reward Function Weights
# =============================================================================
REWARD_WEIGHTS = {
    'progress': 1.0,
    'collision': -100.0,
    'lane_deviation': -0.5,
    'speed_maintenance': 0.2,
    'comfort': -0.1,  # Penalize harsh acceleration/steering
    'intersection_success': 50.0,
    'time_penalty': -0.01
}

# =============================================================================
# Intersection Detection
# =============================================================================
INTERSECTION_CONFIG = {
    'detection_radius': 30.0,  # meters
    'approach_threshold': 50.0,
    'exit_threshold': 10.0
}
