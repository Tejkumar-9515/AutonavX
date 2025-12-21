"""
AutoNavX - Adaptive AI for Intersection Mastery

A cutting-edge autonomous navigation system integrating:
- Advanced sensor fusion (LiDAR, Radar, GNSS, IMU)
- Deep Reinforcement Learning (TD3 Algorithm)
- RAIM (LSTM-based lane change prediction)
- Motion planning with Finite State Machine

Developed for CARLA Simulator
"""

from .sensor_fusion import SensorFusion, LiDARProcessor, RadarProcessor, GNSSIMUProcessor
from .rl_agent import TD3Agent, ReplayBuffer
from .raim_module import RAIMPredictor
from .motion_planner import MotionPlanner, FiniteStateMachine
from .autonavx_agent import AutoNavXAgent

__version__ = "1.0.0"
__author__ = "AutoNavX Team"

__all__ = [
    'SensorFusion',
    'LiDARProcessor', 
    'RadarProcessor',
    'GNSSIMUProcessor',
    'TD3Agent',
    'ReplayBuffer',
    'RAIMPredictor',
    'MotionPlanner',
    'FiniteStateMachine',
    'AutoNavXAgent'
]
