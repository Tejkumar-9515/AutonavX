# AutoNavX - Adaptive AI for Intersection Mastery

## Overview

AutoNavX is a cutting-edge autonomous navigation system that integrates advanced sensor fusion, deep reinforcement learning (DRL), and motion planning to achieve collision-free navigation at unsignalized intersections. Developed and validated in the CARLA simulator, it excels in diverse traffic scenarios.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AutoNavX Agent                           │
├─────────────┬─────────────┬─────────────┬─────────────────────│
│   Sensor    │     TD3     │    RAIM     │     Motion          │
│   Fusion    │  RL Agent   │   Module    │     Planner         │
├─────────────┼─────────────┼─────────────┼─────────────────────│
│ • LiDAR MVF │ • Actor     │ • LSTM      │ • FSM               │
│ • Radar EKF │ • Twin      │ • Attention │ • PID Control       │
│ • GNSS/IMU  │   Critics   │ • Risk      │ • Path Planning     │
│ • Collision │ • Replay    │   Scoring   │ • Intersection      │
│   Detection │   Buffer    │             │   Handling          │
└─────────────┴─────────────┴─────────────┴─────────────────────┘
```

## Key Components

### 1. Sensor Fusion (`sensor_fusion.py`)
- **LiDAR Processor**: Multi-View Fusion (MVF) and Dynamic Voxelization for 3D mapping
- **Radar Processor**: Extended Kalman Filter (EKF) for motion tracking and optical flow
- **GNSS/IMU Processor**: Accurate global positioning and sensor fusion

### 2. TD3 Reinforcement Learning (`rl_agent.py`)
- Twin Delayed DDPG algorithm for optimal decision-making
- Actor-Critic architecture with delayed policy updates
- Experience replay buffer for stable training
- Target policy smoothing to reduce overestimation

### 3. RAIM Module (`raim_module.py`)
- LSTM-based proactive lane change prediction
- Risk assessment for surrounding vehicles
- Trajectory prediction for collision avoidance

### 4. Motion Planner (`motion_planner.py`)
- Finite State Machine (FSM) for behavior management
- States: IDLE, LANE_FOLLOWING, APPROACHING_INTERSECTION, NEGOTIATING_INTERSECTION, LANE_CHANGE, EMERGENCY_STOP, CUT_IN_MANEUVER
- PID controllers for steering and speed control
- Collision-free path planning

## Installation

### Prerequisites
1. CARLA Simulator 0.9.16 installed
2. Python 3.8+
3. CUDA-compatible GPU (recommended for training)

### Setup
```bash
# Navigate to the examples directory
cd PythonAPI/examples

# Install dependencies
pip install -r autonavx/requirements.txt

# Ensure CARLA Python API is accessible
# The API should be in PythonAPI/carla
```

## Usage

### Running Demo Mode
```bash
# Start CARLA server first
./CarlaUE4.exe  # Windows
./CarlaUE4.sh   # Linux

# Run AutoNavX demo
python run_autonavx.py --mode demo --traffic 30
```

### Running Inference Mode
```bash
# Run with pre-trained model
python run_autonavx.py --mode inference --model ./autonavx_models/final/td3_model.pth
```

### Training Mode
```bash
# Basic training
python train_autonavx.py --episodes 5000 --save-dir ./autonavx_training

# Training with curriculum learning
python train_autonavx.py --episodes 5000 --curriculum --save-dir ./autonavx_training

# Resume from checkpoint
python train_autonavx.py --resume ./autonavx_training/checkpoint_ep1000 --episodes 3000
```

## Command-Line Arguments

### run_autonavx.py
| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | demo | Running mode: demo, inference, training |
| `--host` | localhost | CARLA server host |
| `--port` | 2000 | CARLA server port |
| `--vehicle` | vehicle.lincoln.mkz_2020 | Vehicle blueprint |
| `--traffic` | 20 | Number of traffic vehicles |
| `--walkers` | 10 | Number of pedestrians |
| `--model` | None | Path to pre-trained model |

### train_autonavx.py
| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | 5000 | Number of training episodes |
| `--traffic` | 20 | Number of traffic vehicles |
| `--curriculum` | False | Enable curriculum learning |
| `--save-dir` | ./autonavx_training | Model save directory |
| `--save-freq` | 100 | Save checkpoint frequency |
| `--resume` | None | Resume from checkpoint |

## Configuration

All system parameters can be modified in `config.py`:

- **Sensor configurations**: LiDAR, Radar, GNSS, IMU parameters
- **TD3 hyperparameters**: Learning rates, discount factor, noise parameters
- **RAIM settings**: LSTM architecture, prediction horizon
- **Motion planner settings**: PID gains, safety margins, FSM transitions
- **Reward weights**: Balance between progress, safety, and comfort

## File Structure

```
autonavx/
├── __init__.py           # Package initialization
├── config.py             # Configuration parameters
├── sensor_fusion.py      # Sensor fusion module
├── rl_agent.py          # TD3 reinforcement learning agent
├── raim_module.py       # LSTM-based intention prediction
├── motion_planner.py    # FSM and motion planning
├── autonavx_agent.py    # Main agent class
└── requirements.txt     # Python dependencies

run_autonavx.py          # Main runner script
train_autonavx.py        # Training script
```

## State Representation

The RL agent receives a 64-dimensional state vector:
- Fused sensor features (48 dims): LiDAR, Radar, GNSS/IMU
- Motion planner features (16 dims): FSM state, targets, waypoints

## Action Space

The TD3 agent outputs 2 continuous actions:
- Steering: [-1, 1] (left to right)
- Throttle/Brake: [-1, 1] (brake to accelerate)

## Reward Function

The reward function balances multiple objectives:
- **Progress**: Reward for moving toward destination
- **Collision**: Heavy penalty for collisions
- **Lane keeping**: Penalty for lane deviation
- **Speed maintenance**: Reward for maintaining target speed
- **Comfort**: Penalty for harsh steering/acceleration
- **Intersection success**: Bonus for safe intersection traversal

## Training Tips

1. **Start Simple**: Begin with low traffic and clear weather
2. **Curriculum Learning**: Enable `--curriculum` for progressive difficulty
3. **Monitor Metrics**: Check training logs for reward trends
4. **Checkpoint Often**: Models are saved periodically for recovery
5. **GPU Acceleration**: Use CUDA for faster training

## Troubleshooting

### CARLA Connection Issues
```bash
# Ensure CARLA is running
# Check if the port is correct
python run_autonavx.py --port 2000
```

### Out of Memory
- Reduce `--traffic` count
- Lower batch size in config.py
- Use CPU if GPU memory is insufficient

### Vehicle Stuck
- This is handled by the stuck detection in training
- FSM will transition to appropriate recovery state

## Future Improvements

- [ ] Multi-agent training support
- [ ] Vision-based perception (camera integration)
- [ ] V2V communication simulation
- [ ] Highway merging scenarios
- [ ] Weather adaptation module

## License

This project is part of the CARLA ecosystem. Please refer to the CARLA license for usage terms.

## Citation

If you use AutoNavX in your research, please cite:
```
@software{autonavx2024,
  title={AutoNavX: Adaptive AI for Intersection Mastery},
  year={2024},
  description={Deep reinforcement learning for autonomous vehicle navigation}
}
```
