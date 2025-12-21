"""
AutoNavX Agent - Main Autonomous Navigation Agent

Integrates:
- Sensor Fusion (LiDAR, Radar, GNSS, IMU)
- TD3 Reinforcement Learning Agent
- RAIM (Risk-Aware Intention Module)
- Motion Planner with FSM
"""

import numpy as np
import carla
import time
from typing import Dict, List, Optional, Tuple
from collections import deque

from .sensor_fusion import SensorFusion
from .rl_agent import TD3Agent
from .raim_module import RAIMPredictor, TrajectoryPredictor
from .motion_planner import MotionPlanner, FiniteStateMachine
from .config import (
    TD3_CONFIG, REWARD_WEIGHTS, VehicleState,
    MAX_SPEED, MIN_SPEED, INTERSECTION_CONFIG
)


class AutoNavXAgent:
    """
    Main AutoNavX autonomous navigation agent.
    
    Combines all components for end-to-end autonomous driving
    at unsignalized intersections.
    """
    
    def __init__(
        self,
        vehicle: carla.Vehicle,
        world: carla.World,
        training_mode: bool = False,
        model_path: str = None
    ):
        """
        Initialize AutoNavX agent.
        
        Args:
            vehicle: CARLA vehicle actor
            world: CARLA world
            training_mode: Enable training mode
            model_path: Path to pre-trained model
        """
        self.vehicle = vehicle
        self.world = world
        self.map = world.get_map()
        self.training_mode = training_mode
        
        # Initialize components
        print("Initializing AutoNavX components...")
        
        # Sensor Fusion
        print("  - Setting up Sensor Fusion...")
        self.sensor_fusion = SensorFusion(vehicle, world)
        
        # Motion Planner
        print("  - Setting up Motion Planner...")
        self.motion_planner = MotionPlanner(vehicle, world)
        
        # RAIM Module
        print("  - Setting up RAIM Module...")
        self.raim = RAIMPredictor()
        self.trajectory_predictor = TrajectoryPredictor()
        
        # TD3 RL Agent
        print("  - Setting up TD3 Agent...")
        self.rl_agent = TD3Agent(
            state_dim=TD3_CONFIG['state_dim'],
            action_dim=TD3_CONFIG['action_dim']
        )
        
        # Load pre-trained model if provided
        if model_path:
            self.rl_agent.load(model_path)
            self.raim.load(model_path.replace('td3', 'raim'))
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.total_episodes = 0
        
        # State tracking
        self.prev_state = None
        self.prev_action = None
        self.prev_location = None
        
        # Destination
        self.destination = None
        
        # Performance metrics
        self.metrics = {
            'collisions': 0,
            'lane_invasions': 0,
            'intersections_cleared': 0,
            'successful_lane_changes': 0,
            'emergency_stops': 0,
            'total_distance': 0.0,
            'episode_rewards': deque(maxlen=100)
        }
        
        # Timing
        self.last_tick_time = time.time()
        self.simulation_time = 0.0
        
        print("AutoNavX Agent initialized successfully!")
    
    def set_destination(self, destination: carla.Location):
        """
        Set navigation destination.
        
        Args:
            destination: Target location
        """
        self.destination = destination
        self.motion_planner.set_destination(destination)
        
        # Reset episode metrics
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.prev_location = self.vehicle.get_location()
        
        self.sensor_fusion.clear_collision_history()
    
    def run_step(self) -> carla.VehicleControl:
        """
        Execute one step of the autonomous navigation.
        
        Returns:
            Vehicle control command
        """
        current_time = time.time()
        dt = current_time - self.last_tick_time
        self.last_tick_time = current_time
        self.simulation_time += dt
        
        self.episode_steps += 1
        
        # Update sensor fusion
        self.sensor_fusion.update(dt)
        
        # Get fused sensor state
        sensor_state = self.sensor_fusion.get_fused_state()
        
        # Get obstacles
        obstacles = self.sensor_fusion.get_obstacles()
        
        # Update surrounding vehicle tracking
        self._update_surrounding_vehicles(obstacles)
        
        # Get RAIM risk assessment
        risk_score = self.raim.compute_risk_score({})
        risky_vehicles = self.raim.get_risky_vehicles()
        
        # Get motion planner features
        planner_state = self.motion_planner.get_state_features()
        
        # Get RAIM features
        raim_state = self.raim.get_state_features()
        
        # Combine all state features
        state = self._build_state(sensor_state, planner_state, raim_state)
        
        # Get action from RL agent
        if self.training_mode:
            action = self.rl_agent.select_action(state, add_noise=True)
        else:
            action = self.rl_agent.select_action(state, add_noise=False)
        
        # Handle high-risk situations
        if risk_score > 0.7 and risky_vehicles:
            # Potential cut-in detected
            self.motion_planner.respond_to_cut_in(self.simulation_time)
        
        # Get base control from motion planner
        base_control = self.motion_planner.update(
            sensor_state, obstacles, risk_score, self.simulation_time
        )
        
        # Blend RL action with motion planner
        final_control = self._blend_controls(base_control, action)
        
        # Calculate reward for training
        if self.training_mode and self.prev_state is not None:
            reward = self._calculate_reward(state, action, obstacles)
            done = self._check_episode_done()
            
            # Store transition
            self.rl_agent.store_transition(
                self.prev_state, self.prev_action, reward, state, done
            )
            
            # Train
            if len(self.rl_agent.replay_buffer) >= TD3_CONFIG['warmup_steps']:
                self.rl_agent.train()
            
            self.episode_reward += reward
            
            if done:
                self._end_episode()
        
        # Update previous state/action
        self.prev_state = state
        self.prev_action = action
        self.prev_location = self.vehicle.get_location()
        
        return final_control
    
    def _build_state(self, sensor_state: np.ndarray, planner_state: np.ndarray,
                     raim_state: np.ndarray) -> np.ndarray:
        """
        Build combined state vector for RL agent.
        
        Args:
            sensor_state: Fused sensor features
            planner_state: Motion planner features
            raim_state: RAIM features
            
        Returns:
            Combined state vector
        """
        # Ensure consistent state dimension
        target_dim = TD3_CONFIG['state_dim']
        
        # Combine states
        combined = np.concatenate([
            sensor_state[:48] if len(sensor_state) >= 48 else np.pad(sensor_state, (0, 48 - len(sensor_state))),
            planner_state,
        ])
        
        # Pad or truncate to target dimension
        if len(combined) < target_dim:
            combined = np.pad(combined, (0, target_dim - len(combined)))
        else:
            combined = combined[:target_dim]
        
        return combined.astype(np.float32)
    
    def _update_surrounding_vehicles(self, obstacles: List[Dict]):
        """Update tracking of surrounding vehicles."""
        for i, obs in enumerate(obstacles):
            vehicle_id = f"vehicle_{i}"
            
            # Build vehicle state for RAIM
            vehicle_state = {
                'position': obs.get('center', np.zeros(3)),
                'velocity': obs.get('velocity', np.zeros(2)),
                'distance_to_ego': np.linalg.norm(obs.get('center', np.zeros(3))[:2])
            }
            
            self.raim.update_vehicle_history(vehicle_id, vehicle_state)
            
            # Update trajectory predictor
            if 'velocity' in obs:
                self.trajectory_predictor.update(
                    vehicle_id,
                    obs.get('center', np.zeros(3)),
                    np.append(obs['velocity'], 0) if len(obs['velocity']) == 2 else obs['velocity']
                )
    
    def _blend_controls(self, base_control: carla.VehicleControl, 
                        rl_action: np.ndarray) -> carla.VehicleControl:
        """
        Blend motion planner control with RL action.
        
        Args:
            base_control: Control from motion planner
            rl_action: Action from RL agent [steering, throttle_brake]
            
        Returns:
            Final blended control
        """
        # Weight for RL action (increases as training progresses)
        if self.training_mode:
            rl_weight = min(0.7, self.rl_agent.total_steps / 100000)
        else:
            rl_weight = 0.5
        
        base_weight = 1.0 - rl_weight
        
        # Convert RL action to control
        rl_control = self.rl_agent.action_to_control(rl_action)
        
        # Blend steering
        blended_steer = (
            base_weight * base_control.steer + 
            rl_weight * rl_control['steering']
        )
        
        # Blend throttle/brake
        blended_throttle = (
            base_weight * base_control.throttle + 
            rl_weight * rl_control['throttle']
        )
        
        blended_brake = (
            base_weight * base_control.brake + 
            rl_weight * rl_control['brake']
        )
        
        # Safety checks
        current_state = self.motion_planner.get_state()
        
        if current_state == VehicleState.EMERGENCY_STOP:
            # Override with emergency stop
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        
        # Ensure throttle and brake aren't both applied
        if blended_throttle > 0.1 and blended_brake > 0.1:
            if blended_brake > blended_throttle:
                blended_throttle = 0.0
            else:
                blended_brake = 0.0
        
        return carla.VehicleControl(
            throttle=float(np.clip(blended_throttle, 0.0, 1.0)),
            steer=float(np.clip(blended_steer, -1.0, 1.0)),
            brake=float(np.clip(blended_brake, 0.0, 1.0))
        )
    
    def _calculate_reward(self, state: np.ndarray, action: np.ndarray,
                         obstacles: List[Dict]) -> float:
        """
        Calculate reward for RL training.
        
        Args:
            state: Current state
            action: Action taken
            obstacles: Detected obstacles
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        current_location = self.vehicle.get_location()
        current_velocity = self.vehicle.get_velocity()
        current_speed = np.sqrt(
            current_velocity.x**2 + current_velocity.y**2
        ) * 3.6  # km/h
        
        # Progress reward
        if self.destination and self.prev_location:
            prev_dist = self.prev_location.distance(self.destination)
            curr_dist = current_location.distance(self.destination)
            progress = prev_dist - curr_dist
            reward += REWARD_WEIGHTS['progress'] * progress
            
            # Track total distance
            self.metrics['total_distance'] += self.prev_location.distance(current_location)
        
        # Collision penalty
        if self.sensor_fusion.has_collision():
            reward += REWARD_WEIGHTS['collision']
            self.metrics['collisions'] += 1
            self.sensor_fusion.clear_collision_history()
        
        # Lane keeping reward
        current_waypoint = self.map.get_waypoint(current_location)
        if current_waypoint:
            lane_center = current_waypoint.transform.location
            lane_offset = current_location.distance(lane_center)
            reward += REWARD_WEIGHTS['lane_deviation'] * lane_offset
        
        # Speed maintenance reward
        target_speed = self.motion_planner.target_speed
        speed_diff = abs(current_speed - target_speed) / target_speed
        reward += REWARD_WEIGHTS['speed_maintenance'] * (1.0 - speed_diff)
        
        # Comfort reward (penalize harsh actions)
        if self.prev_action is not None:
            action_diff = np.abs(action - self.prev_action)
            reward += REWARD_WEIGHTS['comfort'] * np.sum(action_diff)
        
        # Intersection success reward
        fsm_state = self.motion_planner.get_state()
        if fsm_state == VehicleState.LANE_FOLLOWING:
            if hasattr(self, '_was_in_intersection') and self._was_in_intersection:
                reward += REWARD_WEIGHTS['intersection_success']
                self.metrics['intersections_cleared'] += 1
                self._was_in_intersection = False
        elif fsm_state == VehicleState.NEGOTIATING_INTERSECTION:
            self._was_in_intersection = True
        
        # Time penalty
        reward += REWARD_WEIGHTS['time_penalty']
        
        # Emergency stop penalty
        if fsm_state == VehicleState.EMERGENCY_STOP:
            reward -= 1.0
            self.metrics['emergency_stops'] += 1
        
        return float(reward)
    
    def _check_episode_done(self) -> bool:
        """Check if episode should end."""
        current_location = self.vehicle.get_location()
        
        # Reached destination
        if self.destination:
            if current_location.distance(self.destination) < 5.0:
                print("Destination reached!")
                return True
        
        # Collision occurred
        if self.sensor_fusion.has_collision():
            print("Episode ended: Collision")
            return True
        
        # Too many steps
        if self.episode_steps >= TD3_CONFIG['max_steps_per_episode']:
            print("Episode ended: Max steps reached")
            return True
        
        # Stuck (no movement for too long)
        if self.prev_location:
            if current_location.distance(self.prev_location) < 0.01:
                if not hasattr(self, '_stuck_counter'):
                    self._stuck_counter = 0
                self._stuck_counter += 1
                
                if self._stuck_counter > 100:
                    print("Episode ended: Vehicle stuck")
                    return True
            else:
                self._stuck_counter = 0
        
        return False
    
    def _end_episode(self):
        """Handle end of episode."""
        self.total_episodes += 1
        self.metrics['episode_rewards'].append(self.episode_reward)
        
        print(f"\nEpisode {self.total_episodes} Summary:")
        print(f"  Reward: {self.episode_reward:.2f}")
        print(f"  Steps: {self.episode_steps}")
        print(f"  Collisions: {self.metrics['collisions']}")
        print(f"  Distance: {self.metrics['total_distance']:.2f}m")
        
        # Decay exploration noise
        self.rl_agent.decay_exploration_noise()
        
        # Reset for next episode
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.prev_state = None
        self.prev_action = None
        self._was_in_intersection = False
    
    def save_models(self, save_dir: str):
        """Save all models."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        self.rl_agent.save(os.path.join(save_dir, 'td3_model.pth'))
        self.raim.save(os.path.join(save_dir, 'raim_model.pth'))
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, load_dir: str):
        """Load all models."""
        import os
        
        self.rl_agent.load(os.path.join(load_dir, 'td3_model.pth'))
        self.raim.load(os.path.join(load_dir, 'raim_model.pth'))
        
        print(f"Models loaded from {load_dir}")
    
    def get_metrics(self) -> Dict:
        """Get current metrics."""
        metrics = self.metrics.copy()
        
        if self.metrics['episode_rewards']:
            metrics['avg_reward'] = np.mean(self.metrics['episode_rewards'])
        
        metrics['total_episodes'] = self.total_episodes
        metrics['rl_stats'] = self.rl_agent.get_training_stats()
        metrics['current_state'] = self.motion_planner.get_state()
        
        return metrics
    
    def get_debug_info(self) -> Dict:
        """Get debug information for visualization."""
        return {
            'fsm_state': self.motion_planner.get_state(),
            'target_speed': self.motion_planner.target_speed,
            'risk_score': self.raim.compute_risk_score({}),
            'risky_vehicles': len(self.raim.get_risky_vehicles()),
            'obstacles': len(self.sensor_fusion.get_obstacles()),
            'episode_reward': self.episode_reward,
            'episode_steps': self.episode_steps
        }
    
    def destroy(self):
        """Clean up resources."""
        print("Cleaning up AutoNavX agent...")
        self.sensor_fusion.destroy()
        print("AutoNavX agent destroyed.")


class AutoNavXAgentSimple:
    """
    Simplified AutoNavX agent for inference only (no training).
    Lighter weight version for deployment.
    """
    
    def __init__(
        self,
        vehicle: carla.Vehicle,
        world: carla.World,
        model_path: str = None
    ):
        self.vehicle = vehicle
        self.world = world
        self.map = world.get_map()
        
        # Initialize only necessary components
        self.sensor_fusion = SensorFusion(vehicle, world)
        self.motion_planner = MotionPlanner(vehicle, world)
        
        # Load RL agent for inference
        self.rl_agent = TD3Agent()
        if model_path:
            self.rl_agent.load(model_path)
        
        self.last_tick_time = time.time()
        self.simulation_time = 0.0
    
    def set_destination(self, destination: carla.Location):
        """Set destination."""
        self.motion_planner.set_destination(destination)
    
    def run_step(self) -> carla.VehicleControl:
        """Execute one navigation step."""
        current_time = time.time()
        dt = current_time - self.last_tick_time
        self.last_tick_time = current_time
        self.simulation_time += dt
        
        # Update sensors
        self.sensor_fusion.update(dt)
        
        # Get state
        sensor_state = self.sensor_fusion.get_fused_state()
        planner_state = self.motion_planner.get_state_features()
        
        state = np.concatenate([sensor_state[:48], planner_state])
        if len(state) < TD3_CONFIG['state_dim']:
            state = np.pad(state, (0, TD3_CONFIG['state_dim'] - len(state)))
        
        # Get action
        action = self.rl_agent.select_action(state, add_noise=False)
        
        # Get control
        obstacles = self.sensor_fusion.get_obstacles()
        base_control = self.motion_planner.update(
            sensor_state, obstacles, 0.0, self.simulation_time
        )
        
        # Blend controls
        rl_control = self.rl_agent.action_to_control(action)
        
        final_control = carla.VehicleControl(
            throttle=float(np.clip(0.5 * base_control.throttle + 0.5 * rl_control['throttle'], 0, 1)),
            steer=float(np.clip(0.5 * base_control.steer + 0.5 * rl_control['steering'], -1, 1)),
            brake=float(np.clip(0.5 * base_control.brake + 0.5 * rl_control['brake'], 0, 1))
        )
        
        return final_control
    
    def destroy(self):
        """Clean up."""
        self.sensor_fusion.destroy()
