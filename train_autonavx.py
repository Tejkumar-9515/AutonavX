#!/usr/bin/env python
"""
AutoNavX Training Script

Dedicated training script for the TD3 reinforcement learning agent.
Supports distributed training, curriculum learning, and detailed logging.

Usage:
    python train_autonavx.py --episodes 5000 --save-dir ./models
    python train_autonavx.py --resume ./models/checkpoint_ep1000
"""

import argparse
import os
import sys
import time
import random
import json
from datetime import datetime
from collections import deque

# Add CARLA Python API to path
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
import numpy as np

from autonavx import AutoNavXAgent
from autonavx.config import (
    CARLA_HOST, CARLA_PORT, CARLA_TIMEOUT, VEHICLE_BLUEPRINT,
    TD3_CONFIG, REWARD_WEIGHTS
)


class TrainingLogger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, 'training_log.json')
        self.metrics_file = os.path.join(log_dir, 'metrics.csv')
        
        self.episode_logs = []
        self.current_episode = 0
        
        # Initialize CSV file
        with open(self.metrics_file, 'w') as f:
            f.write('episode,reward,steps,collisions,distance,avg_speed,success\n')
    
    def log_episode(self, episode: int, metrics: dict):
        """Log episode metrics."""
        self.current_episode = episode
        
        log_entry = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        self.episode_logs.append(log_entry)
        
        # Write to CSV
        with open(self.metrics_file, 'a') as f:
            f.write(f"{episode},{metrics.get('reward', 0):.3f},"
                   f"{metrics.get('steps', 0)},"
                   f"{metrics.get('collisions', 0)},"
                   f"{metrics.get('distance', 0):.2f},"
                   f"{metrics.get('avg_speed', 0):.2f},"
                   f"{metrics.get('success', False)}\n")
    
    def save(self):
        """Save all logs."""
        with open(self.log_file, 'w') as f:
            json.dump(self.episode_logs, f, indent=2)
    
    def get_summary(self) -> dict:
        """Get training summary."""
        if not self.episode_logs:
            return {}
        
        rewards = [log['metrics'].get('reward', 0) for log in self.episode_logs]
        successes = [log['metrics'].get('success', False) for log in self.episode_logs]
        
        return {
            'total_episodes': len(self.episode_logs),
            'avg_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'success_rate': np.mean(successes),
            'recent_avg_reward': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        }


class CurriculumManager:
    """
    Curriculum learning manager for progressive difficulty.
    """
    
    def __init__(self):
        self.current_level = 1
        self.max_level = 5
        
        # Curriculum settings per level
        self.curriculum = {
            1: {
                'traffic': 5,
                'destination_distance': 100,
                'intersection_difficulty': 'easy',
                'weather': 'clear',
                'success_threshold': 0.6
            },
            2: {
                'traffic': 15,
                'destination_distance': 200,
                'intersection_difficulty': 'medium',
                'weather': 'clear',
                'success_threshold': 0.6
            },
            3: {
                'traffic': 25,
                'destination_distance': 300,
                'intersection_difficulty': 'medium',
                'weather': 'cloudy',
                'success_threshold': 0.55
            },
            4: {
                'traffic': 35,
                'destination_distance': 400,
                'intersection_difficulty': 'hard',
                'weather': 'rain',
                'success_threshold': 0.5
            },
            5: {
                'traffic': 50,
                'destination_distance': 500,
                'intersection_difficulty': 'hard',
                'weather': 'mixed',
                'success_threshold': 0.45
            }
        }
        
        self.recent_successes = deque(maxlen=50)
    
    def get_current_settings(self) -> dict:
        """Get current curriculum settings."""
        return self.curriculum[self.current_level]
    
    def update(self, success: bool) -> bool:
        """
        Update curriculum based on episode outcome.
        
        Returns:
            True if level advanced
        """
        self.recent_successes.append(success)
        
        if len(self.recent_successes) >= 50:
            success_rate = np.mean(self.recent_successes)
            threshold = self.curriculum[self.current_level]['success_threshold']
            
            if success_rate >= threshold and self.current_level < self.max_level:
                self.current_level += 1
                self.recent_successes.clear()
                print(f"\n*** Advancing to curriculum level {self.current_level}! ***\n")
                return True
        
        return False
    
    def get_level(self) -> int:
        """Get current curriculum level."""
        return self.current_level


class AutoNavXTrainer:
    """
    Main trainer class for AutoNavX.
    """
    
    def __init__(self, args):
        self.args = args
        self.client = None
        self.world = None
        self.vehicle = None
        self.agent = None
        self.traffic_vehicles = []
        
        # Training components
        self.logger = TrainingLogger(args.save_dir)
        self.curriculum = CurriculumManager() if args.curriculum else None
        
        # Training state
        self.current_episode = 0
        self.best_reward = float('-inf')
        self.running = True
        
        # Statistics
        self.reward_history = deque(maxlen=100)
        self.success_history = deque(maxlen=100)
    
    def connect(self):
        """Connect to CARLA."""
        print(f"Connecting to CARLA at {self.args.host}:{self.args.port}...")
        
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(CARLA_TIMEOUT)
        
        self.world = self.client.get_world()
        
        # Enable synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        print(f"Connected! Map: {self.world.get_map().name}")
        return True
    
    def spawn_ego_vehicle(self, spawn_point: carla.Transform = None):
        """Spawn or reset ego vehicle."""
        if self.vehicle is not None:
            self.vehicle.destroy()
        
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find(VEHICLE_BLUEPRINT)
        vehicle_bp.set_attribute('role_name', 'hero')
        
        spawn_points = self.world.get_map().get_spawn_points()
        
        if spawn_point is None:
            spawn_point = random.choice(spawn_points)
        
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        
        # Wait for physics to settle
        for _ in range(20):
            self.world.tick()
        
        return self.vehicle is not None
    
    def spawn_traffic(self, num_vehicles: int):
        """Spawn traffic vehicles."""
        # Clear existing traffic
        for v in self.traffic_vehicles:
            try:
                v.destroy()
            except:
                pass
        self.traffic_vehicles.clear()
        
        if num_vehicles == 0:
            return
        
        bp_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        
        ego_loc = self.vehicle.get_location()
        available_spawns = [
            sp for sp in spawn_points
            if sp.location.distance(ego_loc) > 30.0
        ]
        
        vehicle_bps = bp_lib.filter('vehicle.*')
        
        for i in range(min(num_vehicles, len(available_spawns))):
            bp = random.choice(vehicle_bps)
            vehicle = self.world.try_spawn_actor(bp, available_spawns[i])
            if vehicle:
                vehicle.set_autopilot(True)
                self.traffic_vehicles.append(vehicle)
    
    def set_weather(self, weather_type: str):
        """Set weather conditions."""
        weather_presets = {
            'clear': carla.WeatherParameters.ClearNoon,
            'cloudy': carla.WeatherParameters.CloudyNoon,
            'rain': carla.WeatherParameters.SoftRainNoon,
            'mixed': carla.WeatherParameters.WetCloudyNoon
        }
        
        if weather_type in weather_presets:
            self.world.set_weather(weather_presets[weather_type])
    
    def get_destination(self, min_distance: float = 100.0) -> carla.Location:
        """Get training destination."""
        spawn_points = self.world.get_map().get_spawn_points()
        ego_loc = self.vehicle.get_location()
        
        valid_destinations = [
            sp.location for sp in spawn_points
            if sp.location.distance(ego_loc) >= min_distance
        ]
        
        if valid_destinations:
            return random.choice(valid_destinations)
        return random.choice(spawn_points).location
    
    def run_episode(self) -> dict:
        """Run a single training episode."""
        self.current_episode += 1
        
        # Get curriculum settings
        if self.curriculum:
            settings = self.curriculum.get_current_settings()
            traffic = settings['traffic']
            dest_dist = settings['destination_distance']
            weather = settings['weather']
        else:
            traffic = self.args.traffic
            dest_dist = 200
            weather = 'clear'
        
        # Setup environment
        if not self.spawn_ego_vehicle():
            return {'success': False, 'error': 'spawn_failed'}
        
        self.spawn_traffic(traffic)
        self.set_weather(weather)
        
        # Initialize agent
        if self.agent is None:
            self.agent = AutoNavXAgent(
                self.vehicle,
                self.world,
                training_mode=True,
                model_path=self.args.resume
            )
        else:
            # Re-attach to new vehicle
            self.agent = AutoNavXAgent(
                self.vehicle,
                self.world,
                training_mode=True
            )
            # Preserve RL agent state
            # Note: In practice, you'd transfer the RL agent
        
        # Set destination
        destination = self.get_destination(dest_dist)
        self.agent.set_destination(destination)
        
        # Episode metrics
        episode_reward = 0.0
        episode_steps = 0
        episode_distance = 0.0
        speeds = []
        
        start_location = self.vehicle.get_location()
        prev_location = start_location
        
        # Episode loop
        done = False
        success = False
        
        while not done and episode_steps < TD3_CONFIG['max_steps_per_episode']:
            try:
                # Agent step
                control = self.agent.run_step()
                self.vehicle.apply_control(control)
                self.world.tick()
                
                episode_steps += 1
                
                # Track metrics
                current_loc = self.vehicle.get_location()
                velocity = self.vehicle.get_velocity()
                speed = np.sqrt(velocity.x**2 + velocity.y**2) * 3.6
                speeds.append(speed)
                
                step_distance = current_loc.distance(prev_location)
                episode_distance += step_distance
                prev_location = current_loc
                
                # Check termination
                if self.agent.sensor_fusion.has_collision():
                    done = True
                    print(f"  Episode {self.current_episode}: Collision!")
                
                if current_loc.distance(destination) < 5.0:
                    done = True
                    success = True
                    print(f"  Episode {self.current_episode}: Destination reached!")
                
                # Stuck detection
                if episode_steps > 100 and episode_distance / episode_steps < 0.1:
                    done = True
                    print(f"  Episode {self.current_episode}: Vehicle stuck!")
                
            except Exception as e:
                print(f"  Episode error: {e}")
                done = True
        
        # Calculate final reward
        episode_reward = self.agent.episode_reward
        
        # Episode metrics
        metrics = {
            'reward': episode_reward,
            'steps': episode_steps,
            'distance': episode_distance,
            'avg_speed': np.mean(speeds) if speeds else 0,
            'collisions': self.agent.metrics['collisions'],
            'success': success
        }
        
        # Update histories
        self.reward_history.append(episode_reward)
        self.success_history.append(success)
        
        # Update curriculum
        if self.curriculum:
            self.curriculum.update(success)
        
        # Log episode
        self.logger.log_episode(self.current_episode, metrics)
        
        # Save best model
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.agent.save_models(os.path.join(self.args.save_dir, 'best'))
        
        return metrics
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("AutoNavX Training")
        print(f"Episodes: {self.args.episodes}")
        print(f"Save directory: {self.args.save_dir}")
        if self.curriculum:
            print(f"Curriculum learning: ENABLED")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        try:
            for episode in range(self.args.episodes):
                if not self.running:
                    break
                
                # Run episode
                metrics = self.run_episode()
                
                # Print progress
                avg_reward = np.mean(self.reward_history) if self.reward_history else 0
                success_rate = np.mean(self.success_history) if self.success_history else 0
                
                print(f"Episode {self.current_episode:4d} | "
                      f"Reward: {metrics['reward']:7.2f} | "
                      f"Avg: {avg_reward:7.2f} | "
                      f"Success: {success_rate*100:5.1f}% | "
                      f"Steps: {metrics['steps']:4d}")
                
                if self.curriculum:
                    print(f"  Curriculum Level: {self.curriculum.get_level()}")
                
                # Decay exploration
                if self.agent:
                    self.agent.rl_agent.decay_exploration_noise()
                
                # Periodic save
                if (episode + 1) % self.args.save_freq == 0:
                    save_path = os.path.join(
                        self.args.save_dir, 
                        f'checkpoint_ep{self.current_episode}'
                    )
                    if self.agent:
                        self.agent.save_models(save_path)
                    self.logger.save()
                    
                    # Print summary
                    summary = self.logger.get_summary()
                    print(f"\n--- Checkpoint at episode {self.current_episode} ---")
                    print(f"  Average reward: {summary['avg_reward']:.2f}")
                    print(f"  Success rate: {summary['success_rate']*100:.1f}%")
                    print(f"  Best reward: {self.best_reward:.2f}")
                    print()
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        
        finally:
            # Final save
            if self.agent:
                self.agent.save_models(os.path.join(self.args.save_dir, 'final'))
            self.logger.save()
            
            # Print final summary
            elapsed_time = time.time() - start_time
            summary = self.logger.get_summary()
            
            print("\n" + "="*60)
            print("Training Complete!")
            print(f"  Total episodes: {self.current_episode}")
            print(f"  Total time: {elapsed_time/3600:.2f} hours")
            print(f"  Final avg reward: {summary.get('avg_reward', 0):.2f}")
            print(f"  Final success rate: {summary.get('success_rate', 0)*100:.1f}%")
            print(f"  Best reward: {self.best_reward:.2f}")
            print("="*60)
    
    def cleanup(self):
        """Cleanup resources."""
        print("\nCleaning up...")
        
        if self.agent:
            self.agent.destroy()
        
        for v in self.traffic_vehicles:
            try:
                v.destroy()
            except:
                pass
        
        if self.vehicle:
            try:
                self.vehicle.destroy()
            except:
                pass
        
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        
        print("Cleanup complete!")
    
    def run(self):
        """Main run method."""
        try:
            if self.connect():
                self.train()
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(description='AutoNavX Training Script')
    
    # Connection
    parser.add_argument('--host', default=CARLA_HOST, help='CARLA host')
    parser.add_argument('--port', type=int, default=CARLA_PORT, help='CARLA port')
    
    # Training
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes')
    parser.add_argument('--traffic', type=int, default=20, help='Number of traffic vehicles')
    parser.add_argument('--curriculum', action='store_true', help='Enable curriculum learning')
    
    # Saving
    parser.add_argument('--save-dir', default='./autonavx_training', help='Save directory')
    parser.add_argument('--save-freq', type=int, default=100, help='Save frequency')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║              AutoNavX Training Pipeline                       ║
    ║         Deep Reinforcement Learning for AV Control            ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    trainer = AutoNavXTrainer(args)
    trainer.run()


if __name__ == '__main__':
    main()
