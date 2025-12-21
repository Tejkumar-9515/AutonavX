#!/usr/bin/env python
"""
AutoNavX - Adaptive AI for Intersection Mastery
Main runner script for CARLA simulation

Usage:
    python run_autonavx.py --mode inference
    python run_autonavx.py --mode training --episodes 1000
    python run_autonavx.py --mode demo --traffic 30
"""

import argparse
import os
import sys
import time
import random
import signal

# Add CARLA Python API to path
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
import numpy as np

from autonavx import AutoNavXAgent
from autonavx.config import CARLA_HOST, CARLA_PORT, CARLA_TIMEOUT, VEHICLE_BLUEPRINT


class AutoNavXRunner:
    """
    Main runner class for AutoNavX simulation.
    """
    
    def __init__(self, args):
        self.args = args
        self.client = None
        self.world = None
        self.vehicle = None
        self.agent = None
        self.traffic_vehicles = []
        self.traffic_walkers = []
        
        self.running = True
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\nShutting down AutoNavX...")
        self.running = False
    
    def connect(self):
        """Connect to CARLA server."""
        print(f"Connecting to CARLA server at {self.args.host}:{self.args.port}...")
        
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(CARLA_TIMEOUT)
        
        # Get world
        self.world = self.client.get_world()
        
        # Set synchronous mode if training
        if self.args.mode == 'training':
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
        
        print(f"Connected! Map: {self.world.get_map().name}")
        
        return True
    
    def spawn_vehicle(self):
        """Spawn the ego vehicle."""
        bp_lib = self.world.get_blueprint_library()
        
        # Get vehicle blueprint
        vehicle_bp = bp_lib.find(self.args.vehicle)
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', '255,0,0')  # Red
        vehicle_bp.set_attribute('role_name', 'hero')
        
        # Get spawn points
        spawn_points = self.world.get_map().get_spawn_points()
        
        if not spawn_points:
            print("Error: No spawn points available!")
            return False
        
        # Try to spawn at a good location
        spawn_point = random.choice(spawn_points)
        
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        
        if self.vehicle is None:
            # Try other spawn points
            for sp in spawn_points:
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
                if self.vehicle:
                    break
        
        if self.vehicle is None:
            print("Error: Could not spawn vehicle!")
            return False
        
        print(f"Spawned ego vehicle: {self.vehicle.type_id}")
        
        # Wait for vehicle to be stable
        time.sleep(1.0)
        
        return True
    
    def spawn_traffic(self):
        """Spawn traffic vehicles and pedestrians."""
        if self.args.traffic == 0:
            return
        
        print(f"Spawning {self.args.traffic} traffic vehicles...")
        
        bp_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Filter out spawn point used by ego vehicle
        ego_location = self.vehicle.get_location()
        available_spawn_points = [
            sp for sp in spawn_points 
            if sp.location.distance(ego_location) > 10.0
        ]
        
        # Spawn vehicles
        vehicle_bps = bp_lib.filter('vehicle.*')
        
        for i in range(min(self.args.traffic, len(available_spawn_points))):
            bp = random.choice(vehicle_bps)
            
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            
            sp = available_spawn_points[i]
            vehicle = self.world.try_spawn_actor(bp, sp)
            
            if vehicle:
                self.traffic_vehicles.append(vehicle)
                vehicle.set_autopilot(True)
        
        print(f"Spawned {len(self.traffic_vehicles)} traffic vehicles")
        
        # Spawn pedestrians
        if self.args.walkers > 0:
            self._spawn_walkers()
    
    def _spawn_walkers(self):
        """Spawn pedestrian walkers."""
        print(f"Spawning {self.args.walkers} pedestrians...")
        
        bp_lib = self.world.get_blueprint_library()
        walker_bps = bp_lib.filter('walker.pedestrian.*')
        
        # Get spawn locations
        spawn_locations = []
        for _ in range(self.args.walkers):
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_locations.append(carla.Transform(loc))
        
        # Spawn walkers
        for spawn_loc in spawn_locations:
            bp = random.choice(walker_bps)
            walker = self.world.try_spawn_actor(bp, spawn_loc)
            if walker:
                self.traffic_walkers.append(walker)
        
        print(f"Spawned {len(self.traffic_walkers)} pedestrians")
    
    def get_random_destination(self):
        """Get a random destination for navigation."""
        spawn_points = self.world.get_map().get_spawn_points()
        ego_location = self.vehicle.get_location()
        
        # Filter for destinations at least 100m away
        far_spawn_points = [
            sp for sp in spawn_points
            if sp.location.distance(ego_location) > 100.0
        ]
        
        if far_spawn_points:
            return random.choice(far_spawn_points).location
        else:
            return random.choice(spawn_points).location
    
    def run_inference(self):
        """Run AutoNavX in inference mode."""
        print("\n" + "="*50)
        print("AutoNavX - Inference Mode")
        print("="*50)
        
        # Initialize agent
        self.agent = AutoNavXAgent(
            self.vehicle,
            self.world,
            training_mode=False,
            model_path=self.args.model
        )
        
        # Set destination
        destination = self.get_random_destination()
        self.agent.set_destination(destination)
        print(f"Destination set: {destination}")
        
        # Setup spectator camera
        spectator = self.world.get_spectator()
        
        # Main loop
        print("\nRunning... Press Ctrl+C to stop.\n")
        
        while self.running:
            try:
                # Get control from agent
                control = self.agent.run_step()
                
                # Apply control
                self.vehicle.apply_control(control)
                
                # Update spectator camera
                transform = self.vehicle.get_transform()
                spectator.set_transform(carla.Transform(
                    transform.location + carla.Location(x=-8, z=6),
                    carla.Rotation(pitch=-20, yaw=transform.rotation.yaw)
                ))
                
                # Check if destination reached
                if self.vehicle.get_location().distance(destination) < 5.0:
                    print("Destination reached! Getting new destination...")
                    destination = self.get_random_destination()
                    self.agent.set_destination(destination)
                    print(f"New destination: {destination}")
                
                # Print debug info periodically
                if self.agent.episode_steps % 100 == 0:
                    debug_info = self.agent.get_debug_info()
                    print(f"State: {debug_info['fsm_state']} | "
                          f"Speed: {debug_info['target_speed']:.1f} km/h | "
                          f"Risk: {debug_info['risk_score']:.2f} | "
                          f"Obstacles: {debug_info['obstacles']}")
                
                # Tick world if in synchronous mode
                if self.args.sync:
                    self.world.tick()
                else:
                    time.sleep(0.05)
                
            except Exception as e:
                print(f"Error during step: {e}")
                import traceback
                traceback.print_exc()
                break
    
    def run_training(self):
        """Run AutoNavX in training mode."""
        print("\n" + "="*50)
        print("AutoNavX - Training Mode")
        print(f"Episodes: {self.args.episodes}")
        print("="*50)
        
        # Initialize agent
        self.agent = AutoNavXAgent(
            self.vehicle,
            self.world,
            training_mode=True,
            model_path=self.args.model
        )
        
        # Training loop
        for episode in range(self.args.episodes):
            if not self.running:
                break
            
            print(f"\n--- Episode {episode + 1}/{self.args.episodes} ---")
            
            # Reset vehicle position
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)
            self.vehicle.set_transform(spawn_point)
            
            # Wait for vehicle to stabilize
            for _ in range(10):
                self.world.tick()
            
            # Set new destination
            destination = self.get_random_destination()
            self.agent.set_destination(destination)
            
            # Episode loop
            episode_done = False
            while not episode_done and self.running:
                try:
                    control = self.agent.run_step()
                    self.vehicle.apply_control(control)
                    self.world.tick()
                    
                    # Check episode completion
                    if self.agent._check_episode_done():
                        episode_done = True
                    
                except Exception as e:
                    print(f"Error: {e}")
                    episode_done = True
            
            # Print episode stats
            metrics = self.agent.get_metrics()
            print(f"Episode reward: {metrics.get('avg_reward', 0):.2f}")
            
            # Save model periodically
            if (episode + 1) % self.args.save_freq == 0:
                save_path = os.path.join(self.args.save_dir, f'checkpoint_ep{episode+1}')
                self.agent.save_models(save_path)
        
        # Final save
        self.agent.save_models(os.path.join(self.args.save_dir, 'final'))
        print("\nTraining completed!")
    
    def run_demo(self):
        """Run AutoNavX in demo mode with visualization."""
        print("\n" + "="*50)
        print("AutoNavX - Demo Mode")
        print("="*50)
        
        try:
            import pygame
            HAS_PYGAME = True
        except ImportError:
            HAS_PYGAME = False
            print("Note: pygame not found, running without visualization")
        
        # Initialize agent
        self.agent = AutoNavXAgent(
            self.vehicle,
            self.world,
            training_mode=False,
            model_path=self.args.model
        )
        
        # Set destination
        destination = self.get_random_destination()
        self.agent.set_destination(destination)
        
        # Setup spectator
        spectator = self.world.get_spectator()
        
        # Camera for visualization
        camera = None
        image_data = None
        
        if HAS_PYGAME:
            pygame.init()
            display = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("AutoNavX Demo")
            
            # Attach camera
            bp_lib = self.world.get_blueprint_library()
            camera_bp = bp_lib.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '110')
            
            camera_transform = carla.Transform(carla.Location(x=-5, z=3))
            camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            
            def process_image(image):
                nonlocal image_data
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((600, 800, 4))
                array = array[:, :, :3]
                image_data = array
            
            camera.listen(process_image)
        
        print("\nRunning demo... Press Ctrl+C or close window to stop.\n")
        
        clock = pygame.time.Clock() if HAS_PYGAME else None
        
        while self.running:
            try:
                # Handle pygame events
                if HAS_PYGAME:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                self.running = False
                
                # Get control from agent
                control = self.agent.run_step()
                self.vehicle.apply_control(control)
                
                # Update spectator
                transform = self.vehicle.get_transform()
                spectator.set_transform(carla.Transform(
                    transform.location + carla.Location(x=-10, z=8),
                    carla.Rotation(pitch=-25, yaw=transform.rotation.yaw)
                ))
                
                # Render
                if HAS_PYGAME and image_data is not None:
                    surface = pygame.surfarray.make_surface(image_data.swapaxes(0, 1))
                    display.blit(surface, (0, 0))
                    
                    # Draw debug info
                    font = pygame.font.Font(None, 36)
                    debug_info = self.agent.get_debug_info()
                    
                    texts = [
                        f"State: {debug_info['fsm_state']}",
                        f"Speed: {debug_info['target_speed']:.1f} km/h",
                        f"Risk: {debug_info['risk_score']:.2f}",
                        f"Obstacles: {debug_info['obstacles']}"
                    ]
                    
                    for i, text in enumerate(texts):
                        text_surface = font.render(text, True, (255, 255, 255))
                        display.blit(text_surface, (10, 10 + i * 30))
                    
                    pygame.display.flip()
                    clock.tick(20)
                
                # Check destination
                if self.vehicle.get_location().distance(destination) < 5.0:
                    destination = self.get_random_destination()
                    self.agent.set_destination(destination)
                
                if self.args.sync:
                    self.world.tick()
                else:
                    time.sleep(0.05)
                
            except Exception as e:
                print(f"Error: {e}")
                break
        
        # Cleanup
        if camera:
            camera.stop()
            camera.destroy()
        
        if HAS_PYGAME:
            pygame.quit()
    
    def cleanup(self):
        """Clean up all spawned actors."""
        print("\nCleaning up...")
        
        # Destroy agent
        if self.agent:
            self.agent.destroy()
        
        # Destroy traffic
        for vehicle in self.traffic_vehicles:
            try:
                vehicle.destroy()
            except:
                pass
        
        for walker in self.traffic_walkers:
            try:
                walker.destroy()
            except:
                pass
        
        # Destroy ego vehicle
        if self.vehicle:
            try:
                self.vehicle.destroy()
            except:
                pass
        
        # Reset world settings
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        
        print("Cleanup complete!")
    
    def run(self):
        """Main run method."""
        try:
            if not self.connect():
                return
            
            if not self.spawn_vehicle():
                return
            
            self.spawn_traffic()
            
            if self.args.mode == 'inference':
                self.run_inference()
            elif self.args.mode == 'training':
                self.run_training()
            elif self.args.mode == 'demo':
                self.run_demo()
            else:
                print(f"Unknown mode: {self.args.mode}")
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()


def main():
    argparser = argparse.ArgumentParser(
        description='AutoNavX - Adaptive AI for Intersection Mastery'
    )
    
    # Connection settings
    argparser.add_argument(
        '--host',
        default=CARLA_HOST,
        help='CARLA server host'
    )
    argparser.add_argument(
        '--port',
        default=CARLA_PORT,
        type=int,
        help='CARLA server port'
    )
    
    # Mode settings
    argparser.add_argument(
        '--mode',
        choices=['inference', 'training', 'demo'],
        default='demo',
        help='Running mode'
    )
    
    # Vehicle settings
    argparser.add_argument(
        '--vehicle',
        default=VEHICLE_BLUEPRINT,
        help='Vehicle blueprint'
    )
    
    # Traffic settings
    argparser.add_argument(
        '--traffic',
        type=int,
        default=20,
        help='Number of traffic vehicles'
    )
    argparser.add_argument(
        '--walkers',
        type=int,
        default=10,
        help='Number of pedestrians'
    )
    
    # Training settings
    argparser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Number of training episodes'
    )
    argparser.add_argument(
        '--save-freq',
        type=int,
        default=100,
        help='Save model every N episodes'
    )
    argparser.add_argument(
        '--save-dir',
        default='./autonavx_models',
        help='Directory to save models'
    )
    
    # Model settings
    argparser.add_argument(
        '--model',
        default=None,
        help='Path to pre-trained model'
    )
    
    # Simulation settings
    argparser.add_argument(
        '--sync',
        action='store_true',
        default=True,
        help='Enable synchronous mode'
    )
    
    args = argparser.parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     █████╗ ██╗   ██╗████████╗ ██████╗ ███╗   ██╗ █████╗ ██╗   ██╗██╗  ██╗  ║
    ║    ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗████╗  ██║██╔══██╗██║   ██║╚██╗██╔╝  ║
    ║    ███████║██║   ██║   ██║   ██║   ██║██╔██╗ ██║███████║██║   ██║ ╚███╔╝   ║
    ║    ██╔══██║██║   ██║   ██║   ██║   ██║██║╚██╗██║██╔══██║╚██╗ ██╔╝ ██╔██╗   ║
    ║    ██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║ ╚████║██║  ██║ ╚████╔╝ ██╔╝ ██╗  ║
    ║    ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝  ║
    ║                                                               ║
    ║           Adaptive AI for Intersection Mastery                ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    runner = AutoNavXRunner(args)
    runner.run()


if __name__ == '__main__':
    main()
