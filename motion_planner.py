"""
AutoNavX Motion Planner with Finite State Machine

Implements:
- Finite State Machine (FSM) for vehicle state management
- Collision-free path planning using GNSS-LiDAR fusion
- Cut-in maneuver handling
- PID-based vehicle control
"""

import numpy as np
import carla
from enum import Enum
from typing import Dict, List, Tuple, Optional
from collections import deque
import math

from .config import (
    MOTION_PLANNER_CONFIG, VehicleState, INTERSECTION_CONFIG,
    MAX_SPEED, MIN_SPEED
)


class PIDController:
    """PID controller for vehicle control."""
    
    def __init__(self, Kp: float, Ki: float, Kd: float, 
                 output_limits: Tuple[float, float] = (-1.0, 1.0)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
    
    def compute(self, error: float, dt: float) -> float:
        """
        Compute PID output.
        
        Args:
            error: Current error
            dt: Time step
            
        Returns:
            Control output
        """
        # Proportional term
        P = self.Kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10.0, 10.0)
        I = self.Ki * self.integral
        
        # Derivative term
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0
        D = self.Kd * derivative
        
        self.prev_error = error
        
        # Total output
        output = P + I + D
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        return output
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None


class FiniteStateMachine:
    """
    Finite State Machine for autonomous vehicle behavior management.
    """
    
    def __init__(self):
        self.current_state = VehicleState.IDLE
        self.previous_state = None
        self.state_start_time = 0.0
        self.state_duration = 0.0
        
        # State transition history
        self.transition_history = deque(maxlen=100)
        
        # State-specific data
        self.state_data = {}
        
        # Valid transitions
        self.valid_transitions = {
            VehicleState.IDLE: [
                VehicleState.LANE_FOLLOWING,
                VehicleState.EMERGENCY_STOP
            ],
            VehicleState.LANE_FOLLOWING: [
                VehicleState.APPROACHING_INTERSECTION,
                VehicleState.LANE_CHANGE_LEFT,
                VehicleState.LANE_CHANGE_RIGHT,
                VehicleState.EMERGENCY_STOP,
                VehicleState.CUT_IN_MANEUVER
            ],
            VehicleState.APPROACHING_INTERSECTION: [
                VehicleState.NEGOTIATING_INTERSECTION,
                VehicleState.EMERGENCY_STOP,
                VehicleState.LANE_FOLLOWING
            ],
            VehicleState.NEGOTIATING_INTERSECTION: [
                VehicleState.LANE_FOLLOWING,
                VehicleState.EMERGENCY_STOP
            ],
            VehicleState.LANE_CHANGE_LEFT: [
                VehicleState.LANE_FOLLOWING,
                VehicleState.EMERGENCY_STOP
            ],
            VehicleState.LANE_CHANGE_RIGHT: [
                VehicleState.LANE_FOLLOWING,
                VehicleState.EMERGENCY_STOP
            ],
            VehicleState.EMERGENCY_STOP: [
                VehicleState.IDLE,
                VehicleState.LANE_FOLLOWING
            ],
            VehicleState.CUT_IN_MANEUVER: [
                VehicleState.LANE_FOLLOWING,
                VehicleState.EMERGENCY_STOP
            ]
        }
    
    def transition_to(self, new_state: str, current_time: float, data: Dict = None) -> bool:
        """
        Attempt to transition to a new state.
        
        Args:
            new_state: Target state
            current_time: Current simulation time
            data: Optional state-specific data
            
        Returns:
            True if transition was successful
        """
        # Check if transition is valid
        if new_state not in self.valid_transitions.get(self.current_state, []):
            # Allow any state to transition to EMERGENCY_STOP
            if new_state != VehicleState.EMERGENCY_STOP:
                return False
        
        # Record transition
        self.transition_history.append({
            'from': self.current_state,
            'to': new_state,
            'time': current_time,
            'duration': current_time - self.state_start_time
        })
        
        # Update state
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_start_time = current_time
        self.state_duration = 0.0
        
        # Store state data
        if data:
            self.state_data = data
        else:
            self.state_data = {}
        
        return True
    
    def update(self, dt: float):
        """Update state duration."""
        self.state_duration += dt
    
    def get_state(self) -> str:
        """Get current state."""
        return self.current_state
    
    def get_state_duration(self) -> float:
        """Get time spent in current state."""
        return self.state_duration
    
    def is_in_intersection(self) -> bool:
        """Check if currently handling intersection."""
        return self.current_state in [
            VehicleState.APPROACHING_INTERSECTION,
            VehicleState.NEGOTIATING_INTERSECTION
        ]
    
    def is_changing_lane(self) -> bool:
        """Check if currently changing lane."""
        return self.current_state in [
            VehicleState.LANE_CHANGE_LEFT,
            VehicleState.LANE_CHANGE_RIGHT
        ]


class MotionPlanner:
    """
    Motion planner for collision-free navigation.
    
    Implements:
    - Path planning
    - Obstacle avoidance
    - Intersection negotiation
    - Lane change execution
    """
    
    def __init__(self, vehicle: carla.Vehicle, world: carla.World):
        self.vehicle = vehicle
        self.world = world
        self.map = world.get_map()
        
        # Finite State Machine
        self.fsm = FiniteStateMachine()
        
        # PID Controllers
        lateral_pid = MOTION_PLANNER_CONFIG['lateral_pid']
        longitudinal_pid = MOTION_PLANNER_CONFIG['longitudinal_pid']
        
        self.steering_controller = PIDController(
            lateral_pid['Kp'], lateral_pid['Ki'], lateral_pid['Kd']
        )
        self.throttle_controller = PIDController(
            longitudinal_pid['Kp'], longitudinal_pid['Ki'], longitudinal_pid['Kd'],
            output_limits=(0.0, 1.0)
        )
        self.brake_controller = PIDController(
            longitudinal_pid['Kp'] * 2, longitudinal_pid['Ki'], longitudinal_pid['Kd'],
            output_limits=(0.0, 1.0)
        )
        
        # Target waypoints
        self.waypoints = deque(maxlen=100)
        self.current_waypoint_idx = 0
        
        # Path parameters
        self.lookahead_distance = MOTION_PLANNER_CONFIG['lookahead_distance']
        self.safety_margin = MOTION_PLANNER_CONFIG['safety_margin']
        self.ttc_threshold = MOTION_PLANNER_CONFIG['time_to_collision_threshold']
        
        # Target speed
        self.target_speed = MAX_SPEED
        
        # Lane change parameters
        self.lane_change_start_time = 0.0
        self.lane_change_duration = MOTION_PLANNER_CONFIG['lane_change_duration']
        self.lane_change_target_waypoint = None
        
        # Intersection handling
        self.intersection_waypoints = []
        self.intersection_entered = False
        
        # Last update time
        self.last_update_time = 0.0
    
    def set_destination(self, destination: carla.Location):
        """
        Set destination and compute route.
        
        Args:
            destination: Target location
        """
        start_waypoint = self.map.get_waypoint(self.vehicle.get_location())
        end_waypoint = self.map.get_waypoint(destination)
        
        # Simple route planning using CARLA's waypoint system
        route = self._compute_route(start_waypoint, end_waypoint)
        
        self.waypoints.clear()
        for wp in route:
            self.waypoints.append(wp)
        
        self.current_waypoint_idx = 0
        
        # Start lane following
        self.fsm.transition_to(VehicleState.LANE_FOLLOWING, self.last_update_time)
    
    def _compute_route(self, start: carla.Waypoint, end: carla.Waypoint) -> List[carla.Waypoint]:
        """
        Compute route from start to end waypoint.
        
        Args:
            start: Start waypoint
            end: End waypoint
            
        Returns:
            List of waypoints forming the route
        """
        route = [start]
        current = start
        max_iterations = 1000
        
        for _ in range(max_iterations):
            # Check if reached destination
            if current.transform.location.distance(end.transform.location) < 5.0:
                route.append(end)
                break
            
            # Get next waypoints
            next_waypoints = current.next(2.0)
            if not next_waypoints:
                break
            
            # Choose waypoint closest to destination
            best_wp = min(next_waypoints, 
                          key=lambda wp: wp.transform.location.distance(end.transform.location))
            
            route.append(best_wp)
            current = best_wp
        
        return route
    
    def update(self, sensor_data: Dict, obstacles: List[Dict], 
               risk_score: float, current_time: float) -> carla.VehicleControl:
        """
        Main update function for motion planning.
        
        Args:
            sensor_data: Fused sensor data
            obstacles: Detected obstacles
            risk_score: Risk score from RAIM
            current_time: Current simulation time
            
        Returns:
            Vehicle control command
        """
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Update FSM
        self.fsm.update(dt)
        
        # Get current vehicle state
        vehicle_transform = self.vehicle.get_transform()
        vehicle_velocity = self.vehicle.get_velocity()
        current_speed = np.sqrt(
            vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2
        ) * 3.6  # km/h
        
        current_waypoint = self.map.get_waypoint(vehicle_transform.location)
        
        # Check for intersections
        self._check_intersection(current_waypoint, current_time)
        
        # Check for collision risk
        collision_imminent, ttc = self._check_collision_risk(obstacles)
        
        # State-based behavior
        state = self.fsm.get_state()
        
        if collision_imminent and ttc < 1.0:
            # Emergency brake
            self.fsm.transition_to(VehicleState.EMERGENCY_STOP, current_time)
            return self._emergency_stop()
        
        if state == VehicleState.IDLE:
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        
        elif state == VehicleState.LANE_FOLLOWING:
            return self._lane_following_control(current_speed, dt)
        
        elif state == VehicleState.APPROACHING_INTERSECTION:
            return self._approaching_intersection_control(current_speed, obstacles, dt)
        
        elif state == VehicleState.NEGOTIATING_INTERSECTION:
            return self._negotiating_intersection_control(current_speed, obstacles, dt, current_time)
        
        elif state in [VehicleState.LANE_CHANGE_LEFT, VehicleState.LANE_CHANGE_RIGHT]:
            return self._lane_change_control(current_speed, dt, current_time)
        
        elif state == VehicleState.CUT_IN_MANEUVER:
            return self._cut_in_response_control(current_speed, obstacles, dt)
        
        elif state == VehicleState.EMERGENCY_STOP:
            if current_speed < 1.0 and not collision_imminent:
                self.fsm.transition_to(VehicleState.LANE_FOLLOWING, current_time)
            return self._emergency_stop()
        
        return carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)
    
    def _check_intersection(self, current_waypoint: carla.Waypoint, current_time: float):
        """Check if approaching or at intersection."""
        if current_waypoint.is_junction:
            if self.fsm.get_state() == VehicleState.APPROACHING_INTERSECTION:
                self.fsm.transition_to(VehicleState.NEGOTIATING_INTERSECTION, current_time)
                self.intersection_entered = True
            return
        
        # Check if approaching junction
        for dist in [10.0, 20.0, 30.0]:
            next_wps = current_waypoint.next(dist)
            if next_wps:
                for wp in next_wps:
                    if wp.is_junction:
                        if self.fsm.get_state() == VehicleState.LANE_FOLLOWING:
                            self.fsm.transition_to(
                                VehicleState.APPROACHING_INTERSECTION, 
                                current_time,
                                {'junction_distance': dist}
                            )
                        return
        
        # Exited intersection
        if self.intersection_entered and not current_waypoint.is_junction:
            self.fsm.transition_to(VehicleState.LANE_FOLLOWING, current_time)
            self.intersection_entered = False
    
    def _check_collision_risk(self, obstacles: List[Dict]) -> Tuple[bool, float]:
        """
        Check for collision risk with obstacles.
        
        Returns:
            Tuple of (collision_imminent, time_to_collision)
        """
        if not obstacles:
            return False, float('inf')
        
        vehicle_transform = self.vehicle.get_transform()
        vehicle_velocity = self.vehicle.get_velocity()
        
        min_ttc = float('inf')
        collision_imminent = False
        
        for obs in obstacles:
            if 'ttc' in obs:
                ttc = obs['ttc']
            else:
                # Calculate TTC
                obs_pos = obs.get('center', np.zeros(3))
                rel_pos = np.array([
                    obs_pos[0] - vehicle_transform.location.x,
                    obs_pos[1] - vehicle_transform.location.y
                ])
                
                rel_vel = np.array([vehicle_velocity.x, vehicle_velocity.y])
                if 'velocity' in obs:
                    rel_vel -= obs['velocity'][:2]
                
                distance = np.linalg.norm(rel_pos)
                closing_speed = -np.dot(rel_pos, rel_vel) / (distance + 1e-8)
                
                if closing_speed > 0:
                    ttc = distance / closing_speed
                else:
                    ttc = float('inf')
            
            if ttc < min_ttc:
                min_ttc = ttc
            
            if ttc < self.ttc_threshold:
                collision_imminent = True
        
        return collision_imminent, min_ttc
    
    def _get_target_waypoint(self) -> Optional[carla.Waypoint]:
        """Get the target waypoint for path following."""
        if not self.waypoints:
            return None
        
        vehicle_location = self.vehicle.get_location()
        
        # Find closest waypoint ahead of vehicle
        for i, wp in enumerate(self.waypoints):
            dist = wp.transform.location.distance(vehicle_location)
            if dist < 2.0 and i < len(self.waypoints) - 1:
                continue
            if dist >= self.lookahead_distance * 0.5:
                return wp
        
        return self.waypoints[-1] if self.waypoints else None
    
    def _compute_steering(self, target_waypoint: carla.Waypoint, dt: float) -> float:
        """
        Compute steering angle using Pure Pursuit.
        
        Args:
            target_waypoint: Target waypoint to follow
            dt: Time step
            
        Returns:
            Steering value [-1, 1]
        """
        if target_waypoint is None:
            return 0.0
        
        vehicle_transform = self.vehicle.get_transform()
        
        # Get target in vehicle frame
        target_loc = target_waypoint.transform.location
        
        # Vector from vehicle to target
        vec_to_target = np.array([
            target_loc.x - vehicle_transform.location.x,
            target_loc.y - vehicle_transform.location.y
        ])
        
        # Vehicle forward vector
        yaw = np.radians(vehicle_transform.rotation.yaw)
        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
        
        # Cross product for steering direction
        cross = forward_vec[0] * vec_to_target[1] - forward_vec[1] * vec_to_target[0]
        
        # Dot product for distance
        distance = np.linalg.norm(vec_to_target)
        
        # Pure pursuit steering angle
        if distance > 0:
            curvature = 2.0 * cross / (distance ** 2)
            steering = np.arctan(curvature * 2.5)  # Wheelbase factor
        else:
            steering = 0.0
        
        # Normalize to [-1, 1]
        steering = np.clip(steering / np.radians(70), -1.0, 1.0)
        
        return float(steering)
    
    def _compute_throttle_brake(self, current_speed: float, dt: float) -> Tuple[float, float]:
        """
        Compute throttle and brake values.
        
        Args:
            current_speed: Current vehicle speed in km/h
            dt: Time step
            
        Returns:
            Tuple of (throttle, brake) values
        """
        speed_error = self.target_speed - current_speed
        
        if speed_error > 0:
            throttle = self.throttle_controller.compute(speed_error, dt)
            brake = 0.0
        else:
            throttle = 0.0
            brake = self.brake_controller.compute(-speed_error, dt)
        
        return float(throttle), float(brake)
    
    def _lane_following_control(self, current_speed: float, dt: float) -> carla.VehicleControl:
        """Control for lane following state."""
        target_waypoint = self._get_target_waypoint()
        
        self.target_speed = MAX_SPEED
        
        steering = self._compute_steering(target_waypoint, dt)
        throttle, brake = self._compute_throttle_brake(current_speed, dt)
        
        return carla.VehicleControl(
            throttle=throttle,
            steer=steering,
            brake=brake
        )
    
    def _approaching_intersection_control(self, current_speed: float, 
                                           obstacles: List[Dict], dt: float) -> carla.VehicleControl:
        """Control when approaching intersection."""
        target_waypoint = self._get_target_waypoint()
        
        # Reduce speed when approaching intersection
        junction_distance = self.fsm.state_data.get('junction_distance', 30.0)
        speed_factor = min(1.0, junction_distance / 30.0)
        self.target_speed = max(MIN_SPEED, MAX_SPEED * 0.5 * speed_factor)
        
        steering = self._compute_steering(target_waypoint, dt)
        throttle, brake = self._compute_throttle_brake(current_speed, dt)
        
        return carla.VehicleControl(
            throttle=throttle,
            steer=steering,
            brake=brake
        )
    
    def _negotiating_intersection_control(self, current_speed: float, 
                                          obstacles: List[Dict], dt: float,
                                          current_time: float) -> carla.VehicleControl:
        """Control when negotiating through intersection."""
        target_waypoint = self._get_target_waypoint()
        
        # Check for crossing traffic
        crossing_traffic = False
        for obs in obstacles:
            obs_pos = obs.get('center', np.zeros(3))
            distance = np.linalg.norm(obs_pos[:2])
            
            if distance < 15.0 and 'velocity' in obs:
                obs_vel = obs['velocity']
                if np.linalg.norm(obs_vel) > 2.0:
                    crossing_traffic = True
                    break
        
        if crossing_traffic:
            self.target_speed = MIN_SPEED
        else:
            self.target_speed = MAX_SPEED * 0.6
        
        steering = self._compute_steering(target_waypoint, dt)
        throttle, brake = self._compute_throttle_brake(current_speed, dt)
        
        # Extra caution at intersections
        if current_speed > self.target_speed + 5:
            brake = max(brake, 0.3)
            throttle = 0.0
        
        return carla.VehicleControl(
            throttle=throttle,
            steer=steering,
            brake=brake
        )
    
    def _lane_change_control(self, current_speed: float, dt: float, 
                             current_time: float) -> carla.VehicleControl:
        """Control during lane change."""
        # Check if lane change is complete
        lane_change_progress = (current_time - self.lane_change_start_time) / self.lane_change_duration
        
        if lane_change_progress >= 1.0:
            self.fsm.transition_to(VehicleState.LANE_FOLLOWING, current_time)
            return self._lane_following_control(current_speed, dt)
        
        # Follow target lane waypoint
        if self.lane_change_target_waypoint:
            steering = self._compute_steering(self.lane_change_target_waypoint, dt)
        else:
            steering = 0.0
        
        self.target_speed = MAX_SPEED * 0.8
        throttle, brake = self._compute_throttle_brake(current_speed, dt)
        
        return carla.VehicleControl(
            throttle=throttle,
            steer=steering,
            brake=brake
        )
    
    def _cut_in_response_control(self, current_speed: float, 
                                  obstacles: List[Dict], dt: float) -> carla.VehicleControl:
        """Control in response to vehicle cutting in."""
        target_waypoint = self._get_target_waypoint()
        
        # Find the cutting-in vehicle
        closest_obstacle = None
        min_distance = float('inf')
        
        for obs in obstacles:
            obs_pos = obs.get('center', np.zeros(3))
            distance = np.linalg.norm(obs_pos[:2])
            
            if distance < min_distance:
                min_distance = distance
                closest_obstacle = obs
        
        # Adjust speed based on distance
        if min_distance < 10.0:
            self.target_speed = MIN_SPEED
        elif min_distance < 20.0:
            self.target_speed = MAX_SPEED * 0.5
        else:
            self.target_speed = MAX_SPEED * 0.7
        
        steering = self._compute_steering(target_waypoint, dt)
        throttle, brake = self._compute_throttle_brake(current_speed, dt)
        
        return carla.VehicleControl(
            throttle=throttle,
            steer=steering,
            brake=brake
        )
    
    def _emergency_stop(self) -> carla.VehicleControl:
        """Emergency stop control."""
        return carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0
        )
    
    def initiate_lane_change(self, direction: str, current_time: float) -> bool:
        """
        Initiate a lane change maneuver.
        
        Args:
            direction: 'left' or 'right'
            current_time: Current simulation time
            
        Returns:
            True if lane change initiated successfully
        """
        current_waypoint = self.map.get_waypoint(self.vehicle.get_location())
        
        if direction == 'left':
            target_waypoint = current_waypoint.get_left_lane()
            new_state = VehicleState.LANE_CHANGE_LEFT
        else:
            target_waypoint = current_waypoint.get_right_lane()
            new_state = VehicleState.LANE_CHANGE_RIGHT
        
        if target_waypoint is None:
            return False
        
        self.lane_change_target_waypoint = target_waypoint
        self.lane_change_start_time = current_time
        
        return self.fsm.transition_to(new_state, current_time)
    
    def respond_to_cut_in(self, current_time: float):
        """Respond to a vehicle cutting in."""
        self.fsm.transition_to(VehicleState.CUT_IN_MANEUVER, current_time)
    
    def get_state(self) -> str:
        """Get current FSM state."""
        return self.fsm.get_state()
    
    def get_state_features(self) -> np.ndarray:
        """Get motion planner features for RL state."""
        features = np.zeros(16, dtype=np.float32)
        
        # Current state encoding
        state_encoding = {
            VehicleState.IDLE: 0,
            VehicleState.LANE_FOLLOWING: 1,
            VehicleState.APPROACHING_INTERSECTION: 2,
            VehicleState.NEGOTIATING_INTERSECTION: 3,
            VehicleState.LANE_CHANGE_LEFT: 4,
            VehicleState.LANE_CHANGE_RIGHT: 5,
            VehicleState.EMERGENCY_STOP: 6,
            VehicleState.CUT_IN_MANEUVER: 7
        }
        features[0] = state_encoding.get(self.fsm.get_state(), 0)
        
        # State duration
        features[1] = self.fsm.get_state_duration()
        
        # Target speed
        features[2] = self.target_speed
        
        # Waypoint info
        target_wp = self._get_target_waypoint()
        if target_wp:
            vehicle_loc = self.vehicle.get_location()
            features[3] = target_wp.transform.location.distance(vehicle_loc)
            features[4] = target_wp.transform.rotation.yaw
        
        # Lane change info
        if self.fsm.is_changing_lane():
            features[5] = 1.0
            progress = self.fsm.get_state_duration() / self.lane_change_duration
            features[6] = min(1.0, progress)
        
        # Intersection info
        if self.fsm.is_in_intersection():
            features[7] = 1.0
        
        return features
