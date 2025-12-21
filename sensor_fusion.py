"""
AutoNavX Sensor Fusion Module

Implements:
- LiDAR Processing with Multi-View Fusion (MVF) and Dynamic Voxelization
- Radar Processing with Extended Kalman Filter (EKF) for motion tracking
- GNSS and IMU fusion for accurate localization
"""

import numpy as np
import carla
from collections import deque
from typing import Dict, List, Tuple, Optional
import weakref
import math

from .config import (
    LIDAR_CONFIG, RADAR_CONFIG, GNSS_CONFIG, IMU_CONFIG,
    VOXEL_SIZE, POINT_CLOUD_RANGE, MAX_POINTS_PER_VOXEL, MAX_VOXELS,
    EKF_PROCESS_NOISE, EKF_MEASUREMENT_NOISE, EKF_INITIAL_COVARIANCE
)


class LiDARProcessor:
    """
    LiDAR data processor implementing Multi-View Fusion (MVF) 
    and Dynamic Voxelization for 3D mapping.
    """
    
    def __init__(self, vehicle: carla.Vehicle, world: carla.World):
        self.vehicle = vehicle
        self.world = world
        self.sensor = None
        self.point_cloud = None
        self.voxel_grid = None
        self._sensor_callback = None
        
        # MVF parameters
        self.voxel_size = np.array(VOXEL_SIZE)
        self.point_cloud_range = np.array(POINT_CLOUD_RANGE)
        self.max_points_per_voxel = MAX_POINTS_PER_VOXEL
        self.max_voxels = MAX_VOXELS
        
        # Calculate grid dimensions
        self.grid_size = np.round(
            (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size
        ).astype(np.int32)
        
        self._setup_sensor()
    
    def _setup_sensor(self):
        """Set up the LiDAR sensor on the vehicle."""
        bp_lib = self.world.get_blueprint_library()
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        
        # Configure LiDAR
        lidar_bp.set_attribute('channels', str(LIDAR_CONFIG['channels']))
        lidar_bp.set_attribute('range', str(LIDAR_CONFIG['range']))
        lidar_bp.set_attribute('points_per_second', str(LIDAR_CONFIG['points_per_second']))
        lidar_bp.set_attribute('rotation_frequency', str(LIDAR_CONFIG['rotation_frequency']))
        lidar_bp.set_attribute('upper_fov', str(LIDAR_CONFIG['upper_fov']))
        lidar_bp.set_attribute('lower_fov', str(LIDAR_CONFIG['lower_fov']))
        
        # Spawn sensor
        transform = carla.Transform(
            carla.Location(
                x=LIDAR_CONFIG['position']['x'],
                y=LIDAR_CONFIG['position']['y'],
                z=LIDAR_CONFIG['position']['z']
            )
        )
        
        self.sensor = self.world.spawn_actor(lidar_bp, transform, attach_to=self.vehicle)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda data: LiDARProcessor._on_lidar_data(weak_self, data))
    
    @staticmethod
    def _on_lidar_data(weak_self, data):
        """Callback for LiDAR data."""
        self = weak_self()
        if not self:
            return
        
        # Convert raw data to numpy array
        points = np.frombuffer(data.raw_data, dtype=np.float32)
        points = np.reshape(points, (-1, 4))  # x, y, z, intensity
        self.point_cloud = points
    
    def dynamic_voxelization(self, points: np.ndarray) -> Dict:
        """
        Perform dynamic voxelization on point cloud.
        
        Args:
            points: Nx4 array of points (x, y, z, intensity)
            
        Returns:
            Dictionary containing voxel features and coordinates
        """
        if points is None or len(points) == 0:
            return {'voxels': None, 'coords': None, 'num_points': None}
        
        # Filter points within range
        mask = (
            (points[:, 0] >= self.point_cloud_range[0]) &
            (points[:, 0] < self.point_cloud_range[3]) &
            (points[:, 1] >= self.point_cloud_range[1]) &
            (points[:, 1] < self.point_cloud_range[4]) &
            (points[:, 2] >= self.point_cloud_range[2]) &
            (points[:, 2] < self.point_cloud_range[5])
        )
        points = points[mask]
        
        if len(points) == 0:
            return {'voxels': None, 'coords': None, 'num_points': None}
        
        # Calculate voxel indices
        voxel_coords = np.floor(
            (points[:, :3] - self.point_cloud_range[:3]) / self.voxel_size
        ).astype(np.int32)
        
        # Use dictionary for dynamic voxelization
        voxel_dict = {}
        for i, coord in enumerate(voxel_coords):
            key = tuple(coord)
            if key not in voxel_dict:
                voxel_dict[key] = []
            if len(voxel_dict[key]) < self.max_points_per_voxel:
                voxel_dict[key].append(points[i])
        
        # Convert to arrays
        num_voxels = min(len(voxel_dict), self.max_voxels)
        voxels = np.zeros((num_voxels, self.max_points_per_voxel, 4), dtype=np.float32)
        coords = np.zeros((num_voxels, 3), dtype=np.int32)
        num_points = np.zeros(num_voxels, dtype=np.int32)
        
        for i, (key, pts) in enumerate(list(voxel_dict.items())[:num_voxels]):
            coords[i] = key
            num_points[i] = len(pts)
            voxels[i, :len(pts)] = pts
        
        return {
            'voxels': voxels,
            'coords': coords,
            'num_points': num_points
        }
    
    def multi_view_fusion(self, voxel_data: Dict) -> np.ndarray:
        """
        Apply Multi-View Fusion to extract features from different perspectives.
        
        Args:
            voxel_data: Output from dynamic_voxelization
            
        Returns:
            Fused feature map
        """
        if voxel_data['voxels'] is None:
            return np.zeros((self.grid_size[0], self.grid_size[1], 64))
        
        voxels = voxel_data['voxels']
        coords = voxel_data['coords']
        num_points = voxel_data['num_points']
        
        # Bird's Eye View (BEV) feature extraction
        bev_map = np.zeros((self.grid_size[0], self.grid_size[1], 4))
        
        for i in range(len(coords)):
            x, y, z = coords[i]
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                pts = voxels[i, :num_points[i]]
                # Features: max height, mean intensity, point density, height variance
                bev_map[x, y, 0] = max(bev_map[x, y, 0], np.max(pts[:, 2]))
                bev_map[x, y, 1] = np.mean(pts[:, 3])
                bev_map[x, y, 2] += num_points[i]
                bev_map[x, y, 3] = np.var(pts[:, 2]) if num_points[i] > 1 else 0
        
        # Range View feature extraction
        range_map = self._compute_range_view(voxel_data)
        
        # Fuse BEV and Range views
        fused_features = self._fuse_views(bev_map, range_map)
        
        return fused_features
    
    def _compute_range_view(self, voxel_data: Dict) -> np.ndarray:
        """Compute range view projection of point cloud."""
        if self.point_cloud is None:
            return np.zeros((64, 2048, 5))
        
        points = self.point_cloud
        
        # Spherical projection
        depth = np.linalg.norm(points[:, :3], axis=1)
        yaw = np.arctan2(points[:, 1], points[:, 0])
        pitch = np.arcsin(points[:, 2] / (depth + 1e-8))
        
        # Discretize
        fov_up = np.radians(LIDAR_CONFIG['upper_fov'])
        fov_down = np.radians(LIDAR_CONFIG['lower_fov'])
        fov = fov_up - fov_down
        
        proj_x = 0.5 * (yaw / np.pi + 1.0) * 2048
        proj_y = (1.0 - (pitch - fov_down) / fov) * 64
        
        proj_x = np.clip(proj_x, 0, 2047).astype(np.int32)
        proj_y = np.clip(proj_y, 0, 63).astype(np.int32)
        
        # Create range image
        range_image = np.zeros((64, 2048, 5))
        for i in range(len(points)):
            range_image[proj_y[i], proj_x[i]] = [
                depth[i], points[i, 0], points[i, 1], points[i, 2], points[i, 3]
            ]
        
        return range_image
    
    def _fuse_views(self, bev_map: np.ndarray, range_map: np.ndarray) -> np.ndarray:
        """Fuse BEV and range view features."""
        # Simple concatenation-based fusion (can be replaced with learned fusion)
        bev_flat = bev_map.reshape(-1)
        range_flat = range_map.reshape(-1)
        
        # Reduce dimensionality for state representation
        bev_features = np.mean(bev_map, axis=(0, 1))
        range_features = np.mean(range_map, axis=(0, 1))
        
        return np.concatenate([bev_features, range_features])
    
    def get_obstacles(self) -> List[Dict]:
        """
        Detect obstacles from LiDAR data.
        
        Returns:
            List of detected obstacles with position and dimensions
        """
        if self.point_cloud is None:
            return []
        
        # Simple clustering-based obstacle detection
        voxel_data = self.dynamic_voxelization(self.point_cloud)
        if voxel_data['voxels'] is None:
            return []
        
        obstacles = []
        coords = voxel_data['coords']
        voxels = voxel_data['voxels']
        num_points = voxel_data['num_points']
        
        # Group nearby voxels into obstacles
        visited = set()
        for i in range(len(coords)):
            key = tuple(coords[i])
            if key in visited:
                continue
            
            # Simple region growing
            cluster_points = []
            stack = [i]
            while stack:
                idx = stack.pop()
                coord_key = tuple(coords[idx])
                if coord_key in visited:
                    continue
                visited.add(coord_key)
                
                pts = voxels[idx, :num_points[idx]]
                cluster_points.extend(pts.tolist())
            
            if len(cluster_points) > 10:  # Minimum points for obstacle
                cluster_points = np.array(cluster_points)
                obstacles.append({
                    'center': np.mean(cluster_points[:, :3], axis=0),
                    'min_bound': np.min(cluster_points[:, :3], axis=0),
                    'max_bound': np.max(cluster_points[:, :3], axis=0),
                    'num_points': len(cluster_points)
                })
        
        return obstacles
    
    def get_state_features(self) -> np.ndarray:
        """Get processed LiDAR features for RL state."""
        if self.point_cloud is None:
            return np.zeros(16)
        
        voxel_data = self.dynamic_voxelization(self.point_cloud)
        features = self.multi_view_fusion(voxel_data)
        
        # Compress to fixed size
        if len(features) > 16:
            features = features[:16]
        else:
            features = np.pad(features, (0, 16 - len(features)))
        
        return features.astype(np.float32)
    
    def destroy(self):
        """Clean up sensor."""
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()


class RadarProcessor:
    """
    Radar data processor with Extended Kalman Filter (EKF) 
    for motion tracking and optical flow.
    """
    
    def __init__(self, vehicle: carla.Vehicle, world: carla.World):
        self.vehicle = vehicle
        self.world = world
        self.sensor = None
        self.radar_data = None
        
        # EKF state: [x, y, vx, vy]
        self.ekf_state = np.zeros(4)
        self.ekf_covariance = EKF_INITIAL_COVARIANCE.copy()
        self.process_noise = EKF_PROCESS_NOISE
        self.measurement_noise = EKF_MEASUREMENT_NOISE
        
        # Track history for optical flow
        self.track_history = deque(maxlen=10)
        self.tracked_objects = {}
        
        self._setup_sensor()
    
    def _setup_sensor(self):
        """Set up the radar sensor on the vehicle."""
        bp_lib = self.world.get_blueprint_library()
        radar_bp = bp_lib.find('sensor.other.radar')
        
        # Configure radar
        radar_bp.set_attribute('horizontal_fov', str(RADAR_CONFIG['horizontal_fov']))
        radar_bp.set_attribute('vertical_fov', str(RADAR_CONFIG['vertical_fov']))
        radar_bp.set_attribute('range', str(RADAR_CONFIG['range']))
        radar_bp.set_attribute('points_per_second', str(RADAR_CONFIG['points_per_second']))
        
        # Spawn sensor
        transform = carla.Transform(
            carla.Location(
                x=RADAR_CONFIG['position']['x'],
                y=RADAR_CONFIG['position']['y'],
                z=RADAR_CONFIG['position']['z']
            )
        )
        
        self.sensor = self.world.spawn_actor(radar_bp, transform, attach_to=self.vehicle)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda data: RadarProcessor._on_radar_data(weak_self, data))
    
    @staticmethod
    def _on_radar_data(weak_self, data):
        """Callback for radar data."""
        self = weak_self()
        if not self:
            return
        
        # Process radar detections
        detections = []
        for detection in data:
            detections.append({
                'altitude': detection.altitude,
                'azimuth': detection.azimuth,
                'depth': detection.depth,
                'velocity': detection.velocity
            })
        self.radar_data = detections
        
        # Update tracked objects
        self._update_tracking(detections)
    
    def _update_tracking(self, detections: List[Dict]):
        """Update object tracking with new detections."""
        current_time = self.world.get_snapshot().timestamp.elapsed_seconds
        
        for det in detections:
            # Convert polar to Cartesian
            x = det['depth'] * np.cos(det['azimuth']) * np.cos(det['altitude'])
            y = det['depth'] * np.sin(det['azimuth']) * np.cos(det['altitude'])
            vx = det['velocity'] * np.cos(det['azimuth'])
            vy = det['velocity'] * np.sin(det['azimuth'])
            
            measurement = np.array([x, y, vx, vy])
            
            # Find closest tracked object or create new
            matched = False
            for obj_id, obj in self.tracked_objects.items():
                dist = np.linalg.norm(obj['state'][:2] - measurement[:2])
                if dist < 5.0:  # Association threshold
                    # Update with EKF
                    self._ekf_update(obj, measurement)
                    obj['last_seen'] = current_time
                    matched = True
                    break
            
            if not matched:
                # Create new track
                new_id = len(self.tracked_objects)
                self.tracked_objects[new_id] = {
                    'state': measurement.copy(),
                    'covariance': EKF_INITIAL_COVARIANCE.copy(),
                    'last_seen': current_time
                }
        
        # Remove stale tracks
        stale_ids = [
            obj_id for obj_id, obj in self.tracked_objects.items()
            if current_time - obj['last_seen'] > 1.0
        ]
        for obj_id in stale_ids:
            del self.tracked_objects[obj_id]
    
    def _ekf_predict(self, obj: Dict, dt: float):
        """EKF prediction step."""
        # State transition matrix
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        obj['state'] = F @ obj['state']
        obj['covariance'] = F @ obj['covariance'] @ F.T + self.process_noise * dt
    
    def _ekf_update(self, obj: Dict, measurement: np.ndarray):
        """EKF update step."""
        # Measurement matrix (direct observation)
        H = np.eye(4)
        
        # Innovation
        y = measurement - H @ obj['state']
        
        # Innovation covariance
        S = H @ obj['covariance'] @ H.T + self.measurement_noise
        
        # Kalman gain
        K = obj['covariance'] @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        obj['state'] = obj['state'] + K @ y
        obj['covariance'] = (np.eye(4) - K @ H) @ obj['covariance']
    
    def compute_optical_flow(self) -> np.ndarray:
        """
        Compute optical flow from radar detections.
        
        Returns:
            Flow vectors for tracked objects
        """
        flow_vectors = []
        
        for obj_id, obj in self.tracked_objects.items():
            flow_vectors.append({
                'position': obj['state'][:2],
                'velocity': obj['state'][2:4]
            })
        
        return flow_vectors
    
    def get_approaching_vehicles(self) -> List[Dict]:
        """
        Get vehicles approaching the ego vehicle.
        
        Returns:
            List of approaching vehicles with position, velocity, and TTC
        """
        approaching = []
        
        for obj_id, obj in self.tracked_objects.items():
            x, y, vx, vy = obj['state']
            
            # Check if approaching (negative relative velocity in radial direction)
            radial_velocity = (x * vx + y * vy) / (np.sqrt(x**2 + y**2) + 1e-8)
            
            if radial_velocity < -0.5:  # Approaching
                distance = np.sqrt(x**2 + y**2)
                ttc = -distance / radial_velocity if radial_velocity < 0 else float('inf')
                
                approaching.append({
                    'position': np.array([x, y]),
                    'velocity': np.array([vx, vy]),
                    'distance': distance,
                    'ttc': ttc
                })
        
        return approaching
    
    def get_state_features(self) -> np.ndarray:
        """Get processed radar features for RL state."""
        features = np.zeros(16)
        
        if self.radar_data is None or len(self.radar_data) == 0:
            return features
        
        # Extract key features from radar data
        approaching = self.get_approaching_vehicles()
        
        if len(approaching) > 0:
            # Closest approaching vehicle
            closest = min(approaching, key=lambda x: x['distance'])
            features[0] = closest['distance']
            features[1] = closest['ttc']
            features[2:4] = closest['velocity']
            
            # Number of approaching vehicles
            features[4] = len(approaching)
            
            # Average TTC
            features[5] = np.mean([v['ttc'] for v in approaching])
        
        # Tracked objects statistics
        if len(self.tracked_objects) > 0:
            positions = np.array([obj['state'][:2] for obj in self.tracked_objects.values()])
            velocities = np.array([obj['state'][2:4] for obj in self.tracked_objects.values()])
            
            features[6] = len(self.tracked_objects)
            features[7:9] = np.mean(positions, axis=0)
            features[9:11] = np.std(positions, axis=0)
            features[11:13] = np.mean(velocities, axis=0)
        
        return features.astype(np.float32)
    
    def destroy(self):
        """Clean up sensor."""
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()


class GNSSIMUProcessor:
    """
    GNSS and IMU fusion for accurate global positioning and localization.
    """
    
    def __init__(self, vehicle: carla.Vehicle, world: carla.World):
        self.vehicle = vehicle
        self.world = world
        self.gnss_sensor = None
        self.imu_sensor = None
        
        # State estimation
        self.position = np.zeros(3)  # lat, lon, alt
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.orientation = np.zeros(3)  # roll, pitch, yaw
        self.angular_velocity = np.zeros(3)
        
        # Kalman filter for sensor fusion
        self.state = np.zeros(9)  # [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.covariance = np.eye(9) * 10
        
        # Data buffers
        self.gnss_data = None
        self.imu_data = None
        self.last_update_time = 0
        
        self._setup_sensors()
    
    def _setup_sensors(self):
        """Set up GNSS and IMU sensors."""
        bp_lib = self.world.get_blueprint_library()
        
        # GNSS sensor
        gnss_bp = bp_lib.find('sensor.other.gnss')
        gnss_bp.set_attribute('noise_alt_stddev', str(GNSS_CONFIG['noise_alt_stddev']))
        gnss_bp.set_attribute('noise_lat_stddev', str(GNSS_CONFIG['noise_lat_stddev']))
        gnss_bp.set_attribute('noise_lon_stddev', str(GNSS_CONFIG['noise_lon_stddev']))
        
        gnss_transform = carla.Transform(carla.Location(x=0, y=0, z=2))
        self.gnss_sensor = self.world.spawn_actor(gnss_bp, gnss_transform, attach_to=self.vehicle)
        
        weak_self = weakref.ref(self)
        self.gnss_sensor.listen(lambda data: GNSSIMUProcessor._on_gnss_data(weak_self, data))
        
        # IMU sensor
        imu_bp = bp_lib.find('sensor.other.imu')
        imu_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
        self.imu_sensor = self.world.spawn_actor(imu_bp, imu_transform, attach_to=self.vehicle)
        
        self.imu_sensor.listen(lambda data: GNSSIMUProcessor._on_imu_data(weak_self, data))
    
    @staticmethod
    def _on_gnss_data(weak_self, data):
        """Callback for GNSS data."""
        self = weak_self()
        if not self:
            return
        
        self.gnss_data = {
            'latitude': data.latitude,
            'longitude': data.longitude,
            'altitude': data.altitude
        }
        self.position = np.array([data.latitude, data.longitude, data.altitude])
    
    @staticmethod
    def _on_imu_data(weak_self, data):
        """Callback for IMU data."""
        self = weak_self()
        if not self:
            return
        
        self.imu_data = {
            'accelerometer': np.array([
                data.accelerometer.x,
                data.accelerometer.y,
                data.accelerometer.z
            ]),
            'gyroscope': np.array([
                data.gyroscope.x,
                data.gyroscope.y,
                data.gyroscope.z
            ]),
            'compass': data.compass
        }
        
        self.acceleration = self.imu_data['accelerometer']
        self.angular_velocity = self.imu_data['gyroscope']
        self.orientation[2] = self.imu_data['compass']  # yaw from compass
    
    def fuse_sensors(self, dt: float):
        """
        Fuse GNSS and IMU data using Extended Kalman Filter.
        
        Args:
            dt: Time step since last update
        """
        if self.gnss_data is None or self.imu_data is None:
            return
        
        # Prediction step using IMU
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Process noise
        Q = np.eye(9) * 0.1
        Q[3:6, 3:6] *= 0.5  # Lower noise for velocity
        
        # Predict
        self.state = F @ self.state
        self.state[3:6] += self.acceleration * dt
        self.state[6:9] += self.angular_velocity * dt
        self.covariance = F @ self.covariance @ F.T + Q
        
        # Update with GNSS measurement
        H = np.zeros((3, 9))
        H[0:3, 0:3] = np.eye(3)
        
        R = np.eye(3) * 0.5  # GNSS measurement noise
        
        gnss_pos = self._gnss_to_local(self.position)
        y = gnss_pos - H @ self.state
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.covariance = (np.eye(9) - K @ H) @ self.covariance
    
    def _gnss_to_local(self, gnss_pos: np.ndarray) -> np.ndarray:
        """Convert GNSS coordinates to local frame."""
        # Simple conversion (in real implementation, use proper geodetic conversion)
        # Reference point at map origin
        ref_lat = 0.0
        ref_lon = 0.0
        
        x = (gnss_pos[1] - ref_lon) * 111320 * np.cos(np.radians(gnss_pos[0]))
        y = (gnss_pos[0] - ref_lat) * 110540
        z = gnss_pos[2]
        
        return np.array([x, y, z])
    
    def get_pose(self) -> Dict:
        """
        Get current pose estimate.
        
        Returns:
            Dictionary with position, velocity, orientation
        """
        return {
            'position': self.state[0:3].copy(),
            'velocity': self.state[3:6].copy(),
            'orientation': self.state[6:9].copy(),
            'acceleration': self.acceleration.copy(),
            'angular_velocity': self.angular_velocity.copy()
        }
    
    def get_state_features(self) -> np.ndarray:
        """Get processed GNSS/IMU features for RL state."""
        features = np.zeros(16)
        
        # Position
        features[0:3] = self.state[0:3]
        # Velocity
        features[3:6] = self.state[3:6]
        # Orientation
        features[6:9] = self.state[6:9]
        # Acceleration
        features[9:12] = self.acceleration
        # Angular velocity
        features[12:15] = self.angular_velocity
        # Speed magnitude
        features[15] = np.linalg.norm(self.state[3:6])
        
        return features.astype(np.float32)
    
    def destroy(self):
        """Clean up sensors."""
        if self.gnss_sensor is not None:
            self.gnss_sensor.stop()
            self.gnss_sensor.destroy()
        if self.imu_sensor is not None:
            self.imu_sensor.stop()
            self.imu_sensor.destroy()


class SensorFusion:
    """
    Main sensor fusion class that combines all sensor modalities.
    """
    
    def __init__(self, vehicle: carla.Vehicle, world: carla.World):
        self.vehicle = vehicle
        self.world = world
        
        # Initialize all sensor processors
        self.lidar = LiDARProcessor(vehicle, world)
        self.radar = RadarProcessor(vehicle, world)
        self.gnss_imu = GNSSIMUProcessor(vehicle, world)
        
        # Collision sensor
        self.collision_sensor = None
        self.collision_history = deque(maxlen=10)
        self._setup_collision_sensor()
        
        # Lane invasion sensor
        self.lane_sensor = None
        self.lane_invasion_history = deque(maxlen=10)
        self._setup_lane_sensor()
        
        self.last_fusion_time = 0
    
    def _setup_collision_sensor(self):
        """Set up collision detection sensor."""
        bp_lib = self.world.get_blueprint_library()
        collision_bp = bp_lib.find('sensor.other.collision')
        
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, 
            carla.Transform(), 
            attach_to=self.vehicle
        )
        
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(
            lambda event: SensorFusion._on_collision(weak_self, event)
        )
    
    @staticmethod
    def _on_collision(weak_self, event):
        """Collision callback."""
        self = weak_self()
        if not self:
            return
        
        self.collision_history.append({
            'frame': event.frame,
            'actor': event.other_actor.type_id if event.other_actor else 'unknown',
            'impulse': np.array([
                event.normal_impulse.x,
                event.normal_impulse.y,
                event.normal_impulse.z
            ])
        })
    
    def _setup_lane_sensor(self):
        """Set up lane invasion sensor."""
        bp_lib = self.world.get_blueprint_library()
        lane_bp = bp_lib.find('sensor.other.lane_invasion')
        
        self.lane_sensor = self.world.spawn_actor(
            lane_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        
        weak_self = weakref.ref(self)
        self.lane_sensor.listen(
            lambda event: SensorFusion._on_lane_invasion(weak_self, event)
        )
    
    @staticmethod
    def _on_lane_invasion(weak_self, event):
        """Lane invasion callback."""
        self = weak_self()
        if not self:
            return
        
        self.lane_invasion_history.append({
            'frame': event.frame,
            'markings': [str(m.type) for m in event.crossed_lane_markings]
        })
    
    def update(self, dt: float):
        """
        Update all sensor fusion.
        
        Args:
            dt: Time step since last update
        """
        self.gnss_imu.fuse_sensors(dt)
        self.last_fusion_time += dt
    
    def get_fused_state(self) -> np.ndarray:
        """
        Get fused state from all sensors for RL.
        
        Returns:
            State vector combining all sensor features
        """
        # Get features from each sensor
        lidar_features = self.lidar.get_state_features()
        radar_features = self.radar.get_state_features()
        gnss_imu_features = self.gnss_imu.get_state_features()
        
        # Vehicle state from CARLA
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        
        vehicle_features = np.array([
            transform.location.x,
            transform.location.y,
            transform.location.z,
            np.radians(transform.rotation.yaw),
            velocity.x,
            velocity.y,
            np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # km/h
        ])
        
        # Collision and lane invasion flags
        safety_features = np.array([
            1.0 if len(self.collision_history) > 0 else 0.0,
            1.0 if len(self.lane_invasion_history) > 0 else 0.0
        ])
        
        # Pad vehicle features to match
        vehicle_features = np.pad(vehicle_features, (0, 16 - len(vehicle_features)))
        safety_features = np.pad(safety_features, (0, 7))
        
        # Concatenate all features
        fused_state = np.concatenate([
            lidar_features,    # 16 features
            radar_features,    # 16 features
            gnss_imu_features, # 16 features
            vehicle_features,  # 16 features
        ])
        
        return fused_state.astype(np.float32)
    
    def get_obstacles(self) -> List[Dict]:
        """Get obstacles from all sensors."""
        lidar_obstacles = self.lidar.get_obstacles()
        radar_vehicles = self.radar.get_approaching_vehicles()
        
        # Combine and deduplicate
        all_obstacles = lidar_obstacles.copy()
        
        for rv in radar_vehicles:
            # Check if already in lidar obstacles
            is_duplicate = False
            for lo in lidar_obstacles:
                if np.linalg.norm(rv['position'] - lo['center'][:2]) < 3.0:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_obstacles.append({
                    'center': np.array([rv['position'][0], rv['position'][1], 0]),
                    'velocity': rv['velocity'],
                    'ttc': rv['ttc'],
                    'source': 'radar'
                })
        
        return all_obstacles
    
    def has_collision(self) -> bool:
        """Check if collision occurred recently."""
        return len(self.collision_history) > 0
    
    def clear_collision_history(self):
        """Clear collision history."""
        self.collision_history.clear()
    
    def destroy(self):
        """Clean up all sensors."""
        self.lidar.destroy()
        self.radar.destroy()
        self.gnss_imu.destroy()
        
        if self.collision_sensor is not None:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
        
        if self.lane_sensor is not None:
            self.lane_sensor.stop()
            self.lane_sensor.destroy()
