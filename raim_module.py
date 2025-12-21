"""
AutoNavX RAIM (Risk-Aware Intention Module) 

Implements:
- LSTM-based lane change prediction
- Proactive maneuver prediction for surrounding vehicles
- Risk assessment for intersection navigation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from typing import Dict, List, Tuple, Optional
import os

from .config import RAIM_CONFIG


class LSTMPredictor(nn.Module):
    """LSTM network for trajectory/intention prediction."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.2
    ):
        super(LSTMPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        
        # Attention mechanism for temporal attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            hidden: Previous hidden state (optional)
            
        Returns:
            Output predictions and hidden state
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = F.relu(self.input_proj(x))
        x = self.layer_norm(x)
        
        # LSTM forward
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        
        # Self-attention on LSTM outputs
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        out = lstm_out + attn_out
        
        # Use last timestep
        out = out[:, -1, :]
        
        # Output layers
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.fc2(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


class RAIMPredictor:
    """
    Risk-Aware Intention Module for proactive lane change prediction.
    
    Predicts the intention of surrounding vehicles:
    - Straight (continue current lane)
    - Left lane change
    - Right lane change
    """
    
    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = None,
        num_layers: int = None,
        output_dim: int = None,
        sequence_length: int = None,
        device: str = None
    ):
        # Use config defaults if not specified
        self.input_dim = input_dim or RAIM_CONFIG['input_dim']
        self.hidden_dim = hidden_dim or RAIM_CONFIG['hidden_dim']
        self.num_layers = num_layers or RAIM_CONFIG['num_layers']
        self.output_dim = output_dim or RAIM_CONFIG['output_dim']
        self.sequence_length = sequence_length or RAIM_CONFIG['sequence_length']
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize LSTM predictor
        self.model = LSTMPredictor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            dropout=RAIM_CONFIG['dropout']
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=RAIM_CONFIG['learning_rate']
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Hidden state for online prediction
        self.hidden = None
        
        # History buffer for each tracked vehicle
        self.vehicle_histories = {}
        
        # Intention labels
        self.intention_labels = ['STRAIGHT', 'LANE_CHANGE_LEFT', 'LANE_CHANGE_RIGHT']
    
    def _extract_vehicle_features(self, vehicle_state: Dict) -> np.ndarray:
        """
        Extract features from vehicle state for prediction.
        
        Args:
            vehicle_state: Dictionary containing vehicle information
            
        Returns:
            Feature vector
        """
        features = np.zeros(self.input_dim)
        
        # Position features
        if 'position' in vehicle_state:
            pos = vehicle_state['position']
            features[0:3] = pos[:3] if len(pos) >= 3 else np.pad(pos, (0, 3 - len(pos)))
        
        # Velocity features
        if 'velocity' in vehicle_state:
            vel = vehicle_state['velocity']
            features[3:6] = vel[:3] if len(vel) >= 3 else np.pad(vel, (0, 3 - len(vel)))
        
        # Acceleration features
        if 'acceleration' in vehicle_state:
            acc = vehicle_state['acceleration']
            features[6:9] = acc[:3] if len(acc) >= 3 else np.pad(acc, (0, 3 - len(acc)))
        
        # Heading and angular velocity
        if 'heading' in vehicle_state:
            features[9] = vehicle_state['heading']
        if 'angular_velocity' in vehicle_state:
            features[10] = vehicle_state['angular_velocity']
        
        # Distance to ego vehicle
        if 'distance_to_ego' in vehicle_state:
            features[11] = vehicle_state['distance_to_ego']
        
        # Relative velocity to ego
        if 'relative_velocity' in vehicle_state:
            features[12:14] = vehicle_state['relative_velocity'][:2]
        
        # Lane information
        if 'lane_offset' in vehicle_state:
            features[14] = vehicle_state['lane_offset']
        if 'time_in_lane' in vehicle_state:
            features[15] = vehicle_state['time_in_lane']
        
        # Turn signal indicators (if available)
        if 'left_signal' in vehicle_state:
            features[16] = float(vehicle_state['left_signal'])
        if 'right_signal' in vehicle_state:
            features[17] = float(vehicle_state['right_signal'])
        
        # Historical motion features
        if 'lateral_movement' in vehicle_state:
            features[18] = vehicle_state['lateral_movement']
        if 'longitudinal_movement' in vehicle_state:
            features[19] = vehicle_state['longitudinal_movement']
        
        return features.astype(np.float32)
    
    def update_vehicle_history(self, vehicle_id: str, vehicle_state: Dict):
        """
        Update the history buffer for a tracked vehicle.
        
        Args:
            vehicle_id: Unique identifier for the vehicle
            vehicle_state: Current state of the vehicle
        """
        if vehicle_id not in self.vehicle_histories:
            self.vehicle_histories[vehicle_id] = deque(maxlen=self.sequence_length)
        
        features = self._extract_vehicle_features(vehicle_state)
        self.vehicle_histories[vehicle_id].append(features)
    
    def predict_intention(self, vehicle_id: str) -> Tuple[str, np.ndarray]:
        """
        Predict the intention of a tracked vehicle.
        
        Args:
            vehicle_id: Unique identifier for the vehicle
            
        Returns:
            Tuple of (predicted intention, probability distribution)
        """
        if vehicle_id not in self.vehicle_histories:
            return 'STRAIGHT', np.array([0.8, 0.1, 0.1])
        
        history = list(self.vehicle_histories[vehicle_id])
        
        # Pad if not enough history
        while len(history) < self.sequence_length:
            history.insert(0, np.zeros(self.input_dim, dtype=np.float32))
        
        # Convert to tensor
        x = torch.FloatTensor(np.array(history)).unsqueeze(0).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(x)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Get predicted class
        pred_class = np.argmax(probs)
        intention = self.intention_labels[pred_class]
        
        return intention, probs
    
    def predict_all_intentions(self) -> Dict[str, Tuple[str, np.ndarray]]:
        """
        Predict intentions for all tracked vehicles.
        
        Returns:
            Dictionary mapping vehicle_id to (intention, probabilities)
        """
        predictions = {}
        
        for vehicle_id in self.vehicle_histories.keys():
            predictions[vehicle_id] = self.predict_intention(vehicle_id)
        
        return predictions
    
    def get_risky_vehicles(self, risk_threshold: float = 0.3) -> List[Dict]:
        """
        Get vehicles with high lane change probability.
        
        Args:
            risk_threshold: Probability threshold for lane change
            
        Returns:
            List of risky vehicles with their predictions
        """
        risky_vehicles = []
        
        for vehicle_id, (intention, probs) in self.predict_all_intentions().items():
            lane_change_prob = probs[1] + probs[2]  # Left + Right
            
            if lane_change_prob > risk_threshold:
                risky_vehicles.append({
                    'vehicle_id': vehicle_id,
                    'intention': intention,
                    'probabilities': probs,
                    'lane_change_prob': lane_change_prob,
                    'history': list(self.vehicle_histories[vehicle_id])[-1] if self.vehicle_histories[vehicle_id] else None
                })
        
        # Sort by lane change probability
        risky_vehicles.sort(key=lambda x: x['lane_change_prob'], reverse=True)
        
        return risky_vehicles
    
    def compute_risk_score(self, ego_state: Dict) -> float:
        """
        Compute overall risk score based on surrounding vehicles.
        
        Args:
            ego_state: State of the ego vehicle
            
        Returns:
            Risk score between 0 and 1
        """
        if not self.vehicle_histories:
            return 0.0
        
        risk_score = 0.0
        predictions = self.predict_all_intentions()
        
        for vehicle_id, (intention, probs) in predictions.items():
            history = self.vehicle_histories.get(vehicle_id)
            if history and len(history) > 0:
                latest_state = history[-1]
                distance = np.linalg.norm(latest_state[:3])  # Position features
                
                # Distance weighting
                distance_weight = np.exp(-distance / 20.0)  # Decay over 20 meters
                
                # Lane change probability
                lane_change_prob = probs[1] + probs[2]
                
                # Combined risk
                vehicle_risk = distance_weight * lane_change_prob
                risk_score = max(risk_score, vehicle_risk)
        
        return float(np.clip(risk_score, 0.0, 1.0))
    
    def train_step(self, sequences: np.ndarray, labels: np.ndarray) -> float:
        """
        Single training step.
        
        Args:
            sequences: Batch of sequences (batch, seq_len, input_dim)
            labels: Ground truth labels (batch,)
            
        Returns:
            Training loss
        """
        self.model.train()
        
        x = torch.FloatTensor(sequences).to(self.device)
        y = torch.LongTensor(labels).to(self.device)
        
        self.optimizer.zero_grad()
        
        logits, _ = self.model(x)
        loss = self.criterion(logits, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def clear_history(self, vehicle_id: str = None):
        """Clear vehicle history."""
        if vehicle_id is None:
            self.vehicle_histories.clear()
        elif vehicle_id in self.vehicle_histories:
            del self.vehicle_histories[vehicle_id]
    
    def save(self, filepath: str):
        """Save model weights."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
        
        print(f"RAIM model saved to {filepath}")
    
    def load(self, filepath: str) -> bool:
        """Load model weights."""
        if not os.path.exists(filepath):
            print(f"No model found at {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        print(f"RAIM model loaded from {filepath}")
        return True
    
    def get_state_features(self) -> np.ndarray:
        """
        Get RAIM features for RL state.
        
        Returns:
            Feature vector summarizing surrounding vehicle intentions
        """
        features = np.zeros(16, dtype=np.float32)
        
        predictions = self.predict_all_intentions()
        
        if not predictions:
            return features
        
        # Number of tracked vehicles
        features[0] = len(predictions)
        
        # Aggregate probabilities
        all_probs = np.array([probs for _, probs in predictions.values()])
        
        # Mean probabilities
        features[1:4] = np.mean(all_probs, axis=0)
        
        # Max lane change probability
        features[4] = np.max(all_probs[:, 1] + all_probs[:, 2])
        
        # Number of vehicles likely to change lane
        lane_change_threshold = 0.3
        features[5] = np.sum((all_probs[:, 1] + all_probs[:, 2]) > lane_change_threshold)
        
        # Risk score
        features[6] = self.compute_risk_score({})
        
        # Risky vehicles info
        risky = self.get_risky_vehicles()
        if risky:
            features[7] = len(risky)
            features[8] = risky[0]['lane_change_prob']  # Highest risk
        
        return features


class TrajectoryPredictor:
    """
    Predicts future trajectories of surrounding vehicles.
    """
    
    def __init__(self, prediction_horizon: int = 30, dt: float = 0.1):
        self.prediction_horizon = prediction_horizon
        self.dt = dt
        self.trajectory_histories = {}
    
    def update(self, vehicle_id: str, position: np.ndarray, velocity: np.ndarray):
        """Update vehicle trajectory history."""
        if vehicle_id not in self.trajectory_histories:
            self.trajectory_histories[vehicle_id] = deque(maxlen=50)
        
        self.trajectory_histories[vehicle_id].append({
            'position': position.copy(),
            'velocity': velocity.copy()
        })
    
    def predict_trajectory(self, vehicle_id: str) -> np.ndarray:
        """
        Predict future trajectory using constant velocity model.
        
        Args:
            vehicle_id: Vehicle identifier
            
        Returns:
            Predicted trajectory (horizon, 2)
        """
        if vehicle_id not in self.trajectory_histories or len(self.trajectory_histories[vehicle_id]) == 0:
            return np.zeros((self.prediction_horizon, 2))
        
        history = self.trajectory_histories[vehicle_id]
        latest = history[-1]
        
        # Simple constant velocity prediction
        trajectory = np.zeros((self.prediction_horizon, 2))
        position = latest['position'][:2].copy()
        velocity = latest['velocity'][:2].copy()
        
        for t in range(self.prediction_horizon):
            position = position + velocity * self.dt
            trajectory[t] = position
        
        return trajectory
    
    def check_collision(self, ego_trajectory: np.ndarray, vehicle_id: str, 
                        safety_distance: float = 3.0) -> Tuple[bool, int]:
        """
        Check if ego trajectory collides with predicted vehicle trajectory.
        
        Args:
            ego_trajectory: Planned ego trajectory
            vehicle_id: Vehicle to check against
            safety_distance: Minimum safe distance
            
        Returns:
            Tuple of (collision detected, time step of collision)
        """
        predicted = self.predict_trajectory(vehicle_id)
        
        for t in range(min(len(ego_trajectory), len(predicted))):
            dist = np.linalg.norm(ego_trajectory[t] - predicted[t])
            if dist < safety_distance:
                return True, t
        
        return False, -1
