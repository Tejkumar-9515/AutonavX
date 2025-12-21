"""
AutoNavX TD3 (Twin Delayed DDPG) Reinforcement Learning Agent

Implements:
- TD3 algorithm for continuous action space control
- Actor-Critic architecture with twin critics
- Delayed policy updates
- Target policy smoothing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, Optional, Dict
import os

from .config import TD3_CONFIG


class ReplayBuffer:
    """Experience replay buffer for TD3 training."""
    
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 1000000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add a transition to the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions."""
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[ind]),
            torch.FloatTensor(self.actions[ind]),
            torch.FloatTensor(self.rewards[ind]),
            torch.FloatTensor(self.next_states[ind]),
            torch.FloatTensor(self.dones[ind])
        )
    
    def __len__(self):
        return self.size


class Actor(nn.Module):
    """Actor network for TD3."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 max_action: float = 1.0):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        action = self.max_action * torch.tanh(self.fc4(x))
        return action


class Critic(nn.Module):
    """Twin Critic networks for TD3."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4_q1 = nn.Linear(hidden_dim, 1)
        
        self.ln1_q1 = nn.LayerNorm(hidden_dim)
        self.ln2_q1 = nn.LayerNorm(hidden_dim)
        self.ln3_q1 = nn.LayerNorm(hidden_dim)
        
        # Q2 architecture
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4_q2 = nn.Linear(hidden_dim, 1)
        
        self.ln1_q2 = nn.LayerNorm(hidden_dim)
        self.ln2_q2 = nn.LayerNorm(hidden_dim)
        self.ln3_q2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)
        
        # Q1 forward
        q1 = F.relu(self.ln1_q1(self.fc1_q1(sa)))
        q1 = F.relu(self.ln2_q1(self.fc2_q1(q1)))
        q1 = F.relu(self.ln3_q1(self.fc3_q1(q1)))
        q1 = self.fc4_q1(q1)
        
        # Q2 forward
        q2 = F.relu(self.ln1_q2(self.fc1_q2(sa)))
        q2 = F.relu(self.ln2_q2(self.fc2_q2(q2)))
        q2 = F.relu(self.ln3_q2(self.fc3_q2(q2)))
        q2 = self.fc4_q2(q2)
        
        return q1, q2
    
    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q1 value only (for actor update)."""
        sa = torch.cat([state, action], dim=1)
        
        q1 = F.relu(self.ln1_q1(self.fc1_q1(sa)))
        q1 = F.relu(self.ln2_q1(self.fc2_q1(q1)))
        q1 = F.relu(self.ln3_q1(self.fc3_q1(q1)))
        q1 = self.fc4_q1(q1)
        
        return q1


class TD3Agent:
    """
    TD3 (Twin Delayed DDPG) agent for autonomous vehicle control.
    
    Implements:
    - Twin critics to reduce overestimation
    - Delayed policy updates
    - Target policy smoothing
    """
    
    def __init__(
        self,
        state_dim: int = None,
        action_dim: int = None,
        hidden_dim: int = None,
        max_action: float = None,
        actor_lr: float = None,
        critic_lr: float = None,
        tau: float = None,
        gamma: float = None,
        policy_noise: float = None,
        noise_clip: float = None,
        policy_freq: int = None,
        device: str = None
    ):
        # Use config defaults if not specified
        self.state_dim = state_dim or TD3_CONFIG['state_dim']
        self.action_dim = action_dim or TD3_CONFIG['action_dim']
        self.hidden_dim = hidden_dim or TD3_CONFIG['hidden_dim']
        self.max_action = max_action or TD3_CONFIG['max_action']
        self.tau = tau or TD3_CONFIG['tau']
        self.gamma = gamma or TD3_CONFIG['gamma']
        self.policy_noise = policy_noise or TD3_CONFIG['policy_noise']
        self.noise_clip = noise_clip or TD3_CONFIG['noise_clip']
        self.policy_freq = policy_freq or TD3_CONFIG['policy_freq']
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Actor network
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim, self.max_action).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.hidden_dim, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=actor_lr or TD3_CONFIG['actor_lr']
        )
        
        # Critic networks
        self.critic = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=critic_lr or TD3_CONFIG['critic_lr']
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            self.state_dim, 
            self.action_dim, 
            TD3_CONFIG['buffer_size']
        )
        
        # Training step counter
        self.total_steps = 0
        self.training_steps = 0
        
        # Exploration noise
        self.expl_noise = TD3_CONFIG['expl_noise']
        
        # Training metrics
        self.actor_losses = []
        self.critic_losses = []
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select action given state.
        
        Args:
            state: Current state
            add_noise: Whether to add exploration noise
            
        Returns:
            Action array [steering, throttle/brake]
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, self.expl_noise, size=self.action_dim)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def train(self, batch_size: int = None) -> Dict[str, float]:
        """
        Train the agent on a batch of transitions.
        
        Args:
            batch_size: Size of training batch
            
        Returns:
            Dictionary of training metrics
        """
        batch_size = batch_size or TD3_CONFIG['batch_size']
        
        if len(self.replay_buffer) < batch_size:
            return {}
        
        self.training_steps += 1
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        with torch.no_grad():
            # Target policy smoothing
            noise = (
                torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_actions = (
                self.actor_target(next_states) + noise
            ).clamp(-self.max_action, self.max_action)
            
            # Twin critics
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Current Q estimates
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        self.critic_losses.append(critic_loss.item())
        
        metrics = {'critic_loss': critic_loss.item()}
        
        # Delayed policy updates
        if self.training_steps % self.policy_freq == 0:
            # Actor loss
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            self.actor_losses.append(actor_loss.item())
            metrics['actor_loss'] = actor_loss.item()
            
            # Soft update target networks
            self._soft_update()
        
        return metrics
    
    def _soft_update(self):
        """Soft update target networks."""
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                         reward: float, next_state: np.ndarray, done: bool):
        """Store a transition in the replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1
    
    def decay_exploration_noise(self):
        """Decay exploration noise."""
        self.expl_noise = max(
            TD3_CONFIG['min_expl_noise'],
            self.expl_noise * TD3_CONFIG['expl_noise_decay']
        )
    
    def save(self, filepath: str):
        """Save model weights."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'training_steps': self.training_steps,
            'expl_noise': self.expl_noise
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model weights."""
        if not os.path.exists(filepath):
            print(f"No model found at {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_steps = checkpoint['total_steps']
        self.training_steps = checkpoint['training_steps']
        self.expl_noise = checkpoint.get('expl_noise', TD3_CONFIG['expl_noise'])
        
        print(f"Model loaded from {filepath}")
        return True
    
    def action_to_control(self, action: np.ndarray) -> Dict[str, float]:
        """
        Convert network action output to vehicle control.
        
        Args:
            action: Network output [steering, throttle_brake]
            
        Returns:
            Dictionary with steering, throttle, brake
        """
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(action[1])
        
        if throttle_brake >= 0:
            throttle = throttle_brake
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_brake
        
        return {
            'steering': steering,
            'throttle': throttle,
            'brake': brake
        }
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get training statistics."""
        stats = {
            'total_steps': self.total_steps,
            'training_steps': self.training_steps,
            'exploration_noise': self.expl_noise,
            'buffer_size': len(self.replay_buffer)
        }
        
        if self.actor_losses:
            stats['avg_actor_loss'] = np.mean(self.actor_losses[-100:])
        if self.critic_losses:
            stats['avg_critic_loss'] = np.mean(self.critic_losses[-100:])
        
        return stats


class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise."""
    
    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, 
                 sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu
    
    def reset(self):
        """Reset noise state."""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self) -> np.ndarray:
        """Sample noise."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state
