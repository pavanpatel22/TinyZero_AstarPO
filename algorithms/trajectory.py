from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch
import numpy as np

@dataclass
class TrajectoryStep:
    """Single step in a trajectory."""
    state: str
    action: str
    reward: float
    log_prob: torch.Tensor
    value: Optional[float] = None
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class Trajectory:
    """Complete trajectory of state-action-reward sequences"""
    trajectory_id: str
    environment: str
    initial_state: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_reward: float = 0.0
    terminal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.trajectory_id:
            import uuid
            self.trajectory_id = str(uuid.uuid4())[:8]
    
    def add_step(self, step: TrajectoryStep):
        """Add a step to the trajectory."""
        self.steps.append(step)
        self.total_reward += step.reward
        self.terminal = step.done
        
    def get_sequence_length(self) -> int:
        """Get number of steps in trajectory."""
        return len(self.steps)
    
    def get_final_state(self) -> str:
        """Get the final state of the trajectory."""
        if self.steps:
            return self.steps[-1].state
        return self.initial_state
        
    def to_tensor_dict(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Convert trajectory to tensor dictionary for training."""
        if not self.steps:
            return {}
            
        rewards = torch.tensor([step.reward for step in self.steps], 
                              device=device, dtype=torch.float32)
        log_probs = torch.stack([step.log_prob for step in self.steps]).to(device)
        
        tensor_dict = {
            'rewards': rewards,
            'log_probs': log_probs,
            'total_reward': torch.tensor(self.total_reward, device=device),
            'sequence_length': torch.tensor(len(self.steps), device=device)
        }
        
        if all(step.value is not None for step in self.steps):
            values = torch.tensor([step.value for step in self.steps], 
                                 device=device, dtype=torch.float32)
            tensor_dict['values'] = values
            
        return tensor_dict
    
    def compute_returns(self, gamma: float = 1.0) -> List[float]:
        """Compute discounted returns for each step."""
        returns = []
        cumulative_return = 0.0
        
        # Compute returns backwards
        for step in reversed(self.steps):
            cumulative_return = step.reward + gamma * cumulative_return
            returns.insert(0, cumulative_return)
            
        return returns
    
    def get_statistics(self) -> Dict[str, float]:
        """Compute trajectory statistics."""
        if not self.steps:
            return {}
            
        rewards = [step.reward for step in self.steps]
        return {
            'total_reward': self.total_reward,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': max(rewards),
            'min_reward': min(rewards),
            'steps': len(self.steps),
            'terminal': self.terminal
        }

class TrajectoryBatch:
    """Batch of trajectories for parallel processing."""
    
    def __init__(self, trajectories: List[Trajectory]):
        self.trajectories = trajectories
        self.batch_size = len(trajectories)
        
    def get_batch_statistics(self) -> Dict[str, float]:
        """Compute statistics across all trajectories in batch."""
        if not self.trajectories:
            return {}
            
        stats = {
            'batch_size': self.batch_size,
            'mean_total_reward': np.mean([t.total_reward for t in self.trajectories]),
            'std_total_reward': np.std([t.total_reward for t in self.trajectories]),
            'mean_trajectory_length': np.mean([len(t.steps) for t in self.trajectories]),
            'completion_rate': np.mean([t.terminal for t in self.trajectories]),
        }
        return stats
        
    def filter_completed(self) -> 'TrajectoryBatch':
        """Filter out completed trajectories."""
        completed = [t for t in self.trajectories if t.terminal]
        return TrajectoryBatch(completed)
        
    def sample_minibatch(self, batch_size: int) -> 'TrajectoryBatch':
        """Sample a minibatch of trajectories."""
        if batch_size >= self.batch_size:
            return self
            
        indices = np.random.choice(self.batch_size, batch_size, replace=False)
        sampled_trajectories = [self.trajectories[i] for i in indices]
        return TrajectoryBatch(sampled_trajectories)

class TrajectoryCollector:
    """Collects and manages trajectories during training."""
    
    def __init__(self, max_trajectories: int = 10000):
        self.max_trajectories = max_trajectories
        self.active_trajectories: Dict[str, Trajectory] = {}
        self.completed_trajectories: List[Trajectory] = []
        
    def start_trajectory(self, environment: str, initial_state: str, 
                        trajectory_id: Optional[str] = None) -> str:
        """Start a new trajectory."""
        if trajectory_id is None:
            import uuid
            trajectory_id = str(uuid.uuid4())[:8]
            
        trajectory = Trajectory(
            trajectory_id=trajectory_id,
            environment=environment,
            initial_state=initial_state
        )
        
        self.active_trajectories[trajectory_id] = trajectory
        return trajectory_id
        
    def add_step(self, trajectory_id: str, step: TrajectoryStep):
        """Add a step to an active trajectory."""
        if trajectory_id not in self.active_trajectories:
            raise ValueError(f"Unknown trajectory: {trajectory_id}")
            
        trajectory = self.active_trajectories[trajectory_id]
        trajectory.add_step(step)
        
        # Move to completed if terminal
        if step.done:
            self.completed_trajectories.append(trajectory)
            del self.active_trajectories[trajectory_id]
            
            # Maintain size limit
            if len(self.completed_trajectories) > self.max_trajectories:
                self.completed_trajectories = self.completed_trajectories[-self.max_trajectories:]
                
    def get_active_trajectories(self) -> List[Trajectory]:
        """Get all active trajectories."""
        return list(self.active_trajectories.values())
        
    def get_completed_trajectories(self) -> List[Trajectory]:
        """Get all completed trajectories."""
        return self.completed_trajectories.copy()
        
    def clear_completed(self):
        """Clear completed trajectories."""
        self.completed_trajectories.clear()
        
    def get_trajectory(self, trajectory_id: str) -> Trajectory:
        """Get trajectory by ID."""
        if trajectory_id in self.active_trajectories:
            return self.active_trajectories[trajectory_id]
        else:
            # Search in completed trajectories
            for trajectory in self.completed_trajectories:
                if trajectory.trajectory_id == trajectory_id:
                    return trajectory
            raise ValueError(f"Trajectory not found: {trajectory_id}")