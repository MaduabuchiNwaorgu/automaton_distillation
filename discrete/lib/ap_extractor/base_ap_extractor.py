import torch
from abc import ABC, abstractmethod

class APExtractor(ABC):
    """Base class for extracting atomic propositions from environment states"""
    
    def __init__(self, device='cpu'):
        self.device = device
    
    @abstractmethod
    def extract_aps_batch(self, states, infos=None):
        """Extract atomic propositions from batch of states
        
        Args:
            states: Batch of environment states (tensor or numpy array)
            infos: Optional batch of info dictionaries from environment
            
        Returns:
            tuple: (aps_tensor, rewards_tensor) where:
                - aps_tensor: Long tensor of AP indices for each state
                - rewards_tensor: Float tensor of intrinsic rewards for each state
        """
        pass
    
    @abstractmethod
    def num_transitions(self):
        """Return the number of possible atomic propositions"""
        pass
    
    @abstractmethod
    def reset_episode(self):
        """Reset any episode-specific tracking"""
        pass
    
    def extract_aps_single(self, state, info=None):
        """Extract APs from a single state (convenience method)"""
        if isinstance(state, torch.Tensor):
            state = state.unsqueeze(0)  # Add batch dimension
        else:
            state = torch.tensor(state).unsqueeze(0)
            
        infos = [info] if info is not None else None
        aps, rewards = self.extract_aps_batch(state, infos)
        return aps[0], rewards[0]
    
    def get_ap_name(self, ap_index):
        """Get human-readable name for AP index (optional override)"""
        return f"ap_{ap_index}"
    
    def to(self, device):
        """Move extractor to specified device"""
        self.device = device
        return self