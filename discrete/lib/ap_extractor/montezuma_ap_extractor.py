import torch
import numpy as np
from discrete.lib.ap_extractor.base_ap_extractor import APExtractor

def subarray_detector(big_array, small_array):
    """Check if small_array exists in big_array"""
    def check(a, b, upper_left):
        ul_row, ul_col = upper_left
        b_rows, b_cols = b.shape
        a_slice = a[ul_row: ul_row + b_rows, ul_col: ul_col + b_cols]
        if a_slice.shape != b.shape:
            return False
        return (a_slice == b).all()

    upper_left = np.argwhere(big_array == small_array[0, 0])
    for ul in upper_left:
        if check(big_array, small_array, ul):
            return True
    return False

def intrinsic_reward(new_obj_set, old_obj_set):
    """Calculate intrinsic reward for discovering new objects"""
    new_detected_obj = list(set(new_obj_set) - set(old_obj_set))
    return 1 if new_detected_obj else 0

class MontezumaAPExtractor(APExtractor):
    def __init__(self, intrinsic_reward_weight=0.1, device='cpu'):
        self.agent_unique = np.array([[478, 478, 478], [478, 478, 478], [344, 344, 344], [478, 478, 478]])
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.device = device
        
        self.object_to_ap = {
            'start': 0,
            'middle_ladder': 1,
            'rope': 2,
            'right_ladder': 3,
            'left_ladder': 4,
            'key': 5,
            'door': 6,
            'none': 7
        }
        self.ap_to_object = {v: k for k, v in self.object_to_ap.items()}
        
        # Episode tracking
        self.episode_detected_objects = []
        self.old_obj_set = []
        
    def extract_aps_batch(self, states, infos=None):
        """Extract atomic propositions from batch of states"""
        if isinstance(states, torch.Tensor):
            states_np = states.detach().cpu().numpy()
        else:
            states_np = states
            
        aps = []
        intrinsic_rewards = []
        
        # Handle batch dimension
        if len(states_np.shape) == 4:
            batch_size = states_np.shape[0]
        else:
            batch_size = 1
            states_np = states_np[np.newaxis, ...]
        
        for i in range(batch_size):
            state = states_np[i]
            
            # Convert from (C, H, W) to (H, W, C) if needed
            if state.shape[0] == 4:  # Channels first
                state = np.transpose(state, (1, 2, 0))
            
            # Sum across channels to get unique color map
            observation = np.sum(state, axis=2)
            
            # Detect objects
            detected_objects = self._detect_objects(observation)
            
            # Calculate intrinsic reward
            old_obj_set = self.old_obj_set.copy()
            new_obj_set = list(np.unique(detected_objects))
            intr_reward = intrinsic_reward(new_obj_set, old_obj_set) * self.intrinsic_reward_weight
            intrinsic_rewards.append(intr_reward)
            
            # Update tracking
            self.old_obj_set = new_obj_set
            self.episode_detected_objects = detected_objects
            
            # Get primary AP
            primary_object = self._get_primary_object(detected_objects)
            ap_index = self.object_to_ap.get(primary_object, self.object_to_ap['none'])
            aps.append(ap_index)
        
        aps_tensor = torch.tensor(aps, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(intrinsic_rewards, dtype=torch.float32, device=self.device)
        
        return aps_tensor, rewards_tensor
    
    def _detect_objects(self, observation):
        """Detect all objects in observation"""
        detected_objects = []
        
        # Object detection using template matching
        if subarray_detector(observation[93:134, 76:83], self.agent_unique):
            detected_objects.append('middle_ladder')
        if subarray_detector(observation[96:134, 110:115], self.agent_unique):
            detected_objects.append('rope')
        if subarray_detector(observation[136:179, 132:139], self.agent_unique):
            detected_objects.append('right_ladder')
        if subarray_detector(observation[136:179, 20:27], self.agent_unique):
            detected_objects.append('left_ladder')
        if subarray_detector(observation[99:106, 13:19], self.agent_unique):
            detected_objects.append('key')
        if subarray_detector(observation[50:92, 20:24], self.agent_unique):
            detected_objects.append('door')
        if subarray_detector(observation[50:92, 136:140], self.agent_unique):
            detected_objects.append('door')
        
        return detected_objects if detected_objects else ['none']
    
    def _get_primary_object(self, detected_objects):
        """Get the most important object from detected objects"""
        priority = ['key', 'door', 'left_ladder', 'right_ladder', 'middle_ladder', 'rope', 'none']
        
        for obj in priority:
            if obj in detected_objects:
                return obj
        return 'none'
    
    def reset_episode(self):
        """Reset for new episode"""
        self.episode_detected_objects = []
        self.old_obj_set = []
    
    def num_transitions(self):
        """Return number of possible atomic propositions"""
        return len(self.object_to_ap)
    
    def get_ap_name(self, ap_index):
        """Get human-readable name for AP index"""
        return self.ap_to_object.get(ap_index, 'unknown')