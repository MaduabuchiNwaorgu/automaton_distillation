import torch
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity=1000000, input_shape=(84, 84), history_length=4):
        self.capacity = capacity
        self.input_shape = input_shape
        self.history_length = history_length
        self.count = 0
        self.current = 0
        
        # Storage
        self.actions = np.zeros(self.capacity, dtype=np.int32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dfa_states = np.zeros(self.capacity, dtype=np.int32)
        self.frames = np.zeros((self.capacity, input_shape[0], input_shape[1]), dtype=np.uint8)
        self.terminals = np.zeros(self.capacity, dtype=np.bool_)
        
    def add_experience(self, action, frame, reward, terminal, dfa_state):
        """Add experience to buffer"""
        if frame.shape != (*self.input_shape, 1):
            frame = frame.squeeze()
        
        self.actions[self.current] = action
        self.frames[self.current] = frame
        self.rewards[self.current] = np.sign(reward)  # Clip rewards
        self.dfa_states[self.current] = dfa_state
        self.terminals[self.current] = terminal
        
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.capacity
    
    def get_minibatch(self, batch_size=32):
        """Sample a minibatch of experiences"""
        if self.count < self.history_length:
            raise ValueError('Not enough experiences to sample')
        
        indices = []
        for _ in range(batch_size):
            while True:
                index = random.randint(self.history_length, self.count - 1)
                
                # Check if all frames are from same episode
                if index >= self.current and index - self.history_length <= self.current:
                    continue
                if self.terminals[index - self.history_length:index].any():
                    continue
                break
            indices.append(index)
        
        # Prepare batch
        states = []
        next_states = []
        dfa_states_batch = []
        
        for idx in indices:
            # Stack frames for state
            state_frames = self.frames[idx - self.history_length:idx]
            next_state_frames = self.frames[idx - self.history_length + 1:idx + 1]
            
            states.append(np.transpose(state_frames, (1, 2, 0)))
            next_states.append(np.transpose(next_state_frames, (1, 2, 0)))
            dfa_states_batch.append(self.dfa_states[idx])
        
        states = np.array(states)
        next_states = np.array(next_states)
        
        # Transpose to (batch, channels, height, width) for PyTorch
        states = np.transpose(states, (0, 3, 1, 2))
        next_states = np.transpose(next_states, (0, 3, 1, 2))
        
        return (states, self.actions[indices], self.rewards[indices], 
                next_states, self.terminals[indices], dfa_states_batch)
    
    def num_filled_approx(self):
        """Get approximate number of filled experiences"""
        return self.count