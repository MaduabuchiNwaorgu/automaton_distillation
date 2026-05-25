import torch
import numpy as np
import cv2
import gym
import random

def process_frame(frame, shape=(84, 84)):
    """Process frame to grayscale and resize"""
    frame = frame.astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34:34 + 160, :160]
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))
    return frame

class GameWrapper:
    def __init__(self, env_name, no_op_steps=10, history_length=4, device='cpu'):
        self.env = gym.make(env_name)
        self.no_op_steps = no_op_steps
        self.history_length = history_length
        self.device = device
        
        self.state = None
        self.last_lives = 0
        
    def reset(self, evaluation=False):
        """Reset environment"""
        reset_result = self.env.reset()
        
        if isinstance(reset_result, tuple):
            self.frame, _ = reset_result
        else:
            self.frame = reset_result
            
        self.last_lives = 0
        
        # Random no-op steps for evaluation
        if evaluation:
            for _ in range(random.randint(0, self.no_op_steps)):
                self.env.step(1)
        
        processed = process_frame(self.frame)
        self.state = np.repeat(processed, self.history_length, axis=2)
        
        # Convert to PyTorch tensor
        state_tensor = torch.FloatTensor(self.state).permute(2, 0, 1).to(self.device)
        return state_tensor
    
    def step(self, action):
        """Take environment step"""
        step_result = self.env.step(action)
        
        if len(step_result) == 5:
            new_frame, reward, terminal, truncated, info = step_result
            terminal = terminal or truncated
        else:
            new_frame, reward, terminal, info = step_result
        
        # Life tracking
        if 'ale.lives' in info:
            current_lives = info['ale.lives']
        elif 'lives' in info:
            current_lives = info['lives']
        else:
            current_lives = self.last_lives
            
        life_lost = current_lives < self.last_lives
        self.last_lives = current_lives
        
        # Process frame
        processed_frame = process_frame(new_frame)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=2)
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(self.state).permute(2, 0, 1).to(self.device)
        
        return state_tensor, reward, terminal, life_lost, new_frame
    
    @property
    def action_space_n(self):
        return self.env.action_space.n