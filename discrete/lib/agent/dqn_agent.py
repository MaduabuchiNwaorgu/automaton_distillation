import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random


class DuelingDQN(nn.Module):

    def __init__(self,n_actions,input_shape=(84,84),history_length=4):
        super(DuelingDQN,self).__init__()
        self.input_shape = input_shape
        self.history_length = history_length
        self.n_actions = n_actions


        # CNN Layers
        self.conv1 = nn.Conv2d(history_length,32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(32,64, kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(64,64, kernel_size=4,stride=2)
        self.conv4 = nn.Conv2d(64,1024, kernel_size=4,stride=2)

        # Dueling architecture - split the 1024 features

        self.value_stream = nn.Linear(512,1)
        self.advantage_stream = nn.Linear(512,n_actions)

        # intitalize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.variance_scaling_(module.weight, scale=2.0, mode='fan_in', distribution='uniform')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # Normalize input
        x = x / 255.0
        
        # CNN forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Flatten and split for dueling architecture
        x = x.view(x.size(0), -1)
        value_stream = x[:, :512]
        advantage_stream = x[:, 512:]
        
        values = self.value_stream(value_stream)
        advantages = self.advantage_stream(advantage_stream)
        
        # Combine value and advantage streams
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
    
class DQNAgent:
    def __init__(self, config, n_actions, input_shape=(84, 84), history_length=4):
        self.device = config.device
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.history_length = history_length
        self.batch_size = getattr(config, 'batch_size', 32)

        # Epsilon parameters
        self.eps_initial = getattr(config, 'eps_initial', 1.0)
        self.eps_final = getattr(config, 'eps_final', 0.1)
        self.eps_final_frame = getattr(config, 'eps_final_frame', 0.01)
        self.eps_evaluation = getattr(config, 'eps_evaluation', 0.0)
        self.eps_annealing_frames = getattr(config, 'eps_annealing_frames', 150000)
        self.replay_buffer_start_size = getattr(config, 'replay_buffer_start_size', 8000)
        self.max_frames = getattr(config, 'max_training_steps', 1000000)

        # Calculate epsilon decay slopes
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope * self.replay_buffer_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (
                self.max_frames - self.eps_annealing_frames - self.replay_buffer_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2 * self.max_frames

        # Networks
        self.dqn = DuelingDQN(n_actions, input_shape, history_length).to(self.device)
        self.target_dqn = DuelingDQN(n_actions, input_shape, history_length).to(self.device)

         # Copy weights to target network
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=config.learning_rate)


    def calc_epsilon(self, frame_number, evaluation=False):
        """Calculate epsilon for epsilon-greedy exploration"""
        if evaluation:
            return self.eps_evaluation
        elif frame_number < self.replay_buffer_start_size:
            return self.eps_initial
        elif frame_number < self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope * frame_number + self.intercept
        else:
            return self.slope_2 * frame_number + self.intercept_2
        
    def calc_q_values_batch(self, states, aut_states):
        """Calculate Q-values for batch of states"""
        if len(states.shape) == 3:
            states = states.unsqueeze(0)
        elif len(states.shape) == 4 and states.shape[0] == 1:
            pass  # Already has batch dimension

        with torch.no_grad():
            return self.dqn(states)
        
    def calc_v_values_batch(self, states, aut_states):
        """Calculate V-values (max Q-values) for batch of states"""
        q_values = self.calc_q_values_batch(states, aut_states)
        return q_values.max(dim=1)[0]
    
    def get_action(self, frame_number, state, evaluation=False):
        """Get action using epsilon-greedy policy"""
        epsilon = self.calc_epsilon(frame_number, evaluation)
        
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            if len(state.shape) == 3:
                state = state.unsqueeze(0)
            
            with torch.no_grad():
                q_values = self.dqn(state.to(self.device))
                return q_values.argmax().item()
            
    def update_target_network(self):
        """Update target network weights"""
        self.target_dqn.load_state_dict(self.dqn.state_dict())
    
    def learn(self, batch_data, agents_dict=None, gamma=0.99):
        """Train the DQN with a batch of experiences"""
        states, actions, rewards, next_states, terminals, dfa_states = batch_data
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        terminals = torch.BoolTensor(terminals).to(self.device)
        
        # Current Q-values
        current_q_values = self.dqn(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values (Double DQN)
        with torch.no_grad():
            next_actions = self.dqn(next_states).argmax(dim=1)
            next_q_values = self.target_dqn(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (gamma * next_q_values * (~terminals).unsqueeze(1))
            #1 - terminals
        
        # Loss calculation
        loss = F.huber_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()