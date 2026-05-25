import torch
import torch.nn as nn
from collections import defaultdict
from discrete.lib.agent.dqn_agent import DQNAgent
from discrete.lib.replay_buffer.replay_buffer import DiscreteReplayBuffer

class DynamicMultiAgent:
    def __init__(self, config, automaton, n_actions, input_shape=(84, 84)):
        self.config = config
        self.automaton = automaton
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.device = config.device
        
        # Dictionary of agents for each automaton state
        self.state_agents = {}
        self.state_buffers = {}
        self.frame_numbers = {}
        
        # Initialize with starting state
        self._create_agent_for_state(1)
        
    def _create_agent_for_state(self, state_id):
        """Create new agent and buffer for automaton state"""
        if state_id not in self.state_agents:
            print(f"Creating new agent for automaton state {state_id}")
            
            # Create agent
            agent = DQNAgent(
                config=self.config,
                n_actions=self.n_actions,
                input_shape=self.input_shape
            )
            self.state_agents[state_id] = agent
            self.frame_numbers[state_id] = 0
            
            # Create replay buffer
            buffer = DiscreteReplayBuffer(
                capacity=self.config.rollout_buffer_config.capacity,
                input_shape=self.input_shape
            )
            self.state_buffers[state_id] = buffer
    
    def calc_q_values_batch(self, states, aut_states):
        """Calculate Q-values using appropriate agent for each automaton state"""
        if len(states.shape) == 3:
            states = states.unsqueeze(0)
        
        batch_size = states.shape[0]
        q_values = torch.zeros(batch_size, self.n_actions, device=self.device)
        
        # Update automaton structure
        self.update_automaton_structure()
        
        # Group by automaton state
        state_groups = defaultdict(list)
        for i, aut_state in enumerate(aut_states):
            state_groups[aut_state.item()].append(i)
        
        for aut_state, indices in state_groups.items():
            if aut_state not in self.state_agents:
                self._create_agent_for_state(aut_state)
            
            agent = self.state_agents[aut_state]
            state_batch = states[indices]
            aut_state_batch = aut_states[indices]
            
            try:
                q_vals = agent.calc_q_values_batch(state_batch, aut_state_batch)
                q_values[indices] = q_vals
            except Exception as e:
                print(f"Warning: Q-value calculation failed for state {aut_state}: {e}")
                q_values[indices] = torch.randn(len(indices), self.n_actions, device=self.device)
        
        return q_values
    
    def calc_v_values_batch(self, states, aut_states):
        """Calculate V-values using Q-values"""
        q_values = self.calc_q_values_batch(states, aut_states)
        return q_values.max(dim=1)[0]
    
    def update_automaton_structure(self):
        """Create agents for new automaton states"""
        if hasattr(self.automaton, 'get_dfa_states'):
            current_states = set(self.automaton.get_dfa_states())
        else:
            current_states = set(range(self.automaton.num_states))
        
        existing_states = set(self.state_agents.keys())
        
        new_states = current_states - existing_states
        for state_id in new_states:
            self._create_agent_for_state(state_id)
    
    def get_active_agent(self, aut_state):
        """Get agent for specific automaton state"""
        if aut_state not in self.state_agents:
            self._create_agent_for_state(aut_state)
        return self.state_agents[aut_state]
    
    def add_experience(self, action, frame, reward, terminal, dfa_state):
        """Add experience to appropriate agent's buffer"""
        if dfa_state not in self.state_agents:
            self._create_agent_for_state(dfa_state)
        
        buffer = self.state_buffers[dfa_state]
        buffer.add_experience(action, frame, reward, terminal, dfa_state)
    
    def learn_all_agents(self, gamma=0.99):
        """Train all state-specific agents"""
        self.update_automaton_structure()
        
        total_loss = 0
        agents_trained = 0
        
        for state_id, agent in self.state_agents.items():
            buffer = self.state_buffers[state_id]
            
            if buffer.num_filled_approx() > 100:  # Minimum samples
                try:
                    batch_data = buffer.get_minibatch(batch_size=self.config.batch_size)
                    loss = agent.learn(batch_data, agents_dict=self.state_agents, gamma=gamma)
                    total_loss += loss
                    agents_trained += 1
                    
                    self.frame_numbers[state_id] += 1
                    
                except Exception as e:
                    print(f"Warning: Training failed for agent {state_id}: {e}")
        
        return total_loss / max(agents_trained, 1)
    
    def update_target_networks(self):
        """Update target networks for all agents"""
        for agent in self.state_agents.values():
            agent.update_target_network()
    
    def get_action(self, state, aut_state, evaluation=False):
        """Get action from appropriate agent"""
        agent = self.get_active_agent(aut_state)
        frame_num = self.frame_numbers.get(aut_state, 0)
        return agent.get_action(frame_num, state, evaluation)