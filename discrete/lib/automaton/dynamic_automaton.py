import torch
import json
import numpy as np
from collections import defaultdict

# Import SYNTH functions (copy from deepsynth)
try:
    from synth.synth_wrapper import dfa_init, dfa_update, get_next_state
    SYNTH_AVAILABLE = True
except ImportError:
    print("Warning: SYNTH not available. Using mock implementation.")
    SYNTH_AVAILABLE = False
    
    # Mock implementations for development
    def dfa_init():
        return 2, {}, {'event_uniq': []}, {}
    
    def dfa_update(*args):
        return args[1], [], [], [], [], args[5], args[6]
    
    def get_next_state(trace, events, dfa):
        return 1

class DynamicAutomaton:
    def __init__(self, update_frequency=1000, min_frames_before_update=10000):
        self.update_frequency = update_frequency
        self.min_frames_before_update = min_frames_before_update
        
        # Initialize SYNTH framework
        if SYNTH_AVAILABLE:
            self.num_states_synth, self.var, self.input_dict, self.hyperparams = dfa_init()
        else:
            self.num_states_synth, self.var, self.input_dict, self.hyperparams = 2, {}, {'event_uniq': []}, {}
            
        self.processed_dfa = []
        self.dfa_model = []
        self.nfa_model = []
        self.model_gen = []
        
        # Episode tracking
        self.episode_traces = []
        self.current_episode_trace = ['start']
        self.synth_iter_num = 0
        
        # Current automaton structure
        self.states = [0, 1]
        self.transitions = {}
        
        # Frame tracking
        self.frame_number = 0
        self.last_update_frame = 0
        
    def step_batch(self, current_states, aps):
        """Step automaton for batch of experiences"""
        self.frame_number += len(current_states)
        
        next_states = []
        
        for i, (current_state, ap) in enumerate(zip(current_states, aps)):
            # Convert AP to object name for trace
            ap_name = self._ap_to_name(ap.item())
            
            # Add to episode trace (only non-'none' objects)
            if ap_name != 'none':
                self.current_episode_trace.append(ap_name)
            
            # Get next state from current automaton
            if (self.frame_number > self.min_frames_before_update and 
                self.processed_dfa and SYNTH_AVAILABLE):
                try:
                    next_state = get_next_state(self.current_episode_trace, 
                                              self.input_dict.get('event_uniq', []), 
                                              self.processed_dfa)
                    
                    if next_state == -1 or next_state == [] or next_state is None:
                        next_state = current_state.item()
                    elif isinstance(next_state, list) and len(next_state) > 0:
                        next_state = next_state[0]
                    else:
                        next_state = int(next_state) if next_state is not None else current_state.item()
                        
                except Exception as e:
                    print(f"Warning: SYNTH step failed: {e}")
                    next_state = current_state.item()
            else:
                next_state = current_state.item()
                
            next_states.append(next_state)
            
        # Check if automaton update is needed
        should_update = (
            self.frame_number - self.last_update_frame >= self.update_frequency and
            self.frame_number > self.min_frames_before_update
        )
        
        if should_update:
            self._update_automaton()
            
        return torch.tensor(next_states, dtype=torch.long)
    
    def _update_automaton(self):
        """Update automaton structure using SYNTH"""
        if not SYNTH_AVAILABLE:
            return
            
        try:
            # Prepare trace for SYNTH
            self.episode_traces.append(self.current_episode_trace.copy())
            
            # Combine all episode traces
            trace = []
            for episode_trace in self.episode_traces:
                trace.extend(episode_trace)
            trace.append('start')  # End marker
            
            # Update automaton using SYNTH
            start_time = 0  # Placeholder
            (self.num_states_synth, self.processed_dfa, self.dfa_model, 
             self.nfa_model, self.model_gen, self.var, 
             self.input_dict) = dfa_update(
                trace, self.num_states_synth, self.dfa_model, self.nfa_model,
                self.model_gen, self.var, self.input_dict, self.hyperparams,
                start_time, self.synth_iter_num)
            
            # Extract states from DFA transitions
            if self.processed_dfa:
                self.states = list(set([trans[0] for trans in self.processed_dfa] +
                                     [trans[2] for trans in self.processed_dfa]))
            
            self.synth_iter_num += 1
            self.last_update_frame = self.frame_number
            
            # Keep only recent traces
            self.episode_traces = [self.current_episode_trace.copy()]
            
            print(f"Automaton updated at frame {self.frame_number}: {len(self.states)} states")
            
        except Exception as e:
            print(f"Warning: Automaton update failed: {e}")
    
    def reset_episode(self):
        """Reset for new episode"""
        if len(self.current_episode_trace) > 1:  # Only add if there were actual events
            self.episode_traces.append(self.current_episode_trace.copy())
        self.current_episode_trace = ['start']
    
    def _ap_to_name(self, ap_index):
        """Convert AP index to object name"""
        ap_names = ['start', 'middle_ladder', 'rope', 'right_ladder', 
                   'left_ladder', 'key', 'door', 'none']
        return ap_names[ap_index] if 0 <= ap_index < len(ap_names) else 'none'
    
    @property
    def num_states(self):
        return max(len(self.states), 2)  # At least 2 states
    
    def get_current_trace(self):
        """Get current episode trace"""
        return self.current_episode_trace.copy()
    
    def get_dfa_states(self):
        """Get all DFA states"""
        return self.states.copy()
    
    def save_structure(self, filepath):
        """Save learned automaton structure"""
        structure = {
            'states': self.states,
            'processed_dfa': self.processed_dfa,
            'transitions': self.transitions,
            'input_dict': self.input_dict,
            'episode_traces': self.episode_traces,
            'synth_iter_num': self.synth_iter_num
        }
        with open(filepath, 'w') as f:
            json.dump(structure, f, indent=2)
            
    def load_structure(self, filepath):
        """Load automaton structure"""
        with open(filepath, 'r') as f:
            structure = json.load(f)
        self.states = structure.get('states', [0, 1])
        self.processed_dfa = structure.get('processed_dfa', [])
        self.transitions = structure.get('transitions', {})
        self.input_dict = structure.get('input_dict', {'event_uniq': []})
        self.episode_traces = structure.get('episode_traces', [])
        self.synth_iter_num = structure.get('synth_iter_num', 0)