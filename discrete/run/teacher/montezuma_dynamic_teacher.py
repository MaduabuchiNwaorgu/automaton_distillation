import torch
import numpy as np
import time
import os
from tqdm import tqdm

# Import PyTorch components
from discrete.lib.env.game_wrapper import GameWrapper
from discrete.lib.ap_extractor.montezuma_ap_extractor import MontezumaAPExtractor
from discrete.lib.automaton.dynamic_automaton import DynamicAutomaton
from discrete.lib.agent.dynamic_multi_agent import DynamicMultiAgent
from discrete.lib.config import Configuration, RolloutBufferConfig, EnvConfig
from discrete.lib.constr_q_automaton import construct_q_automaton_from_multi_agent


# SYNTH imports (copy from deepsynth)
try:
    from synth.synth_wrapper import dfa_init, dfa_update, get_next_state
    SYNTH_AVAILABLE = True
except ImportError:
    print("Warning: SYNTH not available")
    SYNTH_AVAILABLE = False

def create_torch_montezuma_config():
    """Create PyTorch configuration for Montezuma's Revenge"""
    
    env_config = EnvConfig(
        env_name="MontezumaRevenge-v4"
        # action_space_size=18,
        # observation_space_shape=(84, 84, 4),
        # max_episode_steps=4500
    )
    
    rollout_buffer_config = RolloutBufferConfig(
        capacity=100000,
        min_size_before_training=8000
    )
    
    config = Configuration(
        # Environment
        env_config=env_config,
        rollout_buffer_config=rollout_buffer_config,
        
        # Training
        max_training_steps=200000,
        learning_rate=0.00001,
        batch_size=32,
        target_update_freq=1000,
        
        # Epsilon parameters
        eps_initial=1.0,
        eps_final=0.1,
        eps_final_frame=0.01,
        eps_evaluation=0.0,
        eps_annealing_frames=150000,
        replay_buffer_start_size=8000,
        
        # Dynamic automaton
        automaton_update_frequency=1000,
        min_frames_before_automaton_update=10000,
        
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
        run_name="montezuma_dynamic_teacher"
    )
    
    return config

def run_torch_dynamic_training():
    """Main training loop for PyTorch dynamic automaton learning"""
    
    config = create_torch_montezuma_config()
    print(f"Starting PyTorch Montezuma training on {config.device}")
    
    # Create directories
    os.makedirs("learned_automatons", exist_ok=True)
    os.makedirs("automaton_q", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Initialize components
    game_wrapper = GameWrapper(
        env_name=config.env_config.env_name,
        device=config.device
    )
    
    ap_extractor = MontezumaAPExtractor(
        intrinsic_reward_weight=0.1,
        device=config.device
    )
    
    # Initialize SYNTH automaton
    if SYNTH_AVAILABLE:
        num_states, var, input_dict, hyperparams = dfa_init()
        processed_dfa = []
        dfa_model = []
        nfa_model = []
        model_gen = []
        synth_iter_num = 0
        set_of_episode_traces = []
    else:
        print("Running without SYNTH - using simple state tracking")
    
    automaton = DynamicAutomaton(
        update_frequency=config.automaton_update_frequency,
        min_frames_before_update=config.min_frames_before_automaton_update
    )
    
    # Multi-agent system
    multi_agent = DynamicMultiAgent(
        config=config,
        automaton=automaton,
        n_actions=game_wrapper.action_space_n,
        input_shape=(84, 84)
    )
    
    # Training variables
    frame_number = 0
    episode_rewards = []
    loss_list = []
    current_dfa_state = 1
    
    print("Starting training loop...")
    
    try:
        while frame_number < config.max_training_steps:
            # Reset environment
            state = game_wrapper.reset()
            terminal = False
            episode_trace = ['start']
            episode_detected_objects = []
            episode_reward_sum = 0
            current_dfa_state = 1
            ap_extractor.reset_episode()
            
            while not terminal and frame_number < config.max_training_steps:
                # Get action from active agent
                action = multi_agent.get_action(state, current_dfa_state, evaluation=False)
                
                # Take environment step
                next_state, reward, terminal, life_lost, raw_frame = game_wrapper.step(action)
                frame_number += 1
                
                # Extract atomic propositions and add intrinsic rewards
                aps, intrinsic_rewards = ap_extractor.extract_aps_batch(next_state)
                ap_index = aps[0].item() if len(aps) > 0 else 7  # 'none'
                intrinsic_reward = intrinsic_rewards[0].item() if len(intrinsic_rewards) > 0 else 0
                
                # Add intrinsic reward
                total_reward = reward + intrinsic_reward
                episode_reward_sum += total_reward
                
                # Update episode trace
                ap_name = ap_extractor.get_ap_name(ap_index)
                if ap_name != 'none':
                    episode_trace.append(ap_name)
                
                # Determine next DFA state
                next_dfa_state = current_dfa_state  # Default
                
                if frame_number >= config.min_frames_before_automaton_update and SYNTH_AVAILABLE:
                    # Update automaton if needed
                    if (frame_number % config.automaton_update_frequency == 0 or
                        get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa) in [-1, []]):
                        
                        # Update SYNTH automaton
                        trace = []
                        set_of_episode_traces.append(episode_trace)
                        for x in set_of_episode_traces:
                            trace = trace + x
                        trace = trace + ['start']
                        
                        start_time = time.time()
                        num_states, processed_dfa, dfa_model, nfa_model, model_gen, var, input_dict = dfa_update(
                            trace, num_states, dfa_model, nfa_model, model_gen, var, input_dict, 
                            hyperparams, start_time, synth_iter_num)
                        
                        dfa_states = list(set([dfa_transitions[0] for dfa_transitions in processed_dfa] +
                                            [dfa_transitions[2] for dfa_transitions in processed_dfa]))
                        synth_iter_num += 1
                        set_of_episode_traces = [episode_trace]
                        
                        # Update multi-agent structure
                        automaton.states = dfa_states
                        multi_agent.update_automaton_structure()
                    
                    # Get next DFA state
                    next_dfa_state = get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa)
                    if next_dfa_state in [-1, []] or next_dfa_state is None:
                        next_dfa_state = current_dfa_state
                
                # Add experience to replay buffer
                if frame_number > config.replay_buffer_start_size:
                    processed_frame = raw_frame[34:34+160, :160]  # Crop frame
                    processed_frame = torch.FloatTensor(processed_frame).unsqueeze(0)  # Add channel dim
                    
                    multi_agent.add_experience(
                        action=action,
                        frame=processed_frame,
                        reward=total_reward,
                        terminal=life_lost,
                        dfa_state=next_dfa_state
                    )
                
                # Training
                if frame_number > config.replay_buffer_start_size and frame_number % 4 == 0:
                    loss = multi_agent.learn_all_agents(gamma=0.99)
                    loss_list.append(loss)
                
                # Update target networks
                if frame_number % config.target_update_freq == 0:
                    multi_agent.update_target_networks()
                
                # Update states
                state = next_state
                current_dfa_state = next_dfa_state
                
                # Reset on life lost
                if life_lost:
                    episode_detected_objects = []
                    current_dfa_state = 1
                    if SYNTH_AVAILABLE:
                        set_of_episode_traces.append(episode_trace)
                    episode_trace = ['start']
                    ap_extractor.reset_episode()
                
                # Logging
                if frame_number % 1000 == 0:
                    avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                    avg_loss = np.mean(loss_list[-100:]) if loss_list else 0
                    print(f"Frame {frame_number}: Avg Reward={avg_reward:.2f}, "
                          f"Avg Loss={avg_loss:.4f}, DFA States={len(multi_agent.state_agents)}")
            
            # End of episode
            episode_rewards.append(episode_reward_sum)
            if SYNTH_AVAILABLE:
                set_of_episode_traces.append(episode_trace)
            
            # Save periodically
            if len(episode_rewards) % 100 == 0:
                print(f"Episode {len(episode_rewards)}: Reward={episode_reward_sum:.2f}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final models
    print("Saving models and automaton...")
    
    # Save automaton structure
    if SYNTH_AVAILABLE:
        automaton_data = {
            'states': list(multi_agent.state_agents.keys()),
            'processed_dfa': processed_dfa,
            'input_dict': input_dict,
            'episode_traces': set_of_episode_traces
        }
        
        import json
        with open(f"learned_automatons/{config.run_name}_automaton.json", 'w') as f:
            json.dump(automaton_data, f, indent=2)
    
    # Save agents
    for state_id, agent in multi_agent.state_agents.items():
        torch.save({
            'dqn_state_dict': agent.dqn.state_dict(),
            'target_dqn_state_dict': agent.target_dqn.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
        }, f"models/{config.run_name}_agent_{state_id}.pth")
    
    print(f"Training completed! Saved {len(multi_agent.state_agents)} agents")
    print(f"Final automaton has {len(multi_agent.state_agents)} states")

    print("Constructing Q-automaton for transfer learning...")
    try:
        construct_q_automaton_from_multi_agent(
            multi_agent=multi_agent,
            ap_extractor=ap_extractor,
            automaton=automaton,
            device=config.device,
            run_name=config.run_name
        )
        print("Q-automaton construction completed!")
    except Exception as e:
        print(f"Warning: Q-automaton construction failed: {e}")

if __name__ == "__main__":
    run_torch_dynamic_training()