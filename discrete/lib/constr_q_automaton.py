import torch
import json
import os
import numpy as np
from tqdm import tqdm

def construct_q_automaton_from_multi_agent(multi_agent, ap_extractor, automaton, device, run_name):
    """Construct Q-automaton from PyTorch multi-agent system"""
    
    print(f"Constructing Q-automaton for {len(multi_agent.state_agents)} agents")
    
    # Combined Q-automaton data
    combined_data = {
        "state_agent_mapping": {},
        "num_states": len(multi_agent.state_agents),
        "num_aps": ap_extractor.num_transitions()
    }
    
    os.makedirs("automaton_q", exist_ok=True)
    
    for state_id, agent in multi_agent.state_agents.items():
        buffer = multi_agent.state_buffers[state_id]
        
        if buffer.num_filled_approx() < 10:
            print(f"Skipping agent {state_id} - insufficient data")
            continue
        
        print(f"Processing agent for state {state_id}")
        
        # Initialize Q-automaton arrays
        aut_num_q = torch.zeros((automaton.num_states, ap_extractor.num_transitions()), 
                               dtype=torch.int32, device=device)
        aut_total_q = torch.zeros_like(aut_num_q, dtype=torch.float32)
        aut_num_v = torch.zeros(automaton.num_states, dtype=torch.int32, device=device)
        aut_total_v = torch.zeros_like(aut_num_v, dtype=torch.float32)
        
        # Sample experiences from buffer and calculate Q-values
        num_samples = min(buffer.num_filled_approx(), 1000)  # Limit for efficiency
        
        for _ in tqdm(range(num_samples // 32), desc=f"Processing state {state_id}"):
            try:
                batch_data = buffer.get_minibatch(batch_size=32)
                states, actions, rewards, next_states, terminals, dfa_states = batch_data
                
                # Convert to tensors
                states_tensor = torch.FloatTensor(states).to(device)
                actions_tensor = torch.LongTensor(actions).to(device)
                dfa_states_tensor = torch.LongTensor(dfa_states).to(device)
                
                # Calculate Q-values
                with torch.no_grad():
                    q_values = agent.dqn(states_tensor)
                    v_values = q_values.max(dim=1)[0]
                    action_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                
                # Extract APs from states
                aps, _ = ap_extractor.extract_aps_batch(states_tensor)
                
                # Accumulate Q-values
                for j in range(len(states)):
                    aut_state = dfa_states_tensor[j].item()
                    ap = aps[j].item()
                    
                    if 0 <= aut_state < automaton.num_states and 0 <= ap < ap_extractor.num_transitions():
                        aut_num_q[aut_state, ap] += 1
                        aut_total_q[aut_state, ap] += action_q_values[j].item()
                        aut_num_v[aut_state] += 1
                        aut_total_v[aut_state] += v_values[j].item()
                        
            except Exception as e:
                print(f"Warning: Batch processing failed: {e}")
                continue
        
        # Save Q-automaton for this agent
        q_data = {
            "aut_num_q": aut_num_q.cpu().tolist(),
            "aut_total_q": aut_total_q.cpu().tolist(),
            "aut_num_v": aut_num_v.cpu().tolist(),
            "aut_total_v": aut_total_v.cpu().tolist()
        }
        
        agent_filename = f"{run_name}_state_{state_id}.json"
        with open(f"automaton_q/{agent_filename}", 'w') as f:
            json.dump(q_data, f, indent=2)
        
        combined_data["state_agent_mapping"][state_id] = agent_filename
        print(f"Saved Q-automaton for agent {state_id}")
    
    # Save combined metadata
    with open(f"automaton_q/{run_name}_combined.json", 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"Q-automaton construction completed for {run_name}")
    print(f"Generated {len(combined_data['state_agent_mapping'])} Q-automaton files")