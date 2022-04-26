import json
import os
import os.path as path

import torch
from tqdm.auto import tqdm

from automaton_transfer.lib.config import Configuration
from automaton_transfer.lib.create_training_state import create_training_state


def construct_q_automaton_from_config(config: Configuration):
    """Will automatically load from checkpoint and construct the Q automaton"""
    agent, rollout_buffer, ap_extractor, automaton, start_iter = create_training_state(config)

    construct_q_automaton(agent, rollout_buffer, ap_extractor, automaton, config.device, config.run_name)


def construct_q_automaton(agent, rollout_buffer, ap_extractor, automaton, device, run_name):
    """
    After an agent is trained, evaluate every state in the rollout buffer to determine the automaton q values
    """

    aut_num_q = torch.zeros((automaton.num_states, ap_extractor.num_transitions()), dtype=torch.int32,
                            device=device)
    aut_total_q = torch.zeros_like(aut_num_q, dtype=torch.float)

    for rollout_indices in tqdm(rollout_buffer.iterate_episode_indices()):
        sample = rollout_buffer.get_rollout_sample_from_indices(rollout_indices)
        q_vals_batch = agent.calc_q_values_batch(sample.states, sample.aut_states).detach()
        action_q_vals = q_vals_batch[range(len(q_vals_batch)), sample.actions]

        aut_num_q[sample.aut_states, sample.aps] += 1
        aut_total_q[sample.aut_states, sample.aps] += action_q_vals

    to_save = {
        "aut_num_q": aut_num_q.tolist(),
        "aut_total_q": aut_total_q.tolist()
    }

    if not path.exists("automaton_q"):
        os.mkdir("automaton_q")

    with open(f"automaton_q/{run_name}.json", "w") as f:
        json.dump(to_save, f)
