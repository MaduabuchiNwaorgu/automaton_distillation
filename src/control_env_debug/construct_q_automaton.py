"""
Construct a Q-automaton from a trained TD3 agent using its replay buffer,
mirroring TMLR/discrete/lib/construct_q_automaton.py.

It aggregates, per automaton state u and AP index, the number of samples and
the sums of Q(s,u,a) and V(s,u) computed from the learned critic and policy.
"""

import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch


def _infer_ap_id(adj_matrix: np.ndarray, u_prev: int, u_next: int) -> int:
    row = adj_matrix[u_prev]
    idx = np.where(row == u_next)[0]
    return int(idx[0]) if len(idx) > 0 else -1


def construct_q_automaton(agent, replay_iter: Iterable[Tuple], automaton, run_name: str, out_dir: Path):
    """
    Build Q-automaton statistics from a TD3 agent and its replay buffer.

    replay_iter yields tuples: (obs, rm_state, action, reward, next_obs, next_rm_state, done)
    """
    device = agent.device
    adj = np.array(automaton.adj_matrix)
    num_states = int(automaton.num_states)
    num_aps = int(adj.shape[1])

    aut_num_q = torch.zeros((num_states, num_aps), dtype=torch.int32, device=device)
    aut_total_q = torch.zeros_like(aut_num_q, dtype=torch.float32)
    aut_num_v = torch.zeros(num_states, dtype=torch.int32, device=device)
    aut_total_v = torch.zeros_like(aut_num_v, dtype=torch.float32)

    for (obs, u_prev, action, _reward, _next_obs, u_next, _done) in replay_iter:
        u_prev = int(u_prev)
        u_next = int(u_next)
        ap = _infer_ap_id(adj, u_prev, u_next)

        with torch.no_grad():
            s = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, -1)
            u_oh = torch.nn.functional.one_hot(torch.tensor([u_prev], device=device), num_classes=num_states).float()
            a = torch.as_tensor(action, dtype=torch.float32, device=device).view(1, -1)

            q1 = agent.critic_1(s, u_oh, a).squeeze(1)
            a_pi = agent.actor(s, u_oh)
            v = agent.critic_1(s, u_oh, a_pi).squeeze(1)

        if 0 <= ap < num_aps:
            aut_num_q[u_prev, ap] += 1
            aut_total_q[u_prev, ap] += float(q1.item())
        aut_num_v[u_prev] += 1
        aut_total_v[u_prev] += float(v.item())

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}.json"
    to_save = {
        "aut_num_q": aut_num_q.cpu().tolist(),
        "aut_total_q": aut_total_q.cpu().tolist(),
        "aut_num_v": aut_num_v.cpu().tolist(),
        "aut_total_v": aut_total_v.cpu().tolist(),
        "num_states": num_states,
        "num_aps": num_aps,
    }
    with open(out_path, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"Saved Q-automaton to {out_path}")
