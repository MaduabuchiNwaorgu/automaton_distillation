import os
import shutil
from typing import Dict, NamedTuple, OrderedDict

import torch

from automaton_transfer.lib.config import Configuration


class Checkpoint(NamedTuple):
    """
    The state of training; it should be self-consistent at any given point in time
    """
    iter_num: int
    ap_extractor_state: Dict
    automaton_state: Dict
    rollout_buffer_state: Dict
    agent_state: OrderedDict


def checkpoint_exists(config: Configuration) -> bool:
    return os.path.exists(f"./checkpoints/{config.run_name}")


def load_checkpoint(config: Configuration) -> Checkpoint:
    checkpoint_path = f"./checkpoints/{config.run_name}"
    checkpoint_dict = torch.load(checkpoint_path, map_location=config.device)
    return Checkpoint(
        iter_num=checkpoint_dict["iter_num"],
        ap_extractor_state=checkpoint_dict["ap_extractor"],
        automaton_state=checkpoint_dict["automaton"],
        rollout_buffer_state=checkpoint_dict["rollout_buffer"],
        agent_state=checkpoint_dict["agent"]
    )


def save_checkpoint(config: Configuration, checkpoint: Checkpoint):
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    checkpoint_path = f"checkpoints/{config.run_name}"

    if os.path.exists(f"checkpoints/{config.run_name}"):
        shutil.move(checkpoint_path, checkpoint_path + "_previous")

    checkpoint_dict = {
        "iter_num": checkpoint.iter_num,
        "ap_extractor": checkpoint.ap_extractor_state,
        "automaton": checkpoint.automaton_state,
        "rollout_buffer": checkpoint.rollout_buffer_state,
        "agent": checkpoint.agent_state
    }
    torch.save(checkpoint_dict, checkpoint_path)
