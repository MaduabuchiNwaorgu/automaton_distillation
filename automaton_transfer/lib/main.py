import copy

from torch.utils.tensorboard import SummaryWriter

from automaton_transfer.lib.config import Configuration
from automaton_transfer.lib.construct_q_automaton import construct_q_automaton
from automaton_transfer.lib.create_training_state import create_training_state
from automaton_transfer.lib.training import train_agent, distill_agent

def run_distill(config: Configuration):
    teacher, teacher_buffer, _, _, _ = create_training_state(config)
    
    student_config = config._replace(
        run_name=f"{config.run_name}_distill"
        # rollout_buffer_config=config.rollout_buffer_config._replace(capacity=1000)
    )
    
    student, student_buffer, ap_extractor, automaton, start_iter = create_training_state(student_config)

    logger = SummaryWriter(f"logs/{student_config.run_name}", purge_step=start_iter)

    distill_agent(student_config, teacher, student, automaton, ap_extractor, teacher_buffer, student_buffer, logger, start_iter)

def run_training(config: Configuration):
    agent, rollout_buffer, ap_extractor, automaton, start_iter = create_training_state(config)

    logger = SummaryWriter(f"logs/{config.run_name}", purge_step=start_iter)

    train_agent(config, agent, automaton, ap_extractor, rollout_buffer, logger, start_iter)

    construct_q_automaton(agent=agent, rollout_buffer=rollout_buffer, ap_extractor=ap_extractor, automaton=automaton,
                          device=config.device, run_name=config.run_name)