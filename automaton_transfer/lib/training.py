import copy
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from automaton_transfer.lib.agent.agent import Agent, TargetAgent
from automaton_transfer.lib.automaton.ap_extractor import APExtractor
from automaton_transfer.lib.automaton.automaton import Automaton
from automaton_transfer.lib.automaton.target_automaton import TargetAutomaton
from automaton_transfer.lib.automaton.reward_machine import RewardMachine
from automaton_transfer.lib.checkpoint import save_checkpoint, Checkpoint
from automaton_transfer.lib.config import Configuration
from automaton_transfer.lib.create_training_state import create_training_state
from automaton_transfer.lib.env.util import make_vec_env, make_env
from automaton_transfer.lib.intrinsic_reward import IntrinsicRewardCalculatorBatchWrapper
from automaton_transfer.lib.rollout_buffer import VecRolloutBufferHelper, RolloutBuffer, CircularRolloutBuffer
from automaton_transfer.lib.updater import Updater


def learn(config: Configuration, optim: Optimizer, agent: Agent, target_agent: TargetAgent,
          rollout_buffer: RolloutBuffer, automaton: Automaton, logger: SummaryWriter, iter_num: int):
    """
	Perform double Q-network gradient descent on a batch of samples from the rollout buffer (from deepsynth)
	"""
    optim.zero_grad()

    rollout_sample, indices, importance = rollout_buffer.sample(config.agent_train_batch_size,
                                                                automaton.num_states,
                                                                priority_scale=config.rollout_buffer_config.priority_scale)

    importance = torch.pow(importance, 1 - config.epsilon)  # So that high-priority states aren't _too_ overrepresented

    # Estimate best action in new states using main Q network
    q_max = agent.calc_q_values_batch(rollout_sample.next_states, rollout_sample.next_aut_states)
    arg_q_max = torch.argmax(q_max, dim=1)

    # Target DQN estimates q-values
    future_q_values = target_agent.calc_q_values_batch(rollout_sample.next_states, rollout_sample.next_aut_states)
    double_q = future_q_values[range(config.agent_train_batch_size), arg_q_max]

    # Calculate targets (bellman equation)
    target_q = rollout_sample.rewards + (config.gamma * double_q * (~rollout_sample.dones).float())
    target_q = target_q.detach()

    if isinstance(automaton, TargetAutomaton):
        target_automaton_q = automaton.target_q_values(rollout_sample.aut_states, rollout_sample.aps, iter_num)
        target_automaton_q_weights = automaton.target_q_weights(rollout_sample.aut_states, rollout_sample.aps, iter_num)

        target_q = (target_automaton_q * target_automaton_q_weights) + (target_q * (1 - target_automaton_q_weights))

        automaton.update_training_observed_count(rollout_sample.aut_states, rollout_sample.aps)

    # What are the q-values that the current agent predicts for the actions it took
    q_values = agent.calc_q_values_batch(rollout_sample.states, rollout_sample.aut_states)
    action_q_values = q_values[range(config.agent_train_batch_size), rollout_sample.actions]

    # Sample q values that we get wrong more often
    error = action_q_values - target_q
    rollout_buffer.set_priorities(indices=indices, errors=error.detach())

    # Actually train the neural network
    loss = F.mse_loss(input=action_q_values, target=target_q, reduction='none')
    loss = (loss * importance).mean()
    loss.backward()

    logger.add_scalar("training/loss", float(loss), global_step=iter_num)

    optim.step()
    
def crm(config: Configuration, optim: Optimizer, agent: Agent, target_agent: TargetAgent,
          rollout_buffer: RolloutBuffer, automaton: RewardMachine, logger: SummaryWriter, iter_num: int):
    """
    Perform double Q-network gradient descent on a batch of samples from the rollout buffer (from deepsynth)
    """
    optim.zero_grad()

    rollout_sample, indices, importance = rollout_buffer.sample(config.agent_train_batch_size,
                                                                automaton.num_states,
                                                                priority_scale=config.rollout_buffer_config.priority_scale)

    importance = torch.pow(importance, 1 - config.epsilon)  # So that high-priority states aren't _too_ overrepresented

    # Generate counterfactual experiences
    aut_states = torch.arange(automaton.aut.num_states * config.agent_train_batch_size, device=automaton.device) // config.agent_train_batch_size
    aps = rollout_sample.aps.repeat(automaton.aut.num_states)
    
    states = rollout_sample.states.repeat(automaton.aut.num_states)
    actions = rollout_sample.actions.repeat(automaton.aut.num_states)
    next_states = rollout_sample.next_states.repeat(automaton.aut.num_states)
    next_aut_states = automaton.aut.step_batch(aut_states, aps)
    rewards = automaton.reward_mat[aut_states, aps]
    dones = rollout_sample.dones.repeat(automaton.aut.num_states)

    # Estimate best action in new states using main Q network
    q_max = agent.calc_q_values_batch(next_states, next_aut_states)
    arg_q_max = torch.argmax(q_max, dim=1)

    # Target DQN estimates q-values
    future_q_values = target_agent.calc_q_values_batch(next_states, next_aut_states)
    double_q = future_q_values[range(config.agent_train_batch_size), arg_q_max]

    # Calculate targets (bellman equation)
    target_q = rewards + (config.gamma * double_q * (~dones).float())
    target_q = target_q.detach()

    # What are the q-values that the current agent predicts for the actions it took
    q_values = agent.calc_q_values_batch(states, aut_states)
    action_q_values = q_values[range(config.agent_train_batch_size), actions]

    # Sample q values that we get wrong more often
    error = action_q_values - target_q
    rollout_buffer.set_priorities(indices=indices, errors=error.detach())

    # Actually train the neural network
    loss = F.mse_loss(input=action_q_values, target=target_q, reduction='none')
    loss = (loss * importance).mean()
    loss.backward()

    logger.add_scalar("training/loss", float(loss), global_step=iter_num)

    optim.step()

def distill(config: Configuration, optim: Optimizer, teacher: Agent, student: Agent,
            rollout_buffer: RolloutBuffer, automaton: Automaton, logger: SummaryWriter, iter_num: int):
    """
    Perform policy distillation on a batch of samples from the rollout buffer
    """
    optim.zero_grad()

    rollout_sample, indices, importance = rollout_buffer.sample(config.agent_train_batch_size,
                                                                automaton.num_states,
                                                                priority_scale=config.rollout_buffer_config.priority_scale)
    
    # Teacher q-values
    teacher_q_values = teacher.calc_q_values_batch(rollout_sample.states, rollout_sample.aut_states)
    teacher_q_values_softmax = F.log_softmax(teacher_q_values / config.temperature, dim=1)
    
    # Student q-values
    student_q_values = student.calc_q_values_batch(rollout_sample.states, rollout_sample.aut_states)
    student_q_values_softmax = F.log_softmax(student_q_values / config.temperature, dim=1)

    # Train student
    loss = F.kl_div(input=student_q_values_softmax, target=teacher_q_values_softmax, log_target=True, reduction='batchmean')
    loss.backward()

    logger.add_scalar("training/loss", float(loss), global_step=iter_num)

    optim.step()


def take_eps_greedy_action_from_q_values(q_values: torch.Tensor, epsilon: float) -> np.ndarray:
    num_actions = q_values.shape[1]
    greedy_actions = torch.argmax(q_values, dim=1)
    modified_actions = torch.where(torch.rand_like(greedy_actions, dtype=torch.float32) > epsilon, greedy_actions,
                                   torch.randint_like(greedy_actions, num_actions))
    return modified_actions.detach().cpu().numpy()


def vec_env_distinct_episodes(states: torch.Tensor, infos: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
	The main annoyance of vecenv is that it automatically resets the environment after encountering a done
	The last observation is buried in the info dict for the vec.
	This function produces a vector of states that represent the step after the previous states,
	and a separate vector of states that represent the input to the next step
	"""
    states_after_current = states.clone()
    for i, info in enumerate(infos):
        if "terminal_observation" in info:
            states_after_current[i] = torch.as_tensor(info["terminal_observation"], device=states.device)

    return states_after_current, states


def reset_done_aut_states(aut_states_after_previous: torch.Tensor, dones: torch.Tensor,
                          automaton: Automaton) -> torch.tensor:
    """
	Reset the automaton state for resetted environments
	:param aut_states_after_previous: The automaton state of the environment, possibly of the terminal state
	:param dones: Which are actually terminal states
	"""

    """
	The orginal code is as follows but it gets error on my computer so I modified the code.
	return torch.where(dones, automaton.default_state, aut_states_after_previous)
	"""

    a = automaton.default_state
    a = np.int64(a)
    # print(dones)
    # print(a)
    b = torch.tensor(aut_states_after_previous.cpu().numpy(), dtype=torch.int64, device="cuda:0")

    return torch.where(dones, a, b)


class TraceHelper:
    """
	Keep track of all AP traces that haven't yet been used for synthesis- mostly an abstraction for vecenv
	"""

    def __init__(self, num_vec_envs: int):
        self.num_vec_envs = num_vec_envs
        self.completed_traces = []
        self.in_progress_traces = [[] for _ in range(num_vec_envs)]
        self.next_step = None  # Need to keep track of the most recent step separately

    def add_aps(self, aps):
        assert self.next_step is None, "Must finalize step before adding APs again"
        self.next_step = aps.tolist()

    def finalize_step(self, dones):
        # See note in train_agent about add_aps vs finalize_step.
        # Short version is that we want a way to recalculate the current state (without including the last AP) if the
        # automaton changes, but include the last AP when updating the automaton
        for i in range(len(dones)):
            self.in_progress_traces[i].append(int(self.next_step[i]))
            if dones[i]:
                self.completed_traces.append(self.in_progress_traces[i])
                self.in_progress_traces[i] = []

        self.next_step = None

    def get_traces_and_clear_completed(self):
        ret_traces = self.completed_traces
        self.completed_traces = []
        in_progress_traces_incl_next = copy.deepcopy(self.in_progress_traces)

        if self.next_step is not None:
            for i, in_progress_trace in enumerate(in_progress_traces_incl_next):
                in_progress_trace.append(self.next_step[i])

        ret_traces.extend(in_progress_traces_incl_next)
        ret_traces = [ret_trace for ret_trace in ret_traces if len(ret_trace) > 0]
        return ret_traces

def distill_agent(config: Configuration,
                  teacher: Agent,
                  student: Agent,
                  automaton: Automaton,
                  ap_extractor: APExtractor,
                  teacher_buffer: RolloutBuffer,
                  student_buffer: RolloutBuffer,
                  logger: SummaryWriter,
                  start_iter_num: int) -> Agent:
    """
    Distill knowledge from a teacher to a student
    :param teacher: The teacher agent for policy distillation
    :param student: The student agent for policy distillation
    :param config: Configuration for the whole training run
    :param automaton: The automaton to use during training. The states and transitions of the input will be updated
    :param ap_extractor: The weights of this will not be updated
    :param teacher_buffer: Teacher experience buffer
    :param student_buffer: Student experience buffer (only for logging purposes)
    :return: The trained agent
    """
    # TODO clarify ndarrays vs Tensors & devices
    env = make_vec_env(config.env_config, config.num_parallel_envs)
    
    buff_helper = VecRolloutBufferHelper(config.num_parallel_envs, student_buffer, logger,
                                         no_done_on_out_of_time=config.no_done_on_out_of_time)

    current_states = torch.as_tensor(env.reset(), device=config.device)
    current_aut_states = torch.tensor([automaton.default_state] * config.num_parallel_envs,
                                      device=config.device, dtype=torch.long)

    optimizer = torch.optim.Adam(student.parameters())

    trace_helper = TraceHelper(config.num_parallel_envs)
    batch_intrins_rew_calculator = IntrinsicRewardCalculatorBatchWrapper(config.intrinsic_reward_calculator,
                                                                         device=config.device)
    batch_intrins_reward_state = batch_intrins_rew_calculator.create_state(config.num_parallel_envs)

    checkpoint_updater = Updater(lambda: save_checkpoint(config, Checkpoint(
        iter_num=i,
        ap_extractor_state=ap_extractor.state_dict(),
        automaton_state=automaton.state_dict(),
        rollout_buffer_state=student_buffer.state_dict(),
        agent_state=student.state_dict()
    )))

    for i in range(start_iter_num, config.max_training_steps):
        # Generate experience
        q_values = student.calc_q_values_batch(torch.as_tensor(current_states, device=config.device, dtype=torch.float),
                                             current_aut_states)
        actions = take_eps_greedy_action_from_q_values(q_values, config.epsilon)
        obs, rewards, dones, infos = env.step(actions)
        obs = torch.as_tensor(obs, device=config.device)
        rewards = torch.as_tensor(rewards, device=config.device)
        dones = torch.as_tensor(dones, device=config.device)
        states_after_current, next_states = vec_env_distinct_episodes(obs, infos)

        aps_after_current = ap_extractor.extract_aps_batch(states_after_current, infos)

        # If dfa_updater changes the automaton, we need to recalculate the current automaton state
        # Since aps_after_current shouldn't be included in this calculation, trace_helper is "two-phase"
        # First, we add the aps to a special staging area where if the current automaton state must be recalculated,
        # these aps aren't included. Then, the finalize_step call merges in the newest aps.
        # The new aps are still included in the traces for the purposes of automaton synthesis.
        trace_helper.add_aps(aps_after_current)

        aut_states_after_current = automaton.step_batch(current_aut_states, aps_after_current)
        assert aut_states_after_current.min() != -1, "Automaton stepping failed"
        
        if isinstance(automaton, TargetAutomaton):
            rewards += automaton.target_reward_shaping(current_aut_states, aut_states_after_current)
        
        trace_helper.finalize_step(dones)

        intr_rewards = batch_intrins_rew_calculator.calc_intr_rewards_batch(batch_intrins_reward_state,
                                                                            current_states,
                                                                            actions,
                                                                            states_after_current,
                                                                            rewards,
                                                                            dones,
                                                                            current_aut_states,
                                                                            aps_after_current,
                                                                            aut_states_after_current)

        next_aut_states = reset_done_aut_states(aut_states_after_current, dones, automaton)
        
        # All of these are part of the same episode. next_states and next_aut_states may be part of a different episode
        buff_helper.add_vec_experiences(current_states=current_states,
                                        actions_after_current=actions,
                                        ext_rewards_after_current=rewards,
                                        intr_rewards_after_current=intr_rewards,
                                        dones_after_current=dones,
                                        states_after_current=states_after_current,
                                        current_aut_states=current_aut_states,
                                        aut_states_after_current=aut_states_after_current,
                                        aps_after_current=aps_after_current,
                                        infos=infos,
                                        global_step=i)

        current_states = next_states
        current_aut_states = next_aut_states

        logger.add_scalar("experience_generation/extrinsic_reward", float(rewards.float().mean()), global_step=i)
        logger.add_scalar("experience_generation/intrinsic_reward", float(intr_rewards.float().mean()), global_step=i)

        # Policy distillation
        distill(config=config, optim=optimizer, teacher=teacher, student=student, rollout_buffer=teacher_buffer,
                automaton=automaton, logger=logger, iter_num=i)

        checkpoint_updater.update_every(config.checkpoint_every_steps)

    return student

def train_agent(config: Configuration,
                agent: Agent,
                automaton: Automaton,
                ap_extractor: APExtractor,
                rollout_buffer: RolloutBuffer,
                logger: SummaryWriter,
                start_iter_num: int) -> Agent:
    """
	Train the agent for an entire generation
	:param agent: The agent to train
	:param config: Configuration for the whole training run
	:param automaton: The automaton to use during training. The states and transitions of the input will be updated
	:param ap_extractor: The weights of this will not be updated
	:param rollout_buffer: Assumed to already be labeled with the correct automaton states and intrinsic rewards, if any states are present
	:return: The trained agent
	"""
    # TODO clarify ndarrays vs Tensors & devices
    env = make_vec_env(config.env_config, config.num_parallel_envs)

    buff_helper = VecRolloutBufferHelper(config.num_parallel_envs, rollout_buffer, logger,
                                         no_done_on_out_of_time=config.no_done_on_out_of_time)

    target_agent = agent.create_target_agent()

    current_states = torch.as_tensor(env.reset(), device=config.device)
    current_aut_states = torch.tensor([automaton.default_state] * config.num_parallel_envs,
                                      device=config.device, dtype=torch.long)

    optimizer = torch.optim.Adam(agent.parameters())

    trace_helper = TraceHelper(config.num_parallel_envs)
    batch_intrins_rew_calculator = IntrinsicRewardCalculatorBatchWrapper(config.intrinsic_reward_calculator,
                                                                         device=config.device)
    batch_intrins_reward_state = batch_intrins_rew_calculator.create_state(config.num_parallel_envs)

    # The next few functions keep the main training loop concise by moving out some counting tasks
    target_agent_updater = Updater(lambda: target_agent.update_weights())

    checkpoint_updater = Updater(lambda: save_checkpoint(config, Checkpoint(
        iter_num=i,
        ap_extractor_state=ap_extractor.state_dict(),
        automaton_state=automaton.state_dict(),
        rollout_buffer_state=rollout_buffer.state_dict(),
        agent_state=agent.state_dict()
    )))
    
    if config.distill:
        student_config = config._replace(
            run_name=f"{config.run_name}_student",
            rollout_buffer_config=config.rollout_buffer_config._replace(capacity=1001)
        )
        
        student, student_buffer, student_ap_extractor, student_automaton, start_iter = create_training_state(student_config)
        
        student_logger = SummaryWriter(f"logs/{student_config.run_name}", purge_step=start_iter_num)
        
        student_env = make_vec_env(config.env_config, config.num_parallel_envs)
        
        student_buff_helper = VecRolloutBufferHelper(config.num_parallel_envs, student_buffer, student_logger,
                                         no_done_on_out_of_time=config.no_done_on_out_of_time)
        
        student_current_states = torch.as_tensor(student_env.reset(), device=config.device)
        student_current_aut_states = torch.tensor([automaton.default_state] * config.num_parallel_envs,
                                                  device=config.device, dtype=torch.long)
        
        student_trace_helper = TraceHelper(config.num_parallel_envs)
        
        student_batch_intrins_rew_calculator = IntrinsicRewardCalculatorBatchWrapper(config.intrinsic_reward_calculator,
                                                                         device=config.device)
        student_batch_intrins_reward_state = batch_intrins_rew_calculator.create_state(config.num_parallel_envs)
        
        student_checkpoint_updater = Updater(lambda: save_checkpoint(student_config, Checkpoint(
            iter_num=i,
            ap_extractor_state=ap_extractor.state_dict(),
            automaton_state=automaton.state_dict(),
            rollout_buffer_state=student_buffer.state_dict(),
            agent_state=student.state_dict()
        )))

    for i in range(start_iter_num, config.max_training_steps):
        # Generate experience
        q_values = agent.calc_q_values_batch(torch.as_tensor(current_states, device=config.device, dtype=torch.float),
                                             current_aut_states)
        actions = take_eps_greedy_action_from_q_values(q_values, config.epsilon)
        obs, rewards, dones, infos = env.step(actions)
        obs = torch.as_tensor(obs, device=config.device)
        rewards = torch.as_tensor(rewards, device=config.device)
        dones = torch.as_tensor(dones, device=config.device)
        states_after_current, next_states = vec_env_distinct_episodes(obs, infos)

        aps_after_current = ap_extractor.extract_aps_batch(states_after_current, infos)

        # If dfa_updater changes the automaton, we need to recalculate the current automaton state
        # Since aps_after_current shouldn't be included in this calculation, trace_helper is "two-phase"
        # First, we add the aps to a special staging area where if the current automaton state must be recalculated,
        # these aps aren't included. Then, the finalize_step call merges in the newest aps.
        # The new aps are still included in the traces for the purposes of automaton synthesis.
        trace_helper.add_aps(aps_after_current)

        aut_states_after_current = automaton.step_batch(current_aut_states, aps_after_current)
        assert aut_states_after_current.min() != -1, "Automaton stepping failed"
        
        if isinstance(automaton, TargetAutomaton):
            rewards += automaton.target_reward_shaping(current_aut_states, aut_states_after_current)
        
        trace_helper.finalize_step(dones)

        intr_rewards = batch_intrins_rew_calculator.calc_intr_rewards_batch(batch_intrins_reward_state,
                                                                            current_states,
                                                                            actions,
                                                                            states_after_current,
                                                                            rewards,
                                                                            dones,
                                                                            current_aut_states,
                                                                            aps_after_current,
                                                                            aut_states_after_current)

        next_aut_states = reset_done_aut_states(aut_states_after_current, dones, automaton)

        # All of these are part of the same episode. next_states and next_aut_states may be part of a different episode
        buff_helper.add_vec_experiences(current_states=current_states,
                                        actions_after_current=actions,
                                        ext_rewards_after_current=rewards,
                                        intr_rewards_after_current=intr_rewards,
                                        dones_after_current=dones,
                                        states_after_current=states_after_current,
                                        current_aut_states=current_aut_states,
                                        aut_states_after_current=aut_states_after_current,
                                        aps_after_current=aps_after_current,
                                        infos=infos,
                                        global_step=i)

        current_states = next_states
        current_aut_states = next_aut_states

        logger.add_scalar("experience_generation/extrinsic_reward", float(rewards.float().mean()), global_step=i)
        logger.add_scalar("experience_generation/intrinsic_reward", float(intr_rewards.float().mean()), global_step=i)

        if rollout_buffer.num_filled_approx() >= config.rollout_buffer_config.min_size_before_training:
            # Train off-policy
            if isinstance(automaton, RewardMachine):
                crm(config=config, optim=optimizer, agent=agent, target_agent=target_agent, rollout_buffer=rollout_buffer,
                    automaton=automaton, logger=logger, iter_num=i)
            else:
                learn(config=config, optim=optimizer, agent=agent, target_agent=target_agent, rollout_buffer=rollout_buffer,
                      automaton=automaton, logger=logger, iter_num=i)
            
            # Policy distillation
            if config.distill:
                distill(config=config, optim=optimizer, teacher=agent, student=student, rollout_buffer=rollout_buffer,
                        automaton=automaton, logger=student_logger, iter_num=i)

            target_agent_updater.update_every(config.target_agent_update_every_steps)

        checkpoint_updater.update_every(config.checkpoint_every_steps)
        
        if config.distill:
            q_values = student.calc_q_values_batch(torch.as_tensor(student_current_states, device=config.device, dtype=torch.float),
                                             student_current_aut_states)
            actions = take_eps_greedy_action_from_q_values(q_values, config.epsilon)
            obs, rewards, dones, infos = student_env.step(actions)
            obs = torch.as_tensor(obs, device=config.device)
            rewards = torch.as_tensor(rewards, device=config.device)
            dones = torch.as_tensor(dones, device=config.device)
            states_after_current, next_states = vec_env_distinct_episodes(obs, infos)

            aps_after_current = student_ap_extractor.extract_aps_batch(states_after_current, infos)

            # If dfa_updater changes the automaton, we need to recalculate the current automaton state
            # Since aps_after_current shouldn't be included in this calculation, trace_helper is "two-phase"
            # First, we add the aps to a special staging area where if the current automaton state must be recalculated,
            # these aps aren't included. Then, the finalize_step call merges in the newest aps.
            # The new aps are still included in the traces for the purposes of automaton synthesis.
            student_trace_helper.add_aps(aps_after_current)

            aut_states_after_current = student_automaton.step_batch(student_current_aut_states, aps_after_current)
            assert aut_states_after_current.min() != -1, "Automaton stepping failed"
            
            if isinstance(student_automaton, TargetAutomaton):
                rewards += student_automaton.target_reward_shaping(student_current_aut_states, aut_states_after_current)
            
            student_trace_helper.finalize_step(dones)

            intr_rewards = batch_intrins_rew_calculator.calc_intr_rewards_batch(batch_intrins_reward_state,
                                                                                student_current_states,
                                                                                actions,
                                                                                states_after_current,
                                                                                rewards,
                                                                                dones,
                                                                                student_current_aut_states,
                                                                                aps_after_current,
                                                                                aut_states_after_current)

            next_aut_states = reset_done_aut_states(aut_states_after_current, dones, student_automaton)

            # All of these are part of the same episode. next_states and next_aut_states may be part of a different episode
            student_buff_helper.add_vec_experiences(current_states=student_current_states,
                                                    actions_after_current=actions,
                                                    ext_rewards_after_current=rewards,
                                                    intr_rewards_after_current=intr_rewards,
                                                    dones_after_current=dones,
                                                    states_after_current=states_after_current,
                                                    current_aut_states=student_current_aut_states,
                                                    aut_states_after_current=aut_states_after_current,
                                                    aps_after_current=aps_after_current,
                                                    infos=infos,
                                                    global_step=i)

            student_current_states = next_states
            student_current_aut_states = next_aut_states

            student_logger.add_scalar("experience_generation/extrinsic_reward", float(rewards.float().mean()), global_step=i)
            student_logger.add_scalar("experience_generation/intrinsic_reward", float(intr_rewards.float().mean()), global_step=i)
            
            student_checkpoint_updater.update_every(config.checkpoint_every_steps)

    return agent
