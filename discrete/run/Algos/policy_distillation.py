import torch
import math
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from discrete.lib.agent.agent import Agent, TargetAgent
from discrete.lib.automaton.ap_extractor import APExtractor
from discrete.lib.automaton.automaton import Automaton
from discrete.lib.automaton.target_automaton import TargetAutomaton
from discrete.lib.automaton.reward_machine import RewardMachine
from discrete.lib.checkpoint import save_checkpoint, Checkpoint,checkpoint_exists, load_checkpoint
from discrete.lib.config import Configuration
from discrete.lib.create_training_state import create_training_state
from discrete.lib.env.util import make_vec_env, make_env
from discrete.lib.intrinsic_reward import IntrinsicRewardCalculatorBatchWrapper
from discrete.lib.rollout_buffer import VecRolloutBufferHelper, RolloutBuffer, CircularRolloutBuffer
from discrete.lib.updater import Updater

from discrete.lib.training import take_eps_greedy_action_from_q_values, vec_env_distinct_episodes, reset_done_aut_states, TraceHelper

from scipy.stats import trim_mean



def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

def get_kl(teacher_dist_info, student_dist_info):
    pi = Normal(loc=teacher_dist_info[0], scale=teacher_dist_info[1])
    pi_new = Normal(student_dist_info[0], scale=student_dist_info[1])
    kl = torch.mean(kl_divergence(pi, pi_new))
    return kl

def get_wasserstein(teacher_dist_info, student_dist_info):
    means_t, stds_t = teacher_dist_info
    means_s, stds_s = student_dist_info
    return torch.sum((means_s - means_t) ** 2) + torch.sum((torch.sqrt(stds_s) - torch.sqrt(stds_t)) ** 2)


def discrete_kl(teacher_logits, student_logits, temperature=2.0):
    """
    teacher_logits, student_logits: shape [batch_size, num_actions]
    We do teacher_probs * (log teacher_probs - log student_probs).
    """
    # softmax over (logits / temperature)
    t = teacher_logits / temperature
    s = student_logits / temperature
    teacher_probs = F.softmax(t, dim=-1) + 1e-9
    student_probs = F.softmax(s, dim=-1) + 1e-9

    # print(teacher_probs.sum(dim=-1))
    kl = torch.sum(teacher_probs * (torch.log(teacher_probs) - torch.log(student_probs)), dim=-1).mean()
    
    return kl


def load_env_and_model(config,agent):

    print("Loading teacher model from checkpoint...")
    checkpoint = load_checkpoint(config)
    agent.load_state_dict(checkpoint.agent_state)

    return agent

class Teacher:
    def __init__(self, ap_extractor: APExtractor,
                 automaton: Automaton,
                 expert_policy: Agent,
                 rollout_buffer: RolloutBuffer,
                 config: Configuration,
                 logger):

        self.env = make_vec_env(config.env_config, config.num_parallel_envs)
        self.policy = expert_policy
        self.buffer_helper = VecRolloutBufferHelper(
            config.num_parallel_envs,
            rollout_buffer,
            logger,
            no_done_on_out_of_time=config.no_done_on_out_of_time
        )
        self.automaton = automaton
        self.buffer = rollout_buffer
        self.ap_extractor = ap_extractor
        self.config = config

        # Current states
        self.current_states = torch.as_tensor(self.env.reset(), device=config.device)
        self.current_aut_states = torch.tensor([automaton.default_state] * config.num_parallel_envs,
                                               device=config.device, dtype=torch.long)
        
        self.trace_helper = TraceHelper(config.num_parallel_envs)
        from discrete.lib.intrinsic_reward import IntrinsicRewardCalculatorBatchWrapper
        self.batch_intrins_rew_calculator = IntrinsicRewardCalculatorBatchWrapper(
            config.intrinsic_reward_calculator, device=config.device
        )
        self.batch_intrins_reward_state = self.batch_intrins_rew_calculator.create_state(config.num_parallel_envs)

    def collect_expert_samples(self, num_steps: int):
        """
        Step the environment `num_steps` times using the teacher’s policy,
        and store all transitions in the rollout buffer.
        """
        for _ in range(num_steps):
            # with torch.no_grad():
            q_values = self.policy.calc_q_values_batch(
                self.current_states.float(),
                self.current_aut_states
            )
            actions = take_eps_greedy_action_from_q_values(q_values, self.config.epsilon)

            # Step env
            obs, rewards, dones, infos = self.env.step(actions)
            obs = torch.as_tensor(obs, device=self.config.device)
            rewards = torch.as_tensor(rewards, device=self.config.device)
            dones = torch.as_tensor(dones, device=self.config.device)

            # Distinct episodes logic
            states_after_current, next_states = vec_env_distinct_episodes(obs, infos)

            # AP extraction
            aps_after_current = self.ap_extractor.extract_aps_batch(states_after_current, infos)
            self.trace_helper.add_aps(aps_after_current)
            aut_states_after_current = self.automaton.step_batch(self.current_aut_states, aps_after_current)
            self.trace_helper.finalize_step(dones)

            # Intrinsic reward
            intr_rewards = self.batch_intrins_rew_calculator.calc_intr_rewards_batch(
                self.batch_intrins_reward_state,
                self.current_states,
                actions,
                states_after_current,
                rewards,
                dones,
                self.current_aut_states,
                aps_after_current,
                aut_states_after_current
            )
            from discrete.lib.training import reset_done_aut_states
            next_aut_states = reset_done_aut_states(aut_states_after_current, dones, self.automaton)

            # Store transitions in buffer
            self.buffer_helper.add_vec_experiences(
                current_states=self.current_states,
                actions_after_current=actions,
                ext_rewards_after_current=rewards,
                intr_rewards_after_current=intr_rewards,
                dones_after_current=dones,
                states_after_current=states_after_current,
                current_aut_states=self.current_aut_states,
                aut_states_after_current=aut_states_after_current,
                aps_after_current=aps_after_current,
                infos=infos,
                global_step=num_steps  # or pass in an external counter if you need
            )

            # Move forward
            self.current_states = next_states
            self.current_aut_states = next_aut_states
            # print(dones)

        # No return; data is in self.buffer
        # self.collected_samples += self.config.num_parallel_envs


        # # Get final batch
        # rollout_sample, indices, _ = self.buffer.sample(
        #     batch_size=min_samples,
        #     num_aut_states=self.automaton.num_states if hasattr(self.automaton, 'num_states') else None,
        #     priority_scale=self.config.rollout_buffer_config.priority_scale,
        #     reward_machine=self.config.reward_machine
        # )

        



class Student:
    def __init__(self,
                 student_policy: Agent,
                 teacher_policy: Agent,
                 args,
                 config: Configuration,
                 automaton: Automaton,
                 rollout_buffer: RolloutBuffer):

        self.policy = student_policy.to(config.device)
        self.expert_policy = teacher_policy.to(config.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.config = config
        self.args = args

        self.automaton = automaton
        self.buffer = rollout_buffer

    def train_step(self) -> float:
        """
        Sample from the buffer, compute KL(teacher||student), do an update.
        Return the loss.
        """
        # Sample from buffer
        rollout_sample, indices, _ = self.buffer.sample(
            batch_size=self.args.student_batch_size,
            num_aut_states=self.automaton.num_states if hasattr(self.automaton, 'num_states') else None,
            priority_scale=self.config.rollout_buffer_config.priority_scale,
            reward_machine=self.config.reward_machine
        )
        states = rollout_sample.states.to(self.config.device)
        aut_states = rollout_sample.aut_states.to(self.config.device)

        # Teacher Q
        with torch.no_grad():
            teacher_q = self.expert_policy.calc_q_values_batch(states.float(), aut_states)

        # Student Q
        student_q = self.policy.calc_q_values_batch(states.float(), aut_states)

        # Discrete KL
        kl = discrete_kl(teacher_q, student_q, temperature=self.config.temperature)

        self.optimizer.zero_grad()
        kl.backward()
        self.optimizer.step()

        return kl.item()

    def save(self, filename="student_final.pkl.gz"):
        torch.save(self.policy.state_dict(), filename)


        # student_q_values = self.policy.calc_q_values_batch(torch.as_tensor(states, device=self.config.device, dtype=torch.float),
        #                                      aut_states)
        # print(student_q_values)
        # student_actions = take_eps_greedy_action_from_q_values(student_q_values, self.config.epsilon)
        # # student_actions = take_eps_greedy_action_from_q_values(student_q_values, self.config.epsilon)

        # # print(student_actions)

        # # Using KL-divergence or Wasserstein distance
        # loss = get_kl([expert_actions, torch.ones_like(expert_actions)*1e-6],
        #               [student_actions, torch.ones_like(student_actions)*1e-6])

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # return loss.item()
    
    # def save(self, filename="student_final.pkl.gz"):
    #     torch.save(self.policy.state_dict(), filename)
    
    # def test(self):

    #     pass 

    
def policy_distillation(config: Configuration,
                        agent: Agent,        # teacher's network
                        student: Agent,      # student's network
                        automaton: Automaton,
                        ap_extractor: APExtractor,
                        rollout_buffer: RolloutBuffer,
                        logger: SummaryWriter,
                        start_iter_num: int,
                        run_name=None,
                        args=None) -> Agent:
    
    # 1) Load teacher weights
    expert_policy = load_env_and_model(config, agent)

    # 2) Create teacher
    teacher = Teacher(
        ap_extractor=ap_extractor,
        automaton=automaton,
        expert_policy=expert_policy,
        rollout_buffer=rollout_buffer,
        config=config,
        logger=logger
    )

    # 3) Create student
    student = Student(
        student_policy=student,
        teacher_policy=expert_policy,
        args=args,
        config=config,
        automaton=automaton,
        rollout_buffer=rollout_buffer
    )

    # 4) Distillation loop
    for step_idx in range(args.max_iterations):

        # (A) Let teacher do N environment steps, e.g. 1 step each iteration
        teacher.collect_expert_samples(num_steps=1)  # or 10 or 100, etc.

        # (B) Periodically do a student training step
        if step_idx % 1000 == 0 and \
           rollout_buffer.num_filled_approx() >= config.rollout_buffer_config.min_size_before_training:
            
            # Student distillation update
            loss_val = student.train_step()

            # Print or log
            print(f"[Step {step_idx}] Distillation Loss: {loss_val:.4f}")
            logger.add_scalar("distillation/loss", loss_val, step_idx)

    # Save final
    student.save("student_final.pkl.gz")
    return student




