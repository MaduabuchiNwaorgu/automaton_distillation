import torch
import torch.optim as optim
from torch.utils.data import DataLoader



from dae.dae import DAE
from beta_vae.beta_vae import BetaVAE
from history import History
from discrete.lib.agent.feature_extractor import FeatureExtractor

from discrete.run.env.dungeon_quest import dungeon_quest_rew_per_step_env_config
from discrete.run.utils import teacher_config_v1
from discrete.lib.env.util import make_vec_env, make_env
from discrete.lib.create_training_state import create_training_state

device = torch.device("cuda:0")
config = teacher_config_v1(
    dungeon_quest_rew_per_step_env_config,
    "dungeon_quest_teacher_rew_per_step",
    device,
    max_training_steps=int(2e6))

obs_shape = (8, 7, 7)
agent, rollout_buffer, ap_extractor, automaton, start_iter = create_training_state(config)
env = make_vec_env(config.env_config, config.num_parallel_envs)

history = History(ap_extractor,automaton, agent,  config)

# Collect expert samples.
print("Collecting expert samples...")
history.collect_expert_samples(num_steps=10000)

# Create a feature extractor to determine the flattened feature dimension.
feature_extractor = FeatureExtractor(input_shape=obs_shape).to(device)
n_obs = feature_extractor.output_size
shape = obs_shape

# Hyperparameters.
num_epochs = 10
batch_size = 8
lr = 1e-4
beta = 4
save_iter = 5

# Instantiate DAE and Beta-VAE trainers.
dae_trainer = DAE(input_shape=obs_shape,
                                    n_obs=n_obs,
                                    num_epochs=num_epochs,
                                    batch_size=batch_size,
                                    lr=1e-3,
                                    noise_std=0.1,
                                    shape=shape)
beta_vae_trainer = BetaVAE(input_shape=obs_shape,
                            n_obs=n_obs,
                            num_epochs=num_epochs,
                            batch_size=batch_size,
                            lr=lr,
                            beta=beta,
                            save_iter=save_iter,
                            shape=shape)

# Train DAE.
dae_trainer.train(history)

# Train β-VAE using the DAE as an optional feature extractor for the reconstruction loss.
beta_vae_trainer.train(history, extractor=dae_trainer.dae)

print("DARLA representation learning complete.")