import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, n_obs, hidden_dim=64):
        """
        n_obs: Dimensionality of the autoencoder input (e.g., the output size of your feature extractor)
        hidden_dim: Dimensionality of the hidden latent representation.
        """
        super(Model, self).__init__()
        # Encoder: maps noisy input to a latent representation
        self.encoder = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU()
        )
        # Decoder: reconstructs the input from the latent representation
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_obs),
            nn.Sigmoid()  )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
