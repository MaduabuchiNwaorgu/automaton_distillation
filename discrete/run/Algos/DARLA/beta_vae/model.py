import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, n_obs, latent_dim=10):
        """
        n_obs: dimensionality of the input to the VAE (e.g., output size of your feature extractor)
        latent_dim: dimensionality of the latent space.
        """
        super(Model, self).__init__()
        # Encoder layers
        self.fc1 = nn.Linear(n_obs, 128)
        self.fc2 = nn.Linear(128, 64)
        # Two separate heads for mean and log variance
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, n_obs),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
