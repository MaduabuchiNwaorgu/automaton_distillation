import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from discrete.lib.agent.feature_extractor import FeatureExtractor
from beta_vae.model import Model

class BetaVAE:
    def __init__(self, input_shape, n_obs, num_epochs, batch_size, lr, beta, save_iter, shape):
        """
        n_obs: dimensionality of the VAE input (should match feature extractor output if using one)
        shape: shape for reshaping output images (e.g., (channels, height, width))
        """
        self.input_shape = input_shape
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta
        self.save_iter = save_iter
        self.shape = shape
        self.n_obs = n_obs

        # Initialize the feature extractor and VAE model
        self.feature_extractor = FeatureExtractor(input_shape=input_shape)
        self.vae = Model(n_obs)

    def encode(self, x):
        return self.vae.encode(x)

    def decode(self, z):
        return self.vae.decode(z)

    def train(self, history, extractor=None):
        """
        Train the β-VAE using history of observations.
        extractor: an optional feature extractor to compute reconstruction loss in feature space.
        """
        print('Training β-VAE...', end='', flush=True)

        # KL Divergence loss
        def KL(mu, log_var):
            # Compute KL divergence per batch
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl /= mu.size(0) * self.n_obs
            return kl

        optimizer = optim.Adam(self.vae.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            minibatches = history.get_minibatches(self.batch_size)
            epoch_loss = 0.0
            batch_count = 0

            for data in minibatches:
                data = data.to(next(self.vae.parameters()).device)
                # Forward pass: get reconstruction, mean, and log variance
                out, mu, log_var = self.vae(data)

                # Compute reconstruction loss
                if extractor is not None:
                    extractor.eval()
                    with torch.no_grad():
                        data_features = extractor(data)
                        out_features = extractor(out)
                    recon_loss = F.mse_loss(out_features, data_features)
                else:
                    recon_loss = F.mse_loss(out, data)

                # Compute KL divergence loss
                kl_loss = self.beta * KL(mu, log_var)
                loss = recon_loss + kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            print(f" Epoch [{epoch+1}/{self.num_epochs}] Loss: {avg_loss:.4f}")

            if (epoch + 1) % self.save_iter == 0:
                # Add code to save model checkpoint if desired.
                pass

        print('DONE')
