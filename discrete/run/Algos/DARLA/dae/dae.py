from dae.model import Model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from discrete.lib.agent.feature_extractor import FeatureExtractor

class DAE:
    def __init__(self, input_shape, n_obs, num_epochs, batch_size, lr, noise_std=0.1, shape=None):
        """
        n_obs: Dimensionality of the DAE input (should match the output size of your feature extractor if using one)
        noise_std: Standard deviation of the Gaussian noise added to inputs.
        shape: Shape for reshaping output images if needed (e.g., (channels, height, width)).
        """
        self.input_shape = input_shape
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.noise_std = noise_std
        self.n_obs = n_obs
        self.shape = shape

        # Initialize feature extractor if using feature-space reconstruction loss
        self.feature_extractor = FeatureExtractor(input_shape=input_shape)
        # Initialize the DAE model
        self.dae = Model(n_obs)

    def train(self, history, extractor=None):
        """
        Train the Denoising Autoencoder using a history of observations.
        
        extractor: Optionally, a feature extractor to compute the reconstruction loss in feature space.
                   If None, the loss is computed in pixel space.
        """
        print('Training Denoising Autoencoder...', end='', flush=True)
        optimizer = optim.Adam(self.dae.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            minibatches = history.get_minibatches(self.batch_size)
            epoch_loss = 0.0
            batch_count = 0

            for data in minibatches:
                

                # Forward pass through the DAE
                out = self.dae(data)
                

                
                loss = torch.pow(data - out, 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            print(f" Epoch [{epoch+1}/{self.num_epochs}] Loss: {avg_loss:.4f}")

        print('DONE')
