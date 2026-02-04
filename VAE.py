# VAE.py
# Template for implementing a Variational Autoencoder (VAE) in PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        """
        Initialize the VAE model.

        Parameters:
        - input_dim: dimensionality of input (e.g., 784 for MNIST)
        - hidden_dim: number of units in the hidden layer
        - latent_dim: dimensionality of the latent space
        """
        super(VAE, self).__init__()

        # ===== TO DO: Define encoder layers =====
        # Example:
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # =======================================

        # ===== TO DO: Define decoder layers =====
        # Example:
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        # =======================================

    def encode(self, x):
        """
        Encode input x into latent mean and log-variance.

        Input:
        - x: tensor of shape (batch_size, input_dim)

        Returns:
        - mu: mean of latent distribution
        - logvar: log-variance of latent distribution
        """
        h1 = F.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)

        return mu, logvar 

    def reparameterize(self, mu, logvar):
        """
        Apply the reparameterization trick to sample z ~ N(mu, sigma^2).

        Input:
        - mu: mean tensor
        - logvar: log-variance tensor

        Returns:
        - z: sampled latent vector
        """
        std = torch.exp(0.5 * logvar) #standard deviation from log variance
        eps = torch.randn_like(std) #sample epsilon from std normal distribution
        z = mu + eps * std
        return z
        

    def decode(self, z):
        """
        Decode latent vector z to reconstruct input x_hat.

        Input:
        - z: latent vector tensor

        Returns:
        - x_hat: reconstructed input
        """
        h2 = F.relu(self.fc2(z))
        x_hat = torch.sigmoid(self.fc3(h2))

        return x_hat

    def forward(self, x):
        """
        Forward pass: encode -> reparameterize -> decode.

        Input:
        - x: input tensor

        Returns:
        - x_hat: reconstructed input
        - mu: latent mean
        - logvar: latent log-variance
        """
        mu, logvar = self.encode(x)
        z= self.reparameterize(mu, logvar)
        x_hat = self.decode(z)

        return x_hat, mu, logvar
        

def vae_loss(x, x_hat, mu, logvar):
    """
    Compute the VAE loss (ELBO) = reconstruction loss + KL divergence.

    Inputs:
    - x: original input
    - x_hat: reconstructed input
    - mu: latent mean
    - logvar: latent log-variance

    Returns:
    - loss: scalar tensor
    """
    #binary cross entropy
    reconstruction_loss = F.binary_cross_entropy(x_hat, x, reduction = 'sum')

    #KL divergence
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = reconstruction_loss + kl_divergence
    return total_loss
    









