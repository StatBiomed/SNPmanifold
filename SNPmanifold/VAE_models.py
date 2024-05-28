import torch
import torch.nn as nn


class VAE_normalized(nn.Module):
    
    def __init__(self, obs_dim, z_dim):
        
        """
        Init module with linear functions
        
        Parameters
        ----------
        obs_dim: integer
            dimension of input
        
        z_dim: integer
            dimension of latent space
        
        """

        super(VAE_normalized, self).__init__()
        self.fc1 = nn.Linear(obs_dim, z_dim, bias = False)
        self.fc2 = nn.Linear(obs_dim, z_dim, bias = False)
        self.fc3 = nn.Linear(z_dim, obs_dim)

    def encode(self, x, cell_SNPread_weight):
        
        """Encoder with observed-SNP normalization"""

        return self.fc1(torch.logit(x, eps = 0.01)) / cell_SNPread_weight * torch.mean(cell_SNPread_weight), self.fc2(torch.logit(x, eps = 0.01)) / cell_SNPread_weight * torch.mean(cell_SNPread_weight)

    def reparameterize(self, mu, log_var):
        
        """Reparameterize trick"""

        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):
        
        """Decoder"""

        return torch.sigmoid(self.fc3(z))

    def forward(self, x, cell_SNPread_weight):
        
        """
        Forward function with observed-SNP normalization

        Parameters
        ----------
        x: torch vector or matrix
            Input the values

        cell_SNPread_weight: torch vector
            Weights for observed-SNP normalization

        """

        mu, log_var = self.encode(x, cell_SNPread_weight) 
        z = self.reparameterize(mu, log_var)
        x_reconst_mu = self.decode(z)

        return x_reconst_mu, mu, log_var
    

class VAE_unnormalized(nn.Module):
    
    def __init__(self, obs_dim, z_dim):
        
        """
        Init module with linear functions
        
        Parameters
        ----------
        obs_dim: integer
            dimension of input
        
        z_dim: integer
            dimension of latent space
        
        """

        super(VAE_unnormalized, self).__init__()
        self.fc1 = nn.Linear(obs_dim, z_dim, bias = False)
        self.fc2 = nn.Linear(obs_dim, z_dim, bias = False)
        self.fc3 = nn.Linear(z_dim, obs_dim)

    def encode(self, x):
        
        """Encoder withput observed-SNP normalization"""

        return self.fc1(torch.logit(x, eps = 0.01)), self.fc2(torch.logit(x, eps = 0.01))

    def reparameterize(self, mu, log_var):
        
        """Reparameterize trick"""

        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):
        
        """Decoder"""

        return torch.sigmoid(self.fc3(z))

    def forward(self, x):
        
        """
        Forward function without observed-SNP normalization

        Parameters
        ----------
        x: torch vector or matrix
            Input the values

        """

        mu, log_var = self.encode(x) 
        z = self.reparameterize(mu, log_var)
        x_reconst_mu = self.decode(z)

        return x_reconst_mu, mu, log_var