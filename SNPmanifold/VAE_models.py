import torch
import torch.nn as nn


class VAE_base(nn.Module):
    def __init__(self, obs_dim, z_dim):
        """Init module with linear functions"""
        super(SNP_VAE, self).__init__()
        self.fc1 = nn.Linear(obs_dim, z_dim, bias = False)
        self.fc2 = nn.Linear(obs_dim, z_dim, bias = False)
        self.fc3 = nn.Linear(z_dim, obs_dim)

    def encode(self, x, cell_SNPread_weight=None):
        """Encoder"""
        _mu = self.fc1(torch.logit(x, eps = 0.01))
        _log_var = self.fc2(torch.logit(x, eps = 0.01))
        
        # optional normalization
        if cell_SNPread_weight is not None:
            _weight = cell_SNPread_weight / torch.mean(cell_SNPread_weight)
            _mu = _mu / _weight
            _log_var = _log_var / _weight
        
        return _mu, _log_var

    def reparameterize(self, mu, log_var):
        """Reparameterize trick"""
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):
        """Decoder"""
        return torch.sigmoid(self.fc3(z))

    def forward(self, x, cell_SNPread_weight=None):
        """
        Forward function with optional normalization
        
        Parameters
        ----------
        x: torch vector or matrix
            Input the values
        cell_SNPread_weight: torch vector
            Weights for normalization
        """
        mu, log_var = self.encode(x, cell_SNPread_weight=None) 
        z = self.reparameterize(mu, log_var)
        x_reconst_mu = self.decode(z)

        return x_reconst_mu, mu, log_var


class VAE_normalized(VAE_base):
    pass


class VAE_unnormalized(VAE_base):
    pass
