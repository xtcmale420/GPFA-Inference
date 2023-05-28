import numpy as np
import scipy
import pdb
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal

# Convert between numpy and pytorch (Transformer)
def enforce_type(x, type_):

    if type_ == np.ndarray:
        if type(x) == list:
            if np.any([not isinstance(xx, np.ndarray) for xx in x]):
                for xx in x:
                    if isinstance(xx, torch.Tensor):
                        xx = xx.detach().numpy()
                    else:
                        xx = np.array(xx)

        else:
            if isinstance(x, torch.Tensor):
                x = x.detach().numpy()
            else:
                x = np.array(x)

    elif type_ == torch.Tensor:

        if type(x) == list:
            if np.any([not isinstance(xx, torch.Tensor) for xx in x]):
                x = [torch.tensor(xx).float() for xx in x]
        else:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            x = x.float()
    return x

class CosineKernel(nn.Module):

    def __init__(self, sigma=None, mu=None):
        super().__init__()        
        if sigma is None:
            sigma = 1.
        if mu is None:
            mu = 1.

        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.mu = nn.Parameter(torch.tensor(mu))
    
    def forward(self, t1, t2):
        result = torch.exp(-2*np.pi**2 * self.sigma**2 * np.abs(t1 - t2)**2) * torch.cos(2 * np.pi * np.abs(t1 - t2) * self.mu)        
        return result
    
class ExpKernel(nn.Module):

    def __init__(self, amplitude, lengthscale):
        super().__init__()
        self.amplitude = nn.Parameter(torch.tensor(amplitude))
        self.lengthscale = nn.Parameter(torch.tensor(lengthscale))

    def forward(self, x1, x2):
        return self.amplitude * torch.exp(-1 * (float(x1) - float(x2))**2/(2 * self.lengthscale**2))

class BaseGP(nn.Module):

    def __init__(self, kernel, **kernel_kwargs):
        super().__init__()
        self.kernel = kernel(**kernel_kwargs)
    
    def get_cov_matrix(self, T):
        return torch.cat([torch.unsqueeze(torch.tensor([self.kernel(i, j) for j in range(T)]), 0) for i in range(T)], dim=0)

    # Marginal likelihood for a Gaussian process with the given kernel
    def marginal_likelihood(self, y, sigma_sq):
        K = self.get_cov_matrix(y.numel())
        return -1/2 * torch.chain_matmul(y, torch.inverse(K + sigma_sq * torch.eye(K.shape[0])), y) - 1/2 * torch.linalg.slogdet(K + sigma_sq * torch.eye(K.shape[0]))[1]

    def sample(self, n_s):
        # return n_s samples
        K = self.get_cov_matrix(n_s)
        # Return 1 sample from a multidimensional distribution
        sample = MultivariateNormal(torch.zeros(n_s), K).sample()
        return sample
