import pdb

import numpy as np
import scipy

import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import mctorch.nn as mnn
import mctorch.optim as moptim

from base import enforce_type

# Condition the likelihood on the orthogonal combinations of the observed data to speed up multi-output inference
def OrthogonalMixingLikelihood(Y, U, logS, sigma_sq, logD, base_models):

    # Form the summary statistics
    H = torch.matmul(U, torch.diag(torch.pow(torch.exp(logS), -1)))
    yproj = torch.matmul(Y, H)

    # Observation noise
    Sigma_T = sigma_sq * torch.pow(torch.exp(logS), -1) + torch.exp(logD) 

    # Log marginal likelihoods of the individual base problems
    lml = torch.sum([base_model.marginal_likelihood(yproj, Sigma_T) for base_model in base_models])
    
    # Keeping notation from OLMM paper, m is the number of latent processes
    n = Y.shape[0]
    p = U.shape[0]
    m = U.shape[1]
    reg = -n/2 * torch.abs(torch.prod(logS)) - n * (p - m)/2 * torch.log(2 * np.pi * sigma_sq) - \
          1/(2 * sigma_sq) * (torch.pow(torch.linalg.norm(Y), 2) - torch.pow(torch.linalg.norm(torch.matmul(Y, U)), 2))
    return reg + lml 

class OrthogonalMixingModel(nn.Module):

    def __init__(self, U, logS, sigma_sq, logD, base_model, **base_model_kwargs):
        super().__init__()
        U = enforce_type(U)
        logS = enforce_type(logS)
        sigma_sq = enforce_type(sigma_sq)
        logD = enforce_type(logD)
        self.base_models = nn.ModuleList([base_model(**base_model_kwargs) for _ in range(torch.numel(logS))])
        
        self.U = mnn.Parameter(U, manifold=mnn.Stiefel(U.shape[0], U.shape[1]))
        self.logS = mnn.Parameter(logS)
        self.logD = mnn.Parameter(logD)
        self.sigma_sq = mnn.Parameter(sigma_sq)

    def fit(self, Y):

        # Fit by optimizing the marginal likelihood
        optimizer = moptim.rAdagrad(params = self.Parameters, lr=1e-2) 

        for _ in range(50):
            loss = OrthogonalMixingLikelihood(Y, self.U, self.logS, self.sigma_sq, self.logD, self.base_models)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # generate samples from the currently fit model
    def sample(self, n_s):
        # Requires base_models to have a sample function implemented
        x = torch.cat([torch.unsqueeze(model.sample(n_s), 1) for model in self.base_models], 1)
        # Sample observational noise in the projected space
        sigma_d = MultivariateNormal(torch.zeros(len(self.base_models)), covariance_matrix=torch.diag(torch.exp(self.logD))).sample(torch.Size([n_s]))    
        # H = U S^1/2
        H = self.U @ torch.diag(torch.exp(1/2 * self.logS))
        y = torch.matmul(x, torch.t(H))
        
        # Observational noise in the output space
        sigma_ = MultivariateNormal(torch.zeros(y.shape[-1]), self.sigma_sq * torch.eye(y.shape[-1])).sample(torch.Size([n_s]))
        y += sigma_
        return y
