import numpy as np
import scipy
import pdb
import torch
from torch import nn

from base import enforce_type

# Chebyshev approximation
def get_chebyshev_coef(f, x0, x1, m):
    
    nx = 1000
    x = np.linspace(x0, x1, nx)
    y = f(x)
    c = np.polynomial.chebyshev.Chebyshev.fit(x, y, deg=m, domain=(x0, x1))
    return np.polynomial.chebyshev.cheb2poly(c.coef)

# Chebyshev approximation (Practical)
class CountDistribution():

    def __init__(self):
        pass
    
    @classmethod
    def f(self, x):
        raise NotImplementedError

    @classmethod
    def approximate(cls, x0, x1):
        x0 = np.array(x0)
        x1 = np.array(x1)
        
        a = torch.zeros(x0.shape[0])
        b = torch.zeros(x0.shape[0])
        c = torch.zeros(x0.shape[0])
        for j in range(x0.shape[0]):
            coef_ = get_chebyshev_coef(cls.f, x0[j], x1[j], 2)
            c[j] = torch.tensor(coef_[0])
            b[j] = torch.tensor(coef_[1])
            a[j] = torch.tensor(coef_[2])

        return a, b, c
    
class PoissonCountDistribution(CountDistribution):

    def __init__(self):
        pass

    @classmethod
    def f(cls, x):
        return np.exp(x)

    @classmethod
    def approximate(cls, y):
        y = np.array(y)
        
        x0 = []
        for i in range(y.shape[1]):
            x0_i = np.mean(y[:, i]) - 2
            x0.append(x0_i)
        x1 = []
        for i in range(y.shape[1]):
            x1_i = np.mean(y[:, i]) + 2
            x1.append(x1_i)
        return super().approximate(x0, x1)

    ############## For diagnostic purposes only ###########################
    @classmethod
    def _ll_approx(cls, lambd, a, b):

        approx_exp = np.sum([np.sum(a[i] * np.multiply(l, l) + b[i] * l)
                             for i, l in enumerate(lambd)])

        return approx_exp

    @classmethod
    def _ll_exact(cls, x):
        return np.sum([np.exp(xx) for xx in x])

    @classmethod
    def get_marginal_ll_approx(cls, y, K, W, a, b, use_torch):
        if use_torch:
            y = enforce_type(y, torch.Tensor)
            K = enforce_type(y, torch.Tensor)
            W = enforce_type(y, torch.Tensor)

            Sigma_inv = torch.kron(2 * W.T @ torch.diag(cls.a) @ W, torch.eye(y.shape[0])) + torch.block_diag([torch.inv(k) for k in K])            
            pdb.set_trace()
        else:
            y = enforce_type(y, np.ndarray)
            K = enforce_type(y, np.ndarray)
            W = enforce_type(y, np.ndarray)

    @classmethod
    def get_marginal_ll_exact(cls, y, K, W, a, b):
        y = enforce_type(y, np.ndarray)
        K = enforce_type(y, np.ndarray)
        W = enforce_type(y, np.ndarray)

        # Numerically integrate to get the exact marginal likleihood
        pass


# Uses Cheybshev approximations to compute the marginal likelihood
class PointProcessGPFA(nn.Module):

    def __init__(self, obs_dim, base_models, W=None):

        super(PointProcessGPFA, self).__init__()

        self.obs_dim = obs_dim
        self.latent_dim = len(base_models)
        self.count_dist = PoissonCountDistribution()

        if W is None:
            W = torch.rand((obs_dim, self.latent_dim))
        else:
            assert(W.shape[0] == self.obs_dim)
            W = enforce_type(W, torch.Tensor)

        self.W = nn.Parameter(W)
        self.base_models = nn.ModuleList(base_models)

    ######### For diagnostic only ##########################
    def _linear_ll(self, x, y):
        W = enforce_type(self.W, np.ndarray)
        K = enforce_type(self.K, np.ndarray)

        xy_cross = sum([yy @ W @ xx for xx, yy in zip(x, y)])
        xx_quad = -1/2 * sum([x[:, i] @ np.linalg.inv(K[i]) @ x[:, i] for i in range(x.shape[1])])

        return xy_cross - xx_quad - 1/2 * sum([np.linalg.slogdet(k)[1] for k in K])

    def exact_likelihood(self, x, y):
        W = enforce_type(self.W, np.ndarray)
        Wx = np.array([W @ xx for xx in x])

        return self._linear_ll(x, y) - self.count_dist._ll_exact(Wx)

    def approx_likelihood(self, x, y):
        W = enforce_type(self.W, np.ndarray)
        Wx = np.array([W @ xx for xx in x])

        return self._linear_ll(x, y) - self.count_dist._ll_approx(Wx, self.a, self.b)

    ########################################################

    def sample(self, T):
        # Requires base_models to have a sample function implemented
        x = torch.cat([torch.unsqueeze(model.sample(T), 1) for model in self.base_models], 1)
        lambd_ = torch.exp(x @ self.W.T)
        y = torch.cat([torch.unsqueeze(torch.poisson(lambd_[:, i]), 1) for i in range(self.obs_dim)], 1)
        return y.detach().numpy()

    def get_chebyshev_coefficients(self, y):
        a, b, c = self.count_dist.approximate(y)

        self.a = a
        self.b = b
        self.c = c

    def approximate_marginal_likelihood(self, y):
        if not hasattr(self, 'a'):
            self.get_chebyshev_coefficients(y)
        y = np.array(y)        
        assert(y.shape[1] == self.obs_dim)
        T = y.shape[0]
        K = torch.block_diag(*[m.get_cov_matrix(T) for m in self.base_models])
        I = torch.eye(T)
        W1 = torch.kron(self.W, I)
        
        a = torch.kron(self.a, torch.ones(T))
        b = torch.kron(self.b, torch.ones(T))
        
        y = torch.tensor(y).transpose(0, 1).ravel()

        # Need to reshape appropriately
        A = torch.diag(a)
        S = torch.chain_matmul(torch.t(W1), A, W1)
        
        Sigma_inv = torch.multiply(2, S) + torch.linalg.inv(K)
        mu = torch.matmul(torch.linalg.inv(Sigma_inv), torch.matmul(torch.t(W1), torch.subtract(y, b)))
        term1 = -1*0.5 * torch.slogdet(Sigma_inv)[1]
        term2 = 0.5*torch.chain_matmul(torch.unsqueeze(mu, 0), Sigma_inv, torch.unsqueeze(mu, 1))[0, 0]
        
        term3 = 0.5*torch.slogdet(K)[1]

        log_likelihood = term1 + term2 - term3
        
        return log_likelihood
