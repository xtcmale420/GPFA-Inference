import numpy as np
import sys
from pp import PointProcessGPFA
from base import BaseGP, CosineKernel
from train import train_loop, Cross_Validation


W = np.array([[-1.01038047e-7, 3.91755635e-1, -2.00601176e-7], [-1.18291113e-7, 3.93738773e-1, -2.05185896e-7], [-1.21354510e-7, 3.84734116e-1, -2.3405887e-7], [-9.20452121e-8, 3.87025115e-1, -2.11742197e-7]])
sigma = [3.58804129, 2.29182921, 3.44943223]
mu = [2.50000134, 1.98118901e-2, 2.49998292]
T = 100

base_models = [BaseGP(CosineKernel, sigma=sigma[i], mu=mu[i]) for i in range(3)]

PG0 = PointProcessGPFA(4, base_models, W)

#base_models = [BaseGP(CosineKernel) for _ in range(3)]
#PG1 = PointProcessGPFA(4, base_models, W=None)

#Generate a list of spike train
y = PG0.sample(T)

#ll1 = PG1.approximate_marginal_likelihood(y)
# What is the likelihood under the ground truth model?
#ll0 = PG0.approximate_marginal_likelihood(y)

#print(ll1)
#print(ll0)

# How well do we optimize?
#train_loop(y, PG1, PG1.approximate_marginal_likelihood)

Cross_Validation(y)