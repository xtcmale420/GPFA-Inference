import torch
from sklearn.model_selection import KFold
from pp import PointProcessGPFA
from base import BaseGP, CosineKernel

# Example structure of a training loop within pytorch
def train_loop(y, model, loss, tol=1e-3):
    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    tol = 1e-3
    delta_loss = 1e6

    while delta_loss > tol:
        optimizer.zero_grad()
        l = -1*loss(y)
        print('l:%f' % l)
        # This takes the calculate value, and automatically calculates gradients with respect to parameters
        l.backward()
        # Optimizer will take the gradients, and then update parameters accordingly
        optimizer.step()
        # Calculate new loss given the parameter update
        l1 = -1*loss(y).detach()
        delta_loss = torch.abs(l1 - l)
        print('l1:%f' % l1)
    
    # Extract the parameters
    n = 0
    sigma = []
    mu = []
    for parameters in model.parameters():
        
        if (n==0):
            W = torch.tensor(parameters)
        elif(n%2==1):
            sigma.append(torch.tensor(parameters))
        elif(n%2==0):
            mu.append(torch.tensor(parameters))   
        n = n+1
    
    return W, sigma, mu

             
# Cross-Validation
def Cross_Validation(y):
    
    sk = KFold(10)
    sk = sk.split(y)
    Marginal_likelihood = torch.tensor(0)
    latent_dim = 0
    print('The closest latent dimension is:', latent_dim)
    # Check each possible value of the dimension of latent
    for i in range(3, y.shape[1]):
        
        Evaluate_Score_Sum = torch.tensor(0)
        
        # K-fold Cross-Validation
        for train, test in sk:
            
            # Integrate traindata and testdata
            traindata = y[train]
            testdata = y[test]

            # Model
            base_models = [BaseGP(CosineKernel) for _ in range(i)]
            model = PointProcessGPFA(4, base_models, W=None)
            
            # Training
            W, sigma, mu = train_loop(traindata, model, model.approximate_marginal_likelihood)

            # Marginal likelihood
            base_models = [BaseGP(CosineKernel, sigma=sigma[j], mu=mu[j]) for j in range(i)]
            PG = PointProcessGPFA(4, base_models, W)
            Evaluate_Score = -1*PG.approximate_marginal_likelihood(testdata)
            Evaluate_Score_Sum = torch.add(Evaluate_Score_Sum, Evaluate_Score)
            
        Evaluate_Score_Sum = torch.div(Evaluate_Score_Sum, 10)
        print('Evaluate Score:', Evaluate_Score_Sum)
        if (Evaluate_Score_Sum < Marginal_likelihood):
            Marginal_likelihood = Evaluate_Score_Sum
            latent_dim = i
        print('Marginal likelihood:', Marginal_likelihood)
        print('The closest latent dimension is:', latent_dim)