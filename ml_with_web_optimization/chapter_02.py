"""

Some notes:
    MCMC: Markov chain Monte Carlo
    burn-in

For colab:
    !pip install -U ariviz==0.9.0 pymc3==3.9.3

"""
import numpy as np
from matplotlib import pyplot as plt
import pymc3 as pm


def ch02_01():
    """
    """
    N = 40
    a = 2

    with pm.Model() as model:
        theta = pm.Uniform('theta', lower=0, upper=1)
        # or
        # theta = pm.Beta('theta', alpha=1, beta=1)
        obs = pm.Binomial('a', p=theta, n=N, observed=a)
        trace = pm.sample(5000, chains=2)
    
        # plot
        pm.traceplot(trace)
        print(pm.summary(trace, hdi_prob=0.95))
        pm.plot_posterior(trace, hdi_prob=0.95)
        
        print((trace['theta'] - 0.01 > 0).mean())

    with pm.Model() as model:
        theta = pm.Uniform('theta', lower=0, upper=1, shape=2)
        obs = pm.Binomial('obs', p=theta, n=[40, 50], observed=[2, 4])
        trace = pm.sample(5000, chains=2)
    
        pm.traceplot(trace, ['theta'], compact=True)

        print((trace['theta'][:, 1] - trace['theta'][:, 0] > 0).mean())

    with pm.Model() as model:
        theta = pm.Uniform('theta', lower=0, upper=1, shape=2)
        obs = pm.Binomial('obs', p=theta, n=[1280, 1600], observed=[64, 128])
        trace = pm.sample(5000, chains=2)
        print((trace['theta'][:, 0] < trace['theta'][:, 1]).mean())
        pm.traceplot(trace, ['theta'], compact=True)
