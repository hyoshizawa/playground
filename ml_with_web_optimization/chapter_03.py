"""
Some notes:
    CTA: call-tolaction

"""
import numpy as np
from matplotlib import pyplot as plt
import pymc3 as pm


def ch03_01():
    """
    """
    n = [434, 382, 394, 88] # page views.
    clicks = [8, 17, 10, 4]
    with pm.Model() as model:
        theta = pm.Uniform('theta', lower=0, upper=1, shape=len(n))
        obs = pm.Binomial('obs', p=theta, n=n, observed=clicks)
        trace = pm.sample(5000, chains=2)
        pm.traceplot(trace, compact=True)

    with model:
        print(pm.summary(trace, hdi_prob=0.95))
        pm.forestplot(trace, combined=True, hdi_prob=0.95)
        
    print((trace['theta'][:, 1] - trace['theta'][:, 0] > 0).mean())
    print((trace['theta'][:, 3] - trace['theta'][:, 0] > 0).mean())


def ch03_02():
    """
    """
    n = [434, 382, 394, 88]  # page views.
    clicks = [8, 17, 10, 4]

    img = [0, 0, 1, 1]
    btn = [0, 1, 0, 1]

    with pm.Model() as model_comb:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
        comb = alpha + beta[0] * img + beta[1] * btn
        theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-comb)))
        obs = pm.Binomial('obs', p=theta, n=n, observed=clicks)
        trace_comb = pm.sample(5000, chains=2)

        pm.traceplot(trace_comb)

        pm.plot_posterior(trace_comb, var_names=['beta'], hdi_prob=0.95)

        print((trace_comb['beta'][:, 1] > 0).mean())
