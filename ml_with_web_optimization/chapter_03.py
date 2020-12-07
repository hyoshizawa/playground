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
