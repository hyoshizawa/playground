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


def ch02_02():
    """
    theta ~ Dirichlet(alpha = (1, 1, 1, 1, 1))
    r ~ Categorical(theta)
    """
    n_a = [20, 10, 36, 91, 170] # N of score of product-A
    data = [0 for _ in range(n_a[0])]
    data += [1 for _ in range(n_a[1])]
    data += [2 for _ in range(n_a[2])]
    data += [3 for _ in range(n_a[3])]
    data += [4 for _ in range(n_a[4])]

    with pm.Model() as model_a:
        theta = pm.Dirichlet('theta', a=np.array([1, 1, 1, 1, 1]))
        obs = pm.Categorical('obs', p=theta, observed=data)
        trace_a = pm.sample(5000, chains=2)

    with model_a:
        pm.traceplot(trace_a)

        pm.plot_posterior(trace_a, hdi_prob=0.95)

    weights = np.array([1, 2, 3, 4, 5])
    m_a = [sum(row * weights) for row in trace_a['theta']]
    plt.hist(m_a, range=(3, 5), bins=50, density=True)
    plt.xlabel(r'$m_A$')
    plt.ylabel(r'$p(m_A)$')
    plt.show()

    n_b = np.array([0, 0, 4, 0, 6])
    with pm.Model() as model_b:
        theta = pm.Dirichlet('theta', a=np.array([1, 1, 1, 1, 1]))
        obs = pm.Multinomial('obs', p=theta, n=n_b.sum(), observed=n_b)
        trace_b = pm.sample(5000, chains=2)
        pm.traceplot(trace_b)

    m_b = [sum(row * weights) for row in trace_b['theta']]
    # plt.clf()
    plt.hist(m_a, range=(2, 5), bins=50, density=True, label='A', alpha=0.7)
    plt.hist(m_b, range=(2, 5), bins=50, density=True, label='B', alpha=0.7)
    plt.xlabel(r'$m$')
    plt.ylabel(r'$p(m)$')
    plt.legend()
    plt.show()
