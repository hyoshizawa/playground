"""
Some notes:
    CTA: call-tolaction

"""
import numpy as np
from matplotlib import pyplot as plt
import pymc3 as pm
from mpl_toolkits.mplot3d import Axes3D


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


def ch03_03():
    """ Comparing thetas by design.
    """
    n = [434, 382, 394, 88]  # page views.
    clicks = [8, 17, 10, 4]
    with pm.Model() as model:
        theta = pm.Uniform('theta', lower=0, upper=1, shape=len(n))
        obs = pm.Binomial('obs', p=theta, n=n, observed=clicks)
        trace = pm.sample(5000, chains=2)

    img = [0, 0, 1, 1]
    btn = [0, 1, 0, 1]
    with pm.Model() as model_comb:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
        comb = alpha + beta[0] * img + beta[1] * btn
        theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-comb)))
        obs = pm.Binomial('obs', p=theta, n=n, observed=clicks)
        trace_comb = pm.sample(5000, chains=2)

    with pm.Model():
        pm.forestplot([trace, trace_comb], var_names=['theta'],
                      hdi_prob=0.95, combined=True,
                      model_names=['Individual', 'Combined'])

    print((trace_comb['theta'][:, 1] - trace_comb['theta'][:, 0] > 0).mean())
    print((trace_comb['theta'][:, 3] - trace_comb['theta'][:, 0] > 0).mean())


def ch03_04():
    """ Interaction term.
    """
    n = [434, 382, 394, 412]  # page views.
    clicks = [8, 17, 10, 8]
    img = [0, 0, 1, 1]
    btn = [0, 1, 0, 1]

    with pm.Model() as model_comb2:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
        comb = alpha + beta[0] * img + beta[1] * btn
        theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-comb)))
        obs = pm.Binomial('obs', p=theta, n=n, observed=clicks)
        trace_comb2 = pm.sample(5000, chains=2)
        pm.traceplot(trace_comb2, compact=True)

    with pm.Model() as model_int:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
        gamma = pm.Normal('gamma', mu=0, sigma=10)
        comb = alpha + beta[0] * img + beta[1] * btn + gamma * img * btn
        theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-comb)))
        obs = pm.Binomial('obs', p=theta, n=n, observed=clicks)
        trace_int = pm.sample(5000, chains=2)
        pm.traceplot(trace_int, compact=True)

    x1 = np.arange(0, 1, 0.1)
    x2 = np.arange(0, 1, 0.1)
    X1, X2 = np.meshgrid(x1, x2)
    fig = plt.figure()
    ax = Axes3D(fig)
    logit_theta = (trace_comb2['alpha'].mean() +
        trace_comb2['beta'][:, 0].mean() * X1 +
        trace_comb2['beta'][:, 1].mean() * X2)
    surf = ax.plot_surface(X1, X2, logit_theta, cmap='plasma')
    fig.colorbar(surf)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$logit(\theta)$')
    plt.show()

    # interaction.
    x1 = np.arange(0, 1, 0.1)
    x2 = np.arange(0, 1, 0.1)
    X1, X2 = np.meshgrid(x1, x2)
    fig = plt.figure()
    ax = Axes3D(fig)
    logit_theta = (trace_int['alpha'].mean() +
                   trace_int['beta'][:, 0].mean() * X1 +
                   trace_int['beta'][:, 1].mean() * X2 +
                   trace_int['gamma'].mean() * X1 * X2)
    surf = ax.plot_surface(X1, X2, logit_theta, cmap='plasma')
    fig.colorbar(surf)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$logit(\theta)$')
    plt.show()
