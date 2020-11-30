"""
"""
import numpy as np
from matplotlib import pyplot as plt


def main_01():
    """

    """
    thetas = np.linspace(0, 1, 1001)
    print(thetas)

    likelihood = lambda r: thetas if r else (1 - thetas)

    def posterior(r, prior):
        lp = likelihood(r) * prior
        return lp / lp.sum()

    p = np.array([1 / len(thetas) for _ in thetas])
    print(p)

    # Bayesian update by click-event(r=1)
    p = posterior(1, p)
    print(p)

    plt.plot(thetas, p)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$p(\theta)$')
    plt.show()

    #
    clicks = 2
    noclicks = 38
    p = np.array([1 / len(thetas) for theta in thetas])
    for _ in range(clicks):
        p = posterior(1, p)
    for _ in range(noclicks):
        p = posterior(0, p)
    print(p)
    plt.plot(thetas, p)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$p(\theta)$')
    plt.show()


def main_02():
    """
    Binomial
    """
    thetas = np.linspace(0, 1, 1001)
    print(thetas)

    likelihood = lambda a, N: thetas ** a * (1 - thetas) ** (N - a)

    def posterior(a, N, prior):
        lp = likelihood(a, N) * prior
        return lp / lp.sum()

    prior = 1 / len(thetas)
    plt.subplot(2, 1, 1)
    plt.plot(thetas, posterior(2, 40, prior), label='Alice - A')
    plt.plot(thetas, posterior(4, 50, prior), label='Alice - B')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$p(\theta)$')
    plt.xlim(0, 0.2)
    plt.legend()