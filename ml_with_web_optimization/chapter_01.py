"""
"""
import numpy as np
from matplotlib import pyplot as plt


def ch01_01():
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


def ch01_02():
    """ Binomial distribution
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
    plt.subplot(2, 1, 2)
    plt.plot(thetas, posterior(64, 1280, prior), label='Bob - A')
    plt.plot(thetas, posterior(128, 1600, prior), label='Bob - B')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$p(\theta)$')
    plt.xlim(0, 0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


def ch01_03():
    """
        theta ~ Beta.
        alpha ~ Binomial.
    """
    thetas = np.linspace(0, 1, 1001)
    print(thetas)

    def betaf(alpha, beta):
        numerator = thetas ** (alpha - 1) * (1 - thetas) ** (beta - 1)
        return numerator / numerator.sum()

    def posterior(a, N):
        return betaf(a + 1, N - a + 1)

    plt.subplot(2, 1, 1)
    plt.plot(thetas, posterior(2, 40), label='Alice - A')
    plt.plot(thetas, posterior(4, 50), label='Alice - B')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$p(\theta)$')
    plt.xlim(0, 0.2)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(thetas, posterior(64, 1280), label='Bob - A')
    plt.plot(thetas, posterior(128, 1600), label='Bob - B')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$p(\theta)$')
    plt.xlim(0, 0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


def ch01_04():
    """

    """
    def hmv(xs, ps, alpha=0.95):
        """
        Parameters:
            xs :
            ps : 
            alpha : 
        Return:

        """
        xps = sorted(zip(xs, ps), key=lambda xp: xp[1], reverse=True)
        xps = np.array(xps)
        xs = xps[:, 0]
        ps = xps[:, 1]
        return np.sort(xs[np.cumsum(ps) <= alpha])

    thetas = np.linspace(0, 1, 1001)

    def posterior(a, N):
        alpha = a + 1
        beta = N - a + 1
        numerator = thetas ** (alpha - 1) * (1 - thetas) ** (beta - 1)
        return numerator / numerator.sum()

    ps = posterior(2, 40)
    hm_thetas = hmv(thetas, ps, alpha=0.95)
    plt.plot(thetas, ps)
    plt.annotate('', xy=(hm_thetas.min(), 0),
                 xytext=(hm_thetas.max(), 0),
                 arrowprops=dict(color='black', shrinkA=0, shrinkB=0,
                                 arrowstyle='<->', linewidth=2))
    plt.annotate('%.3f' % hm_thetas.min(), xy=(hm_thetas.min(), 0),
                 ha='right', va='bottom')
    plt.annotate('%.3f' % hm_thetas.max(), xy=(hm_thetas.max(), 0),
                 ha='left', va='bottom')
    plt.annotate('95% HDI', xy=(hm_thetas.mean(), 0),
                 ha='center', va='bottom')
    # restart here.
