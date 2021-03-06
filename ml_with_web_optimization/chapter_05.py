""" Multi-armed Bandit Algorithm.

"""
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(0)

n_arms = 4


class Env(object):
    thetas = [0.1, 0.2, 0.3, 0.4]

    def react(arm):
        return 1 if np.random.random() < Env.thetas[arm] else 0

    def opt():
        return np.argmax(Env.thetas)


class EpsilonGreedyAgent(object):

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def get_arm(self):
        if np.random.random() < self.epsilon:
            arm = np.random.randint(n_arms)
        else:
            arm = np.argmax(self.values)
        return arm

    def sample(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] = (
            (self.counts[arm] - 1) * self.values[arm] + reward
        ) / self.counts[arm]


class AnnealingEpsilonGreedyAgent(object):

    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def get_arm(self):
        if np.random.random() < self.epsilon:
            arm = np.random.randint(n_arms)
        else:
            arm = np.argmax(self.values)
        self.epsilon *= 0.99
        return arm

    def sample(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] = (
            (self.counts[arm] - 1) * self.values[arm] + reward
        ) / self.counts[arm]


class SoftmaxAgent(object):

    def __init__(self, tau=.05):
        self.tau = tau
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def softmax_p(self):
        logit = self.values / self.tau
        logit = logit - np.max(logit)
        p = np.exp(logit) / sum(np.exp(logit))
        return p

    def get_arm(self):
        arm = np.random.choice(n_arms, p=self.softmax_p())
        return arm

    def sample(self, arm, reward):
        self.counts[arm] = self.counts[arm] + 1
        self.values[arm] = (
            (self.counts[arm] - 1) * self.values[arm] + reward
        ) / self.counts[arm]


class AnnealingSoftmaxAgent(object):

    def __init__(self, tau=1000.):
        self.tau = tau
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def softmax_p(self):
        logit = self.values / self.tau
        logit = logit - np.max(logit)
        p = np.exp(logit) / sum(np.exp(logit))
        return p

    def get_arm(self):
        arm = np.random.choice(n_arms, p=self.softmax_p())
        self.tau = self.tau * 0.9
        return arm

    def sample(self, arm, reward):
        self.counts[arm] = self.counts[arm] + 1
        self.values[arm] = (
            (self.counts[arm] - 1) * self.values[arm] + reward
        ) / self.counts[arm]


class BernoulliTSAgent(object):

    def __init__(self):
        self.counts = [0 for _ in range(n_arms)]
        self.wins = [0 for _ in range(n_arms)]

    def get_arm(self):
        beta = lambda N, a: np.random.beta(a + 1, N - a + 1)
        result = [beta(self.counts[i], self.wins[i]) for i in range(n_arms)]
        arm = result.index(max(result))
        return arm

    def sample(self, arm, reward):
        self.counts[arm] = self.counts[arm] + 1
        self.wins[arm] = self.wins[arm] + reward


def sim(Agent, N=1000, T=1000, **kwargs):
    selected_arms = [[0 for _ in range(T)] for _ in range(N)]
    earned_rewards = [[0 for _ in range(T)] for _ in range(N)]

    for n in range(N):
        agent = Agent(**kwargs)
        for t in range(T):
            arm = agent.get_arm()
            reward = Env.react(arm)
            agent.sample(arm, reward)
            selected_arms[n][t] = arm
            earned_rewards[n][t] = reward
    return np.array(selected_arms), np.array(earned_rewards)


def ch05_01():
    """ """
    arms_eg, rewards_eg = sim(EpsilonGreedyAgent)
    acc = np.mean(arms_eg == Env.opt(), axis=0)

    plt.plot(acc)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\mathbb{E}[x(t) = x^*]$')
    plt.show()


def ch05_02():
    """ """
    arms_eg, rewards_eg = sim(EpsilonGreedyAgent)
    arms_aeg, rewards_eg = sim(AnnealingEpsilonGreedyAgent)
    plt.plot(np.mean(arms_aeg == Env.opt(), axis=0),
             label=r'Annealing $\varepsilon$-greedy')
    plt.plot(np.mean(arms_eg == Env.opt(), axis=0),
             label=r'$\varepsilon$-greedy')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$¥mathbb{E}[x(t) = x^*$')
    plt.legend()
    plt.show()


def ch05_03():
    """"""
    arms_sm, rewards_sm = sim(SoftmaxAgent)
    arms_asm, rewards_asm = sim(AnnealingSoftmaxAgent)

    plt.plot(np.mean(arms_asm == Env.opt(), axis=0),
             label='Annealing Softmax')
    plt.plot(np.mean(arms_sm == Env.opt(), axis=0), label='softmax')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\mathbb{E}[x(t) = x^*]$')
    plt.legend()
    plt.show()


def ch05_04():
    """"""
    arms_ts, rewards_ts = sim(BernoulliTSAgent)
    arms_asm, rewards_asm = sim(AnnealingSoftmaxAgent)

    plt.plot(np.mean(arms_ts == Env.opt(), axis=0),
             label='Thompson Sampling')
    plt.plot(np.mean(arms_asm == Env.opt(), axis=0),
             label='Annealing Softmax')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\mathbb{E}[x(t) = x^*]$')
    plt.legend()
    plt.show()
