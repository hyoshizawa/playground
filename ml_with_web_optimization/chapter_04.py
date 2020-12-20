"""

"""
import random
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def is_valid(x, size):
    """ 実行可能解であることを確認する """
    return all(-1 < i < size for i in list(x))


class HillClimbing():
    """ 山登り法

    Args:
        init_x: 初期解
        init_f: 初期解の評価値
        size: サイズ

    Attributes:
        current_x: 現状解
        current_f: 現状解の評価値
    """
    def __init__(self, init_x, init_f, size):
        self.current_x = init_x
        self.current_f = init_f
        self.size = size
    
    def get_neighbors(self):
        """ 近傍解を出力する

        Returns:
            近傍解のリスト
        """
        neighbor_xs = []
        for i, xi in enumerate(self.current_x):
            neighbor_x = list(self.current_x)
            neighbor_x[i] += 1
            if is_valid(neighbor_x, self.size):
                neighbor_xs.append(tuple(neighbor_x))

            neighbor_x = list(self.current_x)
            neighbor_x[i] -= 1
            if is_valid(neighbor_x, self.size):
                neighbor_xs.append(tuple(neighbor_x))
        return neighbor_xs

    def update(self, neighbor_xs, neighbor_fs):
        """ 良い近傍解があれば現状解を更新する。

        Args:
            neighbor_xs: 評価済の近傍解のリスト
            neighbor_fs: 近傍解の評価値のリスト

        Returns:
            更新前の現状解と更新後の現状解のタプル
        """
        old_x = self.current_x
        if max(neighbor_fs) > self.current_f:
            self.current_x = neighbor_xs[
                neighbor_fs.index(max(neighbor_fs))]
            self.current_f = max(neighbor_fs)
        return (old_x, self.current_x)


class RandomizedHillClimbing:
    """ 乱択山登り法
    
    Args:
        init_x: 初期解
        init_f: 初期解の評価値

    Attributes:
        current_x: 現状解
        current_f: 現状解の評価値
    """
    def __init__(self, init_x, init_f, size):
        self.current_x = init_x
        self.current_f = init_f
        self.size = size

    def get_neighbors(self):
        """ 近傍解を出力する

        Returns:
            近傍解のリスト
        """
        neighbor_xs = []
        for i, xi in enumerate(self.current_x):
            neighbor_x = list(self.current_x)
            neighbor_x[i] += 1
            if is_valid(neighbor_x, self.size):
                neighbor_xs.append(tuple(neighbor_x))
            neighbor_x = list(self.current_x)
            neighbor_x[i] -= 1
            if is_valid(neighbor_x, self.size):
                neighbor_xs.append(tuple(neighbor_x))
        return neighbor_xs

    def get_neighbor(self):
        """ ランダムに近傍解を一つ選択する

        Returns:
            近傍解
        """
        return random.choice(self.get_neighbors())

    def update(self, neighbor_x, neighbor_f):
        """
        """
        old_x = self.current_x
        if self.current_f < neighbor_f:
            self.current_x = neighbor_x
            self.current_f = neighbor_f
        return (old_x, self.current_x)


class SimulatedAnnealing:
    """ 焼きなまし法
    
    Args:
        init_x: 初期解
        init_f: 初期解の評価値

    Attributes:
        current_x: 現状解
        current_f: 現状解の評価値
        temperature: 温度パラメータ
    """
    def __init__(self, init_x, init_f, size):
        self.current_x = init_x
        self.current_f = init_f
        self.temperature = 10
        self.size = size

    def get_neighbors(self):
        """ 近傍解を出力する

        Returns:
            近傍解のリスト
        """
        neighbor_xs = []
        for i, xi in enumerate(self.current_x):
            neighbor_x = list(self.current_x)
            neighbor_x[i] += 1
            if is_valid(neighbor_x, self.size):
                neighbor_xs.append(tuple(neighbor_x))
            neighbor_x = list(self.current_x)
            neighbor_x[i] -= 1
            if is_valid(neighbor_x, self.size):
                neighbor_xs.append(tuple(neighbor_x))
        return neighbor_xs

    def get_neighbor(self):
        """ ランダムに近傍解を一つ選択する

        Returns:
            近傍解
        """
        return random.choice(self.get_neighbors())

    def accept_prob(self, f):
        """受理確率"""
        return np.exp((f - self.current_f) / max(self.temperature, 0.01))

    def update(self, neighbor_x, neighbor_f):
        """
        """
        old_x = self.current_x
        if random.random() < self.accept_prob(neighbor_f):
            self.current_x = neighbor_x
            self.current_f = neighbor_f
        self.temperature *= 0.8
        return (old_x, self.current_x)


def ch04_01():
    """ """
    fig = plt.figure()
    ax = Axes3D(fig)

    size = 5
    _x1, _x2 = np.meshgrid(np.arange(size), np.arange(size))
    x1, x2 = _x1.ravel(), _x2.ravel()

    f = lambda x1, x2: 0.5 * x1 + x2 - 0.3 * x1 * x2
    ax.bar3d(x1, x2, 0, 1, 1, f(x1, x2), color='gray', edgecolor='white',
             shade=True)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x)$')
    plt.xticks(np.arange(0.5, size, 1), range(size))
    plt.yticks(np.arange(0.5, size, 1), range(size))
    plt.show()


def ch04_02():
    """ """
    size = 5
    _x1, _x2 = np.meshgrid(np.arange(size), np.arange(size))
    x1, x2 = _x1.ravel(), _x2.ravel()
    f = lambda x1, x2: 0.5 * x1 + x2 - 0.3 * x1 * x2

    init_x = (0, 0)
    init_f = f(init_x[0], init_x[1])
    hc = HillClimbing(init_x, init_f, size)

    evaluated_xs = {init_x}
    steps = []

    for _ in range(6):
        neighbor_xs = hc.get_neighbors()
        neighbor_fs = [f(x[0], x[1]) for x in neighbor_xs]
        step = hc.update(neighbor_xs, neighbor_fs)

        print('%s -> %s' % (step))
        steps.append(step)
        evaluated_xs.update(neighbor_xs)

    def visualize_path(evaluated_xs, steps):
        """"""
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-.5, size - .5)
        ax.set_ylim(-.5, size - .5)

        for i in range(size):
            for j in range(size):
                if (i, j) in evaluated_xs:
                    ax.text(i, j, '%.1f' % (f(i, j)), ha='center', va='center',
                            bbox=dict(edgecolor='gray', facecolor='none',
                                      linewidth=2))
                else:
                    ax.text(i, j, '%.1f' % (f(i, j)), ha='center', va='center')

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.xaxis.set_minor_locator(
            ticker.FixedLocator(np.arange(-.5, size - .5, 1)))
        ax.yaxis.set_minor_locator(
            ticker.FixedLocator(np.arange(-.5, size - .5, 1)))
        plt.tick_params(axis='both', which='both', bottom='off', top='off',
                        left='off', right='off', labelbottom='off',
                        labelleft='off')
        ax.grid(True, which='minor')
        ax.grid(False, which='minor')

        for step in steps:
            ax.annotate('', xy=step[1], xytext=step[0],
                        arrowprops=dict(shrink=0.2, width=2, lw=0))

    visualize_path(evaluated_xs, steps)


def ch04_03():
    """ """
    size = 5
    _x1, _x2 = np.meshgrid(np.arange(size), np.arange(size))
    x1, x2 = _x1.ravel(), _x2.ravel()
    f = lambda x1, x2: 0.5 * x1 + x2 - 0.3 * x1 * x2

    init_x = (4, 2)
    init_f = f(init_x[0], init_x[1])
    rhc = RandomizedHillClimbing(init_x, init_f, size)

    evaluated_xs = {init_x}
    steps = []

    random.seed(0)
    for _ in range(30):
        neighbor_x = rhc.get_neighbor()
        neighbor_f = f(neighbor_x[0], neighbor_x[1])
        step = rhc.update(neighbor_x, neighbor_f)
        
        steps.append(step)
        evaluated_xs.add(neighbor_x)

    def visualize_path(evaluated_xs, steps):
        """"""
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-.5, size - .5)
        ax.set_ylim(-.5, size - .5)

        for i in range(size):
            for j in range(size):
                if (i, j) in evaluated_xs:
                    ax.text(i, j, '%.1f' % (f(i, j)), ha='center', va='center',
                            bbox=dict(edgecolor='gray', facecolor='none',
                                      linewidth=2))
                else:
                    ax.text(i, j, '%.1f' % (f(i, j)), ha='center', va='center')

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.xaxis.set_minor_locator(
            ticker.FixedLocator(np.arange(-.5, size - .5, 1)))
        ax.yaxis.set_minor_locator(
            ticker.FixedLocator(np.arange(-.5, size - .5, 1)))
        plt.tick_params(axis='both', which='both', bottom='off', top='off',
                        left='off', right='off', labelbottom='off',
                        labelleft='off')
        ax.grid(True, which='minor')
        ax.grid(False, which='minor')

        for step in steps:
            ax.annotate('', xy=step[1], xytext=step[0],
                        arrowprops=dict(shrink=0.2, width=2, lw=0))

    visualize_path(evaluated_xs, steps)


def ch04_04():
    """ """
    size = 5
    _x1, _x2 = np.meshgrid(np.arange(size), np.arange(size))
    x1, x2 = _x1.ravel(), _x2.ravel()
    def f(x1, x2): return 0.5 * x1 + x2 - 0.3 * x1 * x2

    init_x = (4, 2)
    init_f = f(init_x[0], init_x[1])
    sa = SimulatedAnnealing(init_x, init_f, size)

    evaluated_xs = {init_x}
    steps = []

    random.seed(0)
    for _ in range(30):
        neighbor_x = sa.get_neighbor()
        evaluated_xs.add(neighbor_x)
        neighbor_f = f(neighbor_x[0], neighbor_x[1])
        step = sa.update(neighbor_x, neighbor_f)
        steps.append(step)

    def visualize_path(evaluated_xs, steps):
        """"""
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-.5, size - .5)
        ax.set_ylim(-.5, size - .5)

        for i in range(size):
            for j in range(size):
                if (i, j) in evaluated_xs:
                    ax.text(i, j, '%.1f' % (f(i, j)), ha='center', va='center',
                            bbox=dict(edgecolor='gray', facecolor='none',
                                      linewidth=2))
                else:
                    ax.text(i, j, '%.1f' % (f(i, j)), ha='center', va='center')

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.xaxis.set_minor_locator(
            ticker.FixedLocator(np.arange(-.5, size - .5, 1)))
        ax.yaxis.set_minor_locator(
            ticker.FixedLocator(np.arange(-.5, size - .5, 1)))
        plt.tick_params(axis='both', which='both', bottom='off', top='off',
                        left='off', right='off', labelbottom='off',
                        labelleft='off')
        ax.grid(True, which='minor')
        ax.grid(False, which='minor')

        for step in steps:
            ax.annotate('', xy=step[1], xytext=step[0],
                        arrowprops=dict(shrink=0.2, width=2, lw=0))

    visualize_path(evaluated_xs, steps)
