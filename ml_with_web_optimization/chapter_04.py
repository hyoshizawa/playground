"""

"""
import numpy as np
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
        reuturn (old_f, self.current_x)


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


    
