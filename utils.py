import matplotlib

matplotlib.use('TkAgg')  # required for MacOS
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plotv(q, actions):
    pr = range(20)
    dr = range(10)
    v = []
    for p in pr:
        for d in dr:
            v.append([int(p + 1), int(d + 1), np.max([q[p, d, a] for a in actions])])
    v = np.array(v)

    # plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(v[:, 1], v[:, 0], v[:, 2], cmap=plt.cm.coolwarm, linewidth=0.2)
    ax.set_xlabel('Dealer')
    ax.set_ylabel('Player')
    ax.set_zlabel('Value')
    plt.show()


def plotLearningCurve(learning):
    sns.lineplot(x=learning[:, 1], y=learning[:, 2], hue=learning[:, 0], legend='full')
    plt.xlabel('Episodes')
    plt.ylabel('MSE')
    plt.title('MSE by lambda as compared to MC Q')
    plt.show()


def plotMSE(data, lambdas):
    sns.pointplot(x=lambdas, y=data, fmt='%.1f')
    plt.xlabel('Lambda')
    plt.ylabel('MSE')
    plt.show()
