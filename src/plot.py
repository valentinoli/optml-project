import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import numpy as np

from .mixture import GaussianMixture
from .helpers import random_ball

def plot_points_in_ball(num_points: int = 1000, radius: int = 1, dim: int = 2):
    """Plot uniformly random points from ball of a given radius in 2d or 3d"""
    if dim < 2 or dim > 3:
        raise ValueError('Invalid dimension')

    points = random_ball(num_points, dim, radius=radius)

    subplot_kw = {}

    if dim == 3:
        subplot_kw=dict(projection='3d')

    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    if dim == 2:
        ax.set_aspect('equal')
        patch = Circle((0, 0), radius, fill=False, ls='-', lw=0.25)
        ax.add_patch(patch)
        ax.axvline(c='grey', lw=0.5)
        ax.axhline(c='grey', lw=0.5)

    ax.scatter(*np.split(points, dim, axis=1), marker='.')
    plt.show()
    

def plot_scatter(ax, points, marker = 'o', c = None, s = None):
    """Plot a scatter of points on the given axis"""
    ax.scatter(*np.split(points, 2, axis=1), c=c, s=s, marker=marker)

def plot_gmm_initializations_2d():
    """Plot sampled points for the experiment and different means initialization modes for d=2"""
    gmm = GaussianMixture(d=2)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    
    # plot x and y axes
    ax.axvline(c='grey', lw=0.5)
    ax.axhline(c='grey', lw=0.5)
    
    # plot ball of radius R
    patch = Circle((0, 0), radius=gmm.R, fill=False, ls='-', lw=0.25)
    ax.add_patch(patch)
    
    # plot points
    plot_scatter(ax, gmm.points, c='b', s=100)
    
    # plot two types of mean initializations
    plot_scatter(ax, gmm.init_params(from_data=False), c='g', s=100)
    plot_scatter(ax, gmm.init_params(from_data=True), c='r', s=30)
    
    ax.set_title('Mean parameter initialization schemes for GMM (d=2)')
    
    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Points', markerfacecolor='b', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Random initialization', markerfacecolor='g', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Initialization from data', markerfacecolor='r', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='best')
