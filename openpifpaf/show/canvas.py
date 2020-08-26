from contextlib import contextmanager
import logging

import numpy as np
from PIL import Image

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


LOG = logging.getLogger(__name__)


@contextmanager
def canvas(fig_file=None, show=True, dpi=200, nomargin=False, **kwargs):
    if plt is None:
        raise Exception('please install matplotlib')

    if 'figsize' not in kwargs:
        # kwargs['figsize'] = (15, 8)
        kwargs['figsize'] = (10, 6)

    if nomargin:
        fig = plt.figure(dpi=dpi, **kwargs)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        fig.add_axes(ax)
    else:
        fig, ax = plt.subplots(dpi=dpi, **kwargs)

    yield ax

    fig.set_tight_layout(not nomargin)
    if fig_file:
        fig.savefig(fig_file)
    if show:
        plt.show()
    plt.close(fig)


@contextmanager
def image_canvas(image, fig_file=None, show=True, dpi_factor=1.0, fig_width=10.0, **kwargs):
    if plt is None:
        raise Exception('please install matplotlib')

    image = np.asarray(image)
    if 'figsize' not in kwargs:
        kwargs['figsize'] = (fig_width, fig_width * image.shape[0] / image.shape[1])

    fig = plt.figure(dpi=image.shape[1] / kwargs['figsize'][0] * dpi_factor, **kwargs)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.set_xlim(-0.5, image.shape[1] - 0.5)  # imshow uses center-pixel-coordinates
    ax.set_ylim(image.shape[0] - 0.5, -0.5)
    fig.add_axes(ax)
    ax.imshow(image)

    yield ax

    if fig_file:
        fig.savefig(fig_file)
    if show:
        plt.show()
    plt.close(fig)


def load_image(path, scale=1.0):
    with open(path, 'rb') as f:
        image = Image.open(f).convert('RGB')
        image = np.asarray(image) * scale / 255.0
        return image


def white_screen(ax, alpha=0.9):
    ax.add_patch(
        plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, alpha=alpha,
                      facecolor='white')
    )
