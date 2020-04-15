from contextlib import contextmanager
import logging
import numpy as np

from .. import show

try:
    import matplotlib.cm
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    matplotlib.cm.get_cmap('Oranges').set_bad('white', alpha=0.5)
    matplotlib.cm.get_cmap('Blues').set_bad('white', alpha=0.5)
    matplotlib.cm.get_cmap('Greens').set_bad('white', alpha=0.5)
except ImportError:
    plt = None
    make_axes_locatable = None

LOG = logging.getLogger(__name__)


class BaseVisualizer:
    all_indices = []
    common_ax = None
    _image = None
    _processed_image = None

    def __init__(self, head_name):
        self.head_name = head_name
        self._ax = None

        LOG.debug('%s: indices = %s', head_name, self.indices)

    @staticmethod
    def image(image):
        if image is None:
            BaseVisualizer._image = None
            return

        BaseVisualizer._image = np.asarray(image)

    @staticmethod
    def processed_image(image):
        if image is None:
            BaseVisualizer._processed_image = None
            return

        image = np.moveaxis(np.asarray(image), 0, -1)
        image = np.clip(image * 0.25 + 0.5, 0.0, 1.0)
        BaseVisualizer._processed_image = image

    @staticmethod
    def reset():
        BaseVisualizer._image = None
        BaseVisualizer._processed_image = None

    @property
    def indices(self):
        head_names = self.head_name
        if not isinstance(head_names, (tuple, list)):
            head_names = (head_names,)
        return [f for hn, f in self.all_indices if hn in head_names]

    @staticmethod
    def colorbar(ax, colored_element, size='3%', pad=0.05):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=size, pad=pad)
        plt.colorbar(colored_element, cax=cax)

    @contextmanager
    def image_canvas(self, image, *args, **kwargs):
        ax = self._ax or self.common_ax
        if ax is not None:
            ax.set_axis_off()
            ax.imshow(np.asarray(image))
            yield ax
            return

        with show.image_canvas(image, *args, **kwargs) as ax:
            yield ax

    @contextmanager
    def canvas(self, *args, **kwargs):
        ax = self._ax or self.common_ax
        if ax is not None:
            yield ax
            return

        with show.canvas(*args, **kwargs) as ax:
            yield ax

    @staticmethod
    def scale_scalar(field, stride):
        field = np.repeat(field, stride, 0)
        field = np.repeat(field, stride, 1)

        # center (the result is technically still off by half a pixel)
        half_stride = stride // 2
        return field[half_stride:-half_stride+1, half_stride:-half_stride+1]
