from contextlib import contextmanager
import logging
import numpy as np

from .. import show

try:
    import matplotlib.cm
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    matplotlib.cm.get_cmap('Oranges').set_bad('white', alpha=0.5)
except ImportError:
    plt = None
    make_axes_locatable = None

LOG = logging.getLogger(__name__)


class BaseVisualizer:
    common_ax = None
    _image = None
    _processed_image = None

    def __init__(self):
        self._ax = self.common_ax

    def image(self, image):
        self._image = np.asarray(image)
        return self

    def processed_image(self, image):
        image = np.moveaxis(np.asarray(image), 0, -1)
        image = np.clip(image * 0.25 + 0.5, 0.0, 1.0)
        self._processed_image = image
        return self

    def reset(self):
        self._image = None
        self._processed_image = None
        return self

    @staticmethod
    def colorbar(ax, colored_element, size='3%', pad=0.05):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=size, pad=pad)
        plt.colorbar(colored_element, cax=cax)

    @contextmanager
    def image_canvas(self, image, *args, **kwargs):
        if self._ax is not None:
            self._ax.set_axis_off()
            self._ax.imshow(np.asarray(image))
            yield self._ax
            return

        with show.image_canvas(image, *args, **kwargs) as ax:
            yield ax

    @contextmanager
    def canvas(self, *args, **kwargs):
        if self._ax is not None:
            yield self._ax
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
