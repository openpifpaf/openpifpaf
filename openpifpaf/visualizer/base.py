from contextlib import contextmanager
import logging
from typing import List

import numpy as np

from .. import annotation, show

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    plt = None
    make_axes_locatable = None

LOG = logging.getLogger(__name__)
IMAGENOTGIVEN = object()


def itemsetter(list_, index, value):
    list_[index] = value
    return list_


class Base:
    all_indices = []
    common_ax = None
    processed_image_intensity_spread = 2.0

    _image = None
    _processed_image = None
    _image_meta = None
    _ground_truth: List[annotation.Base] = None

    def __init__(self, head_name):
        self.head_name = head_name
        self._ax = None

        LOG.debug('%s: indices = %s', head_name, self.indices())

    @classmethod
    def image(cls, image=IMAGENOTGIVEN, meta=None):
        if image is IMAGENOTGIVEN:  # getter
            if callable(Base._image):  # check whether stored value is lazy
                Base._image = Base._image()  # pylint: disable=not-callable
            return Base._image

        if image is None:
            Base._image = None
            Base._image_meta = None
            return cls

        Base._image = lambda: np.asarray(image)
        Base._image_meta = meta
        return cls

    @classmethod
    def processed_image(cls, image=IMAGENOTGIVEN):
        if image is IMAGENOTGIVEN:  # getter
            if callable(Base._processed_image):  # check whether stored value is lazy
                Base._processed_image = Base._processed_image()  # pylint: disable=not-callable
            return Base._processed_image

        if image is None:
            Base._processed_image = None
            return cls

        def process_image(image):
            image = np.moveaxis(np.asarray(image), 0, -1)
            image = np.clip(image / cls.processed_image_intensity_spread * 0.5 + 0.5, 0.0, 1.0)
            return image

        Base._processed_image = lambda: process_image(image)
        return cls

    @staticmethod
    def ground_truth(ground_truth):
        Base._ground_truth = ground_truth

    @staticmethod
    def reset():
        Base._image = None
        Base._image_meta = None
        Base._processed_image = None
        Base._ground_truth = None

    @classmethod
    def normalized_index(cls, data):
        if isinstance(data, str):
            data = data.split(':')

        # unpack comma separation
        for di, d in enumerate(data):
            if ',' not in d:
                continue
            multiple = [cls.normalized_index(itemsetter(data, di, unpacked))
                        for unpacked in d.split(',')]
            # flatten before return
            return [item for items in multiple for item in items]

        if len(data) >= 2 and len(data[1]) == 0:
            data[1] = -1

        if len(data) == 3:
            return [(data[0], int(data[1]), data[2])]
        if len(data) == 2:
            return [(data[0], int(data[1]), 'all')]
        return [(data[0], -1, 'all')]

    @classmethod
    def set_all_indices(cls, all_indices):
        cls.all_indices = [d for dd in all_indices for d in cls.normalized_index(dd)]

    def indices(self, type_=None, with_all=True):
        head_names = self.head_name
        if not isinstance(head_names, (tuple, list)):
            head_names = (head_names,)
        return [
            f for hn, f, r_type in self.all_indices
            if hn in head_names and (
                type_ is None
                or (with_all and r_type == 'all')
                or type_ == r_type
            )
        ]

    @staticmethod
    def colorbar(ax, colored_element, size='3%', pad=0.01):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=size, pad=pad)
        cb = plt.colorbar(colored_element, cax=cax)
        cb.outline.set_linewidth(0.1)

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
        return field[half_stride:-half_stride + 1, half_stride:-half_stride + 1]
