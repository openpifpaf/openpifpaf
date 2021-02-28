from contextlib import contextmanager
import logging
import os

import numpy as np

try:
    import matplotlib.pyplot as plt  # pylint: disable=import-error
except ModuleNotFoundError as err:
    if err.name != 'matplotlib':
        raise err
    plt = None


LOG = logging.getLogger(__name__)


class Canvas:
    """Canvas for plotting.

    All methods expose Axes objects. To get Figure objects, you can ask the axis
    `ax.get_figure()`.
    """

    all_images_directory = None
    all_images_count = 0
    show = False
    image_width = 7.0
    image_height = None
    blank_dpi = 200
    image_dpi_factor = 1.0
    image_min_dpi = 50.0
    out_file_extension = 'png'
    white_overlay = False

    @classmethod
    def generic_name(cls):
        if cls.all_images_directory is None:
            return None

        os.makedirs(cls.all_images_directory, exist_ok=True)

        cls.all_images_count += 1
        return os.path.join(cls.all_images_directory,
                            '{:04}.{}'.format(cls.all_images_count, cls.out_file_extension))

    @classmethod
    @contextmanager
    def blank(cls, fig_file=None, *, dpi=None, nomargin=False, **kwargs):
        if plt is None:
            raise Exception('please install matplotlib')
        if fig_file is None:
            fig_file = cls.generic_name()
        if dpi is None:
            dpi = cls.blank_dpi

        if 'figsize' not in kwargs:
            kwargs['figsize'] = (10, 6)

        if nomargin:
            if 'gridspec_kw' not in kwargs:
                kwargs['gridspec_kw'] = {}
            kwargs['gridspec_kw']['wspace'] = 0
            kwargs['gridspec_kw']['hspace'] = 0
        fig, ax = plt.subplots(dpi=dpi, **kwargs)

        yield ax

        fig.set_tight_layout(not nomargin)
        if fig_file:
            LOG.debug('writing image to %s', fig_file)
            fig.savefig(fig_file)
        if cls.show:
            plt.show()
        plt.close(fig)

    @classmethod
    @contextmanager
    def image(cls, image, fig_file=None, *, margin=None, **kwargs):
        if plt is None:
            raise Exception('please install matplotlib')
        if fig_file is None:
            fig_file = cls.generic_name()

        image = np.asarray(image)

        if margin is None:
            margin = [0.0, 0.0, 0.0, 0.0]
        elif isinstance(margin, float):
            margin = [margin, margin, margin, margin]
        assert len(margin) == 4

        if 'figsize' not in kwargs:
            # compute figure size: use image ratio and take the drawable area
            # into account that is left after subtracting margins.
            image_ratio = image.shape[0] / image.shape[1]
            image_area_ratio = (1.0 - margin[1] - margin[3]) / (1.0 - margin[0] - margin[2])
            if cls.image_width is not None:
                kwargs['figsize'] = (
                    cls.image_width,
                    cls.image_width * image_ratio / image_area_ratio
                )
            elif cls.image_height:
                kwargs['figsize'] = (
                    cls.image_height * image_area_ratio / image_ratio,
                    cls.image_height
                )

        dpi = max(cls.image_min_dpi, image.shape[1] / kwargs['figsize'][0] * cls.image_dpi_factor)
        fig = plt.figure(dpi=dpi, **kwargs)
        ax = plt.Axes(fig, [0.0 + margin[0],
                            0.0 + margin[1],
                            1.0 - margin[2],
                            1.0 - margin[3]])
        ax.set_axis_off()
        ax.set_xlim(-0.5, image.shape[1] - 0.5)  # imshow uses center-pixel-coordinates
        ax.set_ylim(image.shape[0] - 0.5, -0.5)
        fig.add_axes(ax)
        ax.imshow(image)
        if cls.white_overlay:
            white_screen(ax, cls.white_overlay)

        yield ax

        if fig_file:
            LOG.debug('writing image to %s', fig_file)
            fig.savefig(fig_file)
        if cls.show:
            plt.show()
        plt.close(fig)

    @classmethod
    @contextmanager
    def annotation(cls, ann, *,
                   filename=None,
                   margin=0.5,
                   fig_w=None,
                   fig_h=5.0,
                   **kwargs):
        bbox = ann.bbox()
        xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
        ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
        if fig_w is None:
            fig_w = fig_h / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])
        fig_w *= kwargs.get('ncols', 1)
        fig_h *= kwargs.get('nrows', 1)

        with cls.blank(filename, figsize=(fig_w, fig_h), nomargin=True, **kwargs) as ax:
            iter_ax = [ax] if hasattr(ax, 'set_axis_off') else ax
            for ax_ in iter_ax:
                ax_.set_axis_off()
                ax_.set_xlim(*xlim)
                ax_.set_ylim(*ylim)

            yield ax


# keep previous interface for now:
canvas = Canvas.blank
image_canvas = Canvas.image
annotation_canvas = Canvas.annotation


def white_screen(ax, alpha=0.9):
    ax.add_patch(
        plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, alpha=alpha,
                      facecolor='white')
    )
