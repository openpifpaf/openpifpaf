import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    plt = None
    make_axes_locatable = None

from ..data import COCO_KEYPOINTS, COCO_PERSON_SKELETON
from .. import show


class Visualizer(object):
    def __init__(self, pif_indices, paf_indices, *,
                 file_prefix=None,
                 keypoints=None,
                 skeleton=None,
                 show_seed_confidence=False):
        self.keypoints = keypoints or COCO_KEYPOINTS
        self.skeleton = skeleton or COCO_PERSON_SKELETON

        self.pif_indices = self.process_indices(pif_indices)
        if self.pif_indices and self.pif_indices[0][0] == -1:
            self.pif_indices = [[i] for i, _ in enumerate(self.keypoints)]
        self.paf_indices = self.process_indices(paf_indices)
        if self.paf_indices and self.paf_indices[0][0] == -1:
            self.paf_indices = [[i] for i, _ in enumerate(self.skeleton)]
        self.file_prefix = file_prefix
        self.show_seed_confidence = show_seed_confidence

        self.image = None
        self.processed_image = None

    @staticmethod
    def process_indices(indices):
        return [[int(e) for e in i.split(',')] for i in indices]

    def set_image(self, image, processed_image):
        if image is None and processed_image is not None:
            # image not given, so recover an image of the correct shape
            image = np.moveaxis(np.asarray(processed_image), 0, -1)
            image = np.clip((image + 2.0) / 4.0, 0.0, 1.0)

        self.image = image
        self.processed_image = processed_image

    def resized_image(self, io_scale):
        resized_image = np.moveaxis(
            np.asarray(self.processed_image)[:, ::int(io_scale), ::int(io_scale)], 0, -1)
        return np.clip((resized_image + 2.0) / 4.0, 0.0, 1.0)

    def seeds(self, seeds, io_scale=1.0):
        print('seeds')
        field_indices = {f for _, f, __, ___, ____ in seeds}

        with show.image_canvas(self.image, fig_width=20.0) as ax:
            show.white_screen(ax)
            for f in field_indices:
                x = [xx * io_scale for _, ff, xx, __, ___ in seeds if ff == f]
                y = [yy * io_scale for _, ff, __, yy, ___ in seeds if ff == f]
                c = [cc for cc, ff, _, __, ___ in seeds if ff == f]
                ax.plot(x, y, 'o')
                if self.show_seed_confidence:
                    for xx, yy, cc in zip(x, y, c):
                        ax.text(xx, yy, '{:.2f}'.format(cc))

    @staticmethod
    def occupied(occ):
        occ = occ.copy()
        occ[occ > 0] = 1.0
        with show.canvas() as ax:
            ax.imshow(occ)

    def paf_refined(self, original_paf, refined_paf, io_scale):
        print('refined paf')
        for g in self.paf_indices:
            with show.canvas() as ax:
                ax.imshow(self.image)
                show.white_screen(ax)

                for f in g:
                    print('association field',
                          self.keypoints[self.skeleton[f][0] - 1],
                          self.keypoints[self.skeleton[f][1] - 1])

                    qr = show.arrows(ax,
                                     original_paf[f],
                                     threshold=0.5, width=0.001,
                                     cmap='Oranges', clim=(0.0, 1.0),
                                     xy_scale=io_scale, alpha=0.1)
                    qr = show.arrows(ax,
                                     refined_paf[f],
                                     threshold=0.5, width=0.001,
                                     cmap='Blues', clim=(0.0, 1.0),
                                     xy_scale=io_scale)

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='3%', pad=0.05)
                    plt.colorbar(qr, cax=cax)

                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

    def paf_raw(self, paf, io_scale, reg_components=2):
        print('raw paf')
        intensity_fields, reg1_fields, reg2_fields, reg1_fields_b, reg2_fields_b = paf
        for g in self.paf_indices:
            for f in g:
                print('association field',
                      self.keypoints[self.skeleton[f][0] - 1],
                      self.keypoints[self.skeleton[f][1] - 1])
                fig_file = self.file_prefix + '.paf{}.c.png'.format(f) if self.file_prefix else None
                with show.canvas(fig_file) as ax:
                    ax.imshow(self.resized_image(io_scale))
                    im = ax.imshow(intensity_fields[f], alpha=0.9,
                                   vmin=0.0, vmax=1.0, cmap='Oranges')

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='3%', pad=0.05)
                    plt.colorbar(im, cax=cax)

                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

        for g in self.paf_indices:
            fig_file = (
                self.file_prefix + '.paf{}.v.png'.format(''.join([str(f) for f in g]))
                if self.file_prefix else None
            )
            with show.canvas(fig_file) as ax:
                print('reg', reg_components)
                ax.imshow(self.image)
                show.white_screen(ax)

                for f in g:
                    print('association field',
                          self.keypoints[self.skeleton[f][0] - 1],
                          self.keypoints[self.skeleton[f][1] - 1])
                    q1 = show.quiver(ax, reg1_fields[f], intensity_fields[f],
                                     reg_uncertainty=reg1_fields_b[f],
                                     threshold=0.5, width=0.003, step=1,
                                     cmap='Blues', clim=(0.5, 1.0), xy_scale=io_scale)
                    show.quiver(ax, reg2_fields[f], intensity_fields[f],
                                reg_uncertainty=reg2_fields_b[f],
                                threshold=0.5, width=0.003, step=1,
                                cmap='Greens', clim=(0.5, 1.0), xy_scale=io_scale)

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='3%', pad=0.05)
                    plt.colorbar(q1, cax=cax)

                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

    def paf_connections(self, connections):
        for g in self.paf_indices:
            with show.canvas() as ax:
                ax.imshow(self.image)
                show.white_screen(ax)

                for f in g:
                    print('association field',
                          self.keypoints[self.skeleton[f][0] - 1],
                          self.keypoints[self.skeleton[f][1] - 1])
                    with show.canvas() as ax:
                        ax.imshow(self.resized_image)
                        show.white_screen(ax, alpha=0.9)
                        arrows = show.arrows(ax, connections[f],
                                             cmap='viridis_r', clim=(0.0, 1.0))

                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes('right', size='3%', pad=0.05)
                        plt.colorbar(arrows, cax=cax)

                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)

    def pif_raw(self, pif, io_scale):
        print('raw pif')
        intensity_fields, reg_fields, reg_fields_b, scale_fields = pif
        for g in self.pif_indices:
            for f in g:
                print('pif field', self.keypoints[f])
                fig_file = self.file_prefix + '.pif{}.c.png'.format(f) if self.file_prefix else None
                with show.canvas(fig_file, figsize=(8, 5)) as ax:
                    ax.imshow(self.resized_image(io_scale))
                    im = ax.imshow(intensity_fields[f], alpha=0.9,
                                   vmin=0.0, vmax=1.0, cmap='Oranges')

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='3%', pad=0.05)
                    plt.colorbar(im, cax=cax)

                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

        for g in self.pif_indices:
            for f in g:
                print('pif field', self.keypoints[f])
                fig_file = self.file_prefix + '.pif{}.v.png'.format(f) if self.file_prefix else None
                with show.canvas(fig_file) as ax:
                    ax.imshow(self.image)
                    show.white_screen(ax, alpha=0.5)
                    show.quiver(ax, reg_fields[f], intensity_fields[f],
                                reg_uncertainty=reg_fields_b[f],
                                cmap='viridis_r', clim=(0.5, 1.0),
                                threshold=0.5, xy_scale=io_scale)

                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

        for g in self.pif_indices:
            for f in g:
                print('pif field', self.keypoints[f])
                fig_file = self.file_prefix + '.pif{}.s.png'.format(f) if self.file_prefix else None
                with show.canvas(fig_file) as ax:
                    ax.imshow(self.image)
                    show.white_screen(ax, alpha=0.5)
                    show.circles(ax, scale_fields[f], intensity_fields[f],
                                 xy_scale=io_scale, fill=False)

                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

    def pifhr(self, pifhr):
        print('pifhr')
        for g in self.pif_indices:
            for f in g:
                fig_file = (
                    self.file_prefix + '.pif{}.hr.png'.format(f)
                    if self.file_prefix else None
                )
                with show.canvas(fig_file, figsize=(8, 5)) as ax:
                    ax.imshow(self.image)
                    o = ax.imshow(pifhr[f], alpha=0.9, vmin=0.0, vmax=1.0, cmap='Oranges')

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='3%', pad=0.05)
                    plt.colorbar(o, cax=cax)

                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.set_xlim(0, self.image.shape[1])
                    ax.set_ylim(self.image.shape[0], 0)
