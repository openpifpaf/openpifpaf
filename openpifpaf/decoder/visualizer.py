from contextlib import contextmanager
import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    plt = None
    make_axes_locatable = None

from .. import show


class Visualizer(object):
    pif_indices = [[]]
    paf_indices = [[]]
    occupied_indices = []
    file_prefix = None

    show_occupied = False
    show_seeds = False
    show_seed_confidences = False
    show_pifhr = False
    show_pif_c = False
    show_pif_v = False
    show_pif_s = False
    show_paf_c = False
    show_paf_v = False
    show_paf_s = False

    def __init__(self, *, keypoints, skeleton):
        self.keypoints = keypoints
        self.skeleton = skeleton

        if self.pif_indices and self.pif_indices[0][0] == -1:
            self.pif_indices = [[i] for i, _ in enumerate(self.keypoints)]
        if self.paf_indices and self.paf_indices[0][0] == -1:
            self.paf_indices = [[i] for i, _ in enumerate(self.skeleton)]

        self.image = None
        self.processed_image = None
        self.debug_ax = None

    def set_image(self, image, processed_image, *, debug_ax=None):
        if image is None and processed_image is not None:
            # image not given, so recover an image of the correct shape
            image = np.moveaxis(np.asarray(processed_image), 0, -1)
            image = np.clip((image + 2.0) / 4.0, 0.0, 1.0)

        self.image = np.asarray(image)
        self.processed_image = np.asarray(processed_image)
        self.debug_ax = debug_ax

    @contextmanager
    def image_canvas(self, *args, **kwargs):
        if self.debug_ax is not None:
            yield self.debug_ax
            return

        with show.image_canvas(*args, **kwargs) as ax:
            yield ax

    @contextmanager
    def canvas(self, *args, **kwargs):
        if self.debug_ax is not None:
            yield self.debug_ax
            return

        with show.canvas(*args, **kwargs) as ax:
            yield ax

    def resized_image(self, io_scale):
        resized_image = np.moveaxis(
            self.processed_image[:, ::int(io_scale), ::int(io_scale)], 0, -1)
        return np.clip((resized_image + 2.0) / 4.0, 0.0, 1.0)

    def seeds(self, seeds, io_scale=1.0):
        if not self.show_seeds:
            return

        print('seeds')
        field_indices = {f for _, f, __, ___, ____ in seeds}

        with self.image_canvas(self.image, fig_width=20.0) as ax:
            show.white_screen(ax)
            for f in field_indices:
                x = [xx * io_scale for _, ff, xx, __, ___ in seeds if ff == f]
                y = [yy * io_scale for _, ff, __, yy, ___ in seeds if ff == f]
                c = [cc for cc, ff, _, __, ___ in seeds if ff == f]
                ax.plot(x, y, 'o')
                if self.show_seed_confidences:
                    for xx, yy, cc in zip(x, y, c):
                        ax.text(xx, yy, '{:.2f}'.format(cc))

    def occupied(self, occupied):
        if not self.show_occupied:
            return

        print('occupied field')
        for g in self.occupied_indices:
            for f in g:
                occ = occupied.occupancy[f].copy()
                occ[occ > 0] = 1.0
                with self.canvas() as ax:
                    ax.imshow(occ)

    def paf_refined(self, original_paf, refined_paf, io_scale):
        print('refined paf')
        for g in self.paf_indices:
            with self.canvas() as ax:
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

    def paf_c(self, paf, io_scale):
        if not self.show_paf_c:
            return

        for g in self.paf_indices:
            for f in g:
                print('association field',
                      self.keypoints[self.skeleton[f][0] - 1],
                      self.keypoints[self.skeleton[f][1] - 1])
                fig_file = self.file_prefix + '.paf{}.c.png'.format(f) if self.file_prefix else None
                with self.canvas(fig_file) as ax:
                    ax.imshow(self.resized_image(io_scale))
                    im = ax.imshow(paf[f, 0], alpha=0.9,
                                   vmin=0.0, vmax=1.0, cmap='Oranges')

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='3%', pad=0.05)
                    plt.colorbar(im, cax=cax)

                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

    def paf_v(self, paf, io_scale):
        if not self.show_paf_v:
            return

        for g in self.paf_indices:
            fig_file = (
                self.file_prefix + '.paf{}.v.png'.format(''.join([str(f) for f in g]))
                if self.file_prefix else None
            )
            with self.canvas(fig_file) as ax:
                ax.imshow(self.image)
                show.white_screen(ax)

                for f in g:
                    print('association field',
                          self.keypoints[self.skeleton[f][0] - 1],
                          self.keypoints[self.skeleton[f][1] - 1])
                    q1 = show.quiver(ax, paf[f, 1:3], paf[f, 0],
                                     # reg_uncertainty=reg1_fields_b[f],
                                     threshold=0.5, width=0.003, step=1,
                                     cmap='Blues', clim=(0.0, 1.0), xy_scale=io_scale)
                    show.quiver(ax, paf[f, 5:7], paf[f, 0],
                                # reg_uncertainty=reg2_fields_b[f],
                                threshold=0.5, width=0.003, step=1,
                                cmap='Greens', clim=(0.0, 1.0), xy_scale=io_scale)

                    show.boxes(ax, paf[f, 4],
                               intensity_field=paf[f, 0],
                               regression_field=paf[f, 1:3],
                               cmap='Blues',
                               threshold=0.5,
                               clim=(0.0, 1.0),
                               xy_scale=io_scale, fill=False)
                    show.boxes(ax, paf[f, 8],
                               intensity_field=paf[f, 0],
                               regression_field=paf[f, 5:7],
                               cmap='Greens',
                               threshold=0.5,
                               clim=(0.0, 1.0),
                               xy_scale=io_scale, fill=False)

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='3%', pad=0.05)
                    plt.colorbar(q1, cax=cax)

                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

    def paf_s(self, paf, io_scale):
        if not self.show_paf_s:
            return

        for g in self.paf_indices:
            for f in g:
                if f >= len(paf):
                    continue
                print('association field',
                      self.keypoints[self.skeleton[f][0] - 1],
                      self.keypoints[self.skeleton[f][1] - 1])
                fig_file = self.file_prefix + '.paf{}.s.png'.format(f) if self.file_prefix else None
                with self.canvas(fig_file) as ax:
                    ax.imshow(self.image)
                    show.white_screen(ax, alpha=0.5)
                    show.boxes(ax, paf[f, 4],
                               intensity_field=paf[f, 0],
                               regression_field=paf[f, 1:3],
                               xy_scale=io_scale, fill=False)
                    show.boxes(ax, paf[f, 8],
                               intensity_field=paf[f, 0],
                               regression_field=paf[f, 5:7],
                               xy_scale=io_scale, fill=False)

                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

    def paf_raw(self, paf, io_scale):
        self.paf_c(paf, io_scale)
        self.paf_v(paf, io_scale)
        self.paf_s(paf, io_scale)

    def paf_connections(self, connections):
        for g in self.paf_indices:
            with self.canvas() as ax:
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

    def pif_c(self, pif, io_scale):
        if not self.show_pif_c:
            return

        print('pif c')
        for g in self.pif_indices:
            for f in g:
                if f >= len(pif):
                    continue
                print('pif field', self.keypoints[f])
                fig_file = self.file_prefix + '.pif{}.c.png'.format(f) if self.file_prefix else None
                with self.canvas(fig_file, figsize=(8, 5)) as ax:
                    ax.imshow(self.resized_image(io_scale))
                    im = ax.imshow(pif[f, 0], alpha=0.9,
                                   vmin=0.0, vmax=1.0, cmap='Oranges')

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='3%', pad=0.05)
                    plt.colorbar(im, cax=cax)

                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

    def pif_v(self, pif, io_scale):
        if not self.show_pif_v:
            return

        for g in self.pif_indices:
            for f in g:
                if f >= len(pif):
                    continue
                print('pif field', self.keypoints[f])
                fig_file = self.file_prefix + '.pif{}.v.png'.format(f) if self.file_prefix else None
                with self.canvas(fig_file) as ax:
                    ax.imshow(self.image)
                    show.white_screen(ax, alpha=0.5)
                    q = show.quiver(ax, pif[f, 1:3], pif[f, 0],
                                    # reg_uncertainty=pif[f, 3],
                                    cmap='viridis_r', clim=(0.5, 1.0),
                                    threshold=0.5, xy_scale=io_scale, width=0.001)

                    show.boxes(ax, pif[f, 4],
                               intensity_field=pif[f, 0],
                               regression_field=pif[f, 1:3],
                               clim=(0.5, 1.0),
                               xy_scale=io_scale, fill=False)

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='3%', pad=0.05)
                    plt.colorbar(q, cax=cax)

                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

    def pif_s(self, pif, io_scale):
        if not self.show_pif_s:
            return

        for g in self.pif_indices:
            for f in g:
                if f >= len(pif):
                    continue
                print('pif field', self.keypoints[f])
                fig_file = self.file_prefix + '.pif{}.s.png'.format(f) if self.file_prefix else None
                with self.canvas(fig_file) as ax:
                    ax.imshow(self.image)
                    show.white_screen(ax, alpha=0.5)
                    show.boxes(ax, pif[f, 4],
                               intensity_field=pif[f, 0],
                               regression_field=pif[f, 1:3],
                               xy_scale=io_scale, fill=False)

                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

    def pif_raw(self, pif, io_scale):
        self.pif_c(pif, io_scale)
        self.pif_v(pif, io_scale)
        self.pif_s(pif, io_scale)

    def pifhr(self, pifhr):
        if not self.show_pifhr:
            return

        print('pifhr')
        for g in self.pif_indices:
            for f in g:
                fig_file = (
                    self.file_prefix + '.pif{}.hr.png'.format(f)
                    if self.file_prefix else None
                )
                with self.canvas(fig_file, figsize=(8, 5)) as ax:
                    ax.imshow(self.image)
                    o = ax.imshow(pifhr[f], alpha=0.9, vmin=0.0, vmax=1.0, cmap='Oranges')

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='3%', pad=0.05)
                    plt.colorbar(o, cax=cax)

                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.set_xlim(0, self.image.shape[1])
                    ax.set_ylim(self.image.shape[0], 0)


def cli(parser):
    group = parser.add_argument_group('pose visualizer')
    group.add_argument('--debug-seeds', default=False, action='store_true')
    group.add_argument('--debug-pifhr', default=False, action='store_true')
    group.add_argument('--debug-pif-c', default=False, action='store_true')
    group.add_argument('--debug-pif-v', default=False, action='store_true')
    group.add_argument('--debug-pif-s', default=False, action='store_true')
    group.add_argument('--debug-paf-c', default=False, action='store_true')
    group.add_argument('--debug-paf-v', default=False, action='store_true')
    group.add_argument('--debug-paf-s', default=False, action='store_true')

    group.add_argument('--debug-pif-indices', default=[], nargs='+',
                       help=('indices of PIF fields to create debug plots for '
                             '(group with comma, e.g. "0,1 2" to create one plot '
                             'with field 0 and 1 and another plot with field 2)'))
    group.add_argument('--debug-paf-indices', default=[], nargs='+',
                       help=('indices of PAF fields to create debug plots for '
                             '(same grouping behavior as debug-pif-indices)'))
    group.add_argument('--debug-file-prefix', default=None,
                       help='save debug plots with this prefix')


def configure(args):
    # process debug indices
    args.debug_pif_indices = [[int(e) for e in i.split(',')] for i in args.debug_pif_indices]
    args.debug_paf_indices = [[int(e) for e in i.split(',')] for i in args.debug_paf_indices]

    Visualizer.pif_indices = args.debug_pif_indices
    Visualizer.paf_indices = args.debug_paf_indices
    Visualizer.occupied_indices = args.debug_pif_indices
    Visualizer.file_prefix = args.debug_file_prefix

    Visualizer.show_seeds = args.debug_seeds
    Visualizer.show_pifhr = args.debug_pifhr
    Visualizer.show_pif_c = args.debug_pif_c
    Visualizer.show_pif_v = args.debug_pif_v
    Visualizer.show_pif_s = args.debug_pif_s
    Visualizer.show_paf_c = args.debug_paf_c
    Visualizer.show_paf_v = args.debug_paf_v
    Visualizer.show_paf_s = args.debug_paf_s
