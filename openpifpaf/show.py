from contextlib import contextmanager

import numpy as np
from PIL import Image

try:
    import matplotlib
    import matplotlib.collections
    import matplotlib.pyplot as plt
    import matplotlib.patches
except ImportError:
    matplotlib = None
    plt = None


@contextmanager
def canvas(fig_file=None, show=True, dpi=200, **kwargs):
    if 'figsize' not in kwargs:
        # kwargs['figsize'] = (15, 8)
        kwargs['figsize'] = (10, 6)
    fig, ax = plt.subplots(**kwargs)

    yield ax

    fig.set_tight_layout(True)
    if fig_file:
        fig.savefig(fig_file, dpi=dpi)  # , bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


@contextmanager
def image_canvas(image, fig_file=None, show=True, dpi_factor=1.0, fig_width=10.0, **kwargs):
    image = np.asarray(image)
    if 'figsize' not in kwargs:
        kwargs['figsize'] = (fig_width, fig_width * image.shape[0] / image.shape[1])

    fig = plt.figure(**kwargs)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    fig.add_axes(ax)
    ax.imshow(image)

    yield ax

    if fig_file:
        fig.savefig(fig_file, dpi=image.shape[1] / kwargs['figsize'][0] * dpi_factor)
    if show:
        plt.show()
    plt.close(fig)


def load_image(path, scale=1.0):
    with open(path, 'rb') as f:
        image = Image.open(f).convert('RGB')
        image = np.asarray(image) * scale / 255.0
        return image


class CrowdPainter(object):
    def __init__(self, *, alpha=0.5, color='orange'):
        self.alpha = alpha
        self.color = color

    def draw(self, ax, outlines):
        for outline in outlines:
            assert outline.shape[1] == 2

        patches = []
        for outline in outlines:
            polygon = matplotlib.patches.Polygon(
                outline[:, :2], color=self.color, facecolor=self.color, alpha=self.alpha)
            patches.append(polygon)
        ax.add_collection(matplotlib.collections.PatchCollection(patches, match_original=True))


class KeypointPainter(object):
    show_box = False
    show_joint_confidences = False
    show_joint_scales = False
    show_decoding_order = False
    show_frontier_order = False
    show_only_decoded_connections = False

    def __init__(self, *,
                 xy_scale=1.0, highlight=None, highlight_invisible=False,
                 linewidth=2, markersize=3,
                 color_connections=False,
                 solid_threshold=0.5):
        self.xy_scale = xy_scale
        self.highlight = highlight
        self.highlight_invisible = highlight_invisible
        self.linewidth = linewidth
        self.markersize = markersize
        self.color_connections = color_connections
        self.solid_threshold = solid_threshold

    def _draw_skeleton(self, ax, x, y, v, *, skeleton, color=None, **kwargs):
        if not np.any(v > 0):
            return

        # connections
        lines, line_colors, line_styles = [], [], []
        for ci, (j1i, j2i) in enumerate(np.array(skeleton) - 1):
            c = color
            if self.color_connections:
                c = matplotlib.cm.get_cmap('tab20')(ci / len(skeleton))
            if v[j1i] > 0 and v[j2i] > 0:
                lines.append([(x[j1i], y[j1i]), (x[j2i], y[j2i])])
                line_colors.append(c)
                if v[j1i] > self.solid_threshold and v[j2i] > self.solid_threshold:
                    line_styles.append('solid')
                else:
                    line_styles.append('dashed')
        ax.add_collection(matplotlib.collections.LineCollection(
            lines, colors=line_colors,
            linewidths=kwargs.get('linewidth', self.linewidth),
            linestyles=kwargs.get('linestyle', line_styles),
            capstyle='round',
        ))

        # joints
        for xx, yy, vv in zip(x, y, v):
            if vv == 0.0:
                continue
            ax.add_artist(matplotlib.patches.Circle(
                (xx, yy), self.markersize / 2.0,
                color='white' if self.color_connections else color,
                edgecolor='k' if self.highlight_invisible else None,
                zorder=2,
            ))

        # highlight joints
        if self.highlight is not None:
            for xx, yy, vv in zip(x[self.highlight], y[self.highlight], v[self.highlight]):
                if vv == 0.0:
                    continue
                ax.add_artist(matplotlib.patches.Circle(
                    (xx, yy), self.markersize,
                    color=color,
                    edgecolor=color,
                    zorder=2,
                ))

    def keypoints(self, ax, keypoint_sets, *,
                  skeleton, scores=None, color=None, colors=None, texts=None):
        if keypoint_sets is None:
            return

        if color is None and colors is None:
            colors = range(len(keypoint_sets))

        for i, kps in enumerate(np.asarray(keypoint_sets)):
            assert kps.shape[1] == 3
            x = kps[:, 0] * self.xy_scale
            y = kps[:, 1] * self.xy_scale
            v = kps[:, 2]

            if colors is not None:
                color = colors[i]

            if isinstance(color, (int, np.integer)):
                color = matplotlib.cm.get_cmap('tab20')((color % 20 + 0.05) / 20)

            self._draw_skeleton(ax, x, y, v, skeleton=skeleton, color=color)
            if self.show_box:
                score = scores[i] if scores is not None else None
                self._draw_box(ax, x, y, v, color, score)

            if texts is not None:
                self._draw_text(ax, x, y, v, texts[i], color)

    @staticmethod
    def _draw_box(ax, x, y, w, h, color, score=None, linewidth=1):
        if w < 5.0:
            x -= 2.0
            w += 4.0
        if h < 5.0:
            y -= 2.0
            h += 4.0
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (x, y), w, h, fill=False, color=color, linewidth=linewidth))

        if score:
            ax.text(x, y - linewidth, '{:.4f}'.format(score), fontsize=8, color=color)

    @staticmethod
    def _draw_text(ax, x, y, v, text, color, *, subtext=None):
        if not np.any(v > 0):
            return

        coord_i = np.argmin(y[v > 0])
        ax.annotate(
            text,
            (x[v > 0][coord_i], y[v > 0][coord_i]),
            fontsize=8,
            xytext=(5.0, 5.0),
            textcoords='offset points',
            color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0},
        )
        if subtext is not None:
            ax.annotate(
                subtext,
                (x[v > 0][coord_i], y[v > 0][coord_i]),
                fontsize=5,
                xytext=(5.0, 18.0 + 3.0),
                textcoords='offset points',
                color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0},
            )

    @staticmethod
    def _draw_scales(ax, xs, ys, vs, color, scales):
        for x, y, v, scale in zip(xs, ys, vs, scales):
            if v == 0.0:
                continue
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    (x - scale, y - scale), 2 * scale, 2 * scale, fill=False, color=color))

    @staticmethod
    def _draw_joint_confidences(ax, xs, ys, vs, color):
        for x, y, v in zip(xs, ys, vs):
            if v == 0.0:
                continue
            ax.annotate(
                '{:.0%}'.format(v),
                (x, y),
                fontsize=6,
                xytext=(0.0, 0.0),
                textcoords='offset points',
                verticalalignment='top',
                color='white', bbox={'facecolor': color, 'alpha': 0.2, 'linewidth': 0, 'pad': 0.0},
            )

    def annotations(self, ax, annotations, *,
                    color=None, colors=None, texts=None, subtexts=None):
        if annotations is None:
            return

        if color is None and colors is None:
            colors = range(len(annotations))

        for i, ann in enumerate(annotations):
            if colors is not None:
                color = colors[i]

            text = texts[i] if texts is not None else None
            subtext = subtexts[i] if subtexts is not None else None
            self.annotation(ax, ann, color=color, text=text, subtext=subtext)

    def annotation(self, ax, ann, *, color, text=None, subtext=None):
        if isinstance(color, (int, np.integer)):
            color = matplotlib.cm.get_cmap('tab20')((color % 20 + 0.05) / 20)

        kps = ann.data
        assert kps.shape[1] == 3
        x = kps[:, 0] * self.xy_scale
        y = kps[:, 1] * self.xy_scale
        v = kps[:, 2]

        if self.show_frontier_order:
            frontier = set((s, e) for s, e in ann.frontier_order)
            frontier_skeleton_mask = [
                (s - 1, e - 1) in frontier or (e - 1, s - 1) in frontier
                for s, e in ann.skeleton
            ]
            frontier_skeleton = [se for se, m in zip(ann.skeleton, frontier_skeleton_mask) if m]
            self._draw_skeleton(ax, x, y, v, color='black', skeleton=frontier_skeleton,
                                linestyle='dotted', linewidth=1)

        skeleton = ann.skeleton
        if self.show_only_decoded_connections:
            decoded_connections = set((jsi, jti) for jsi, jti, _, __ in ann.decoding_order)
            skeleton_mask = [
                (s - 1, e - 1) in decoded_connections or (e - 1, s - 1) in decoded_connections
                for s, e in skeleton
            ]
            skeleton = [se for se, m in zip(skeleton, skeleton_mask) if m]

        self._draw_skeleton(ax, x, y, v, color=color, skeleton=skeleton)

        if self.show_joint_scales and ann.joint_scales is not None:
            self._draw_scales(ax, x, y, v, color, ann.joint_scales)

        if self.show_joint_confidences:
            self._draw_joint_confidences(ax, x, y, v, color)

        if self.show_box:
            x_, y_, w_, h_ = ann.bbox()
            self._draw_box(ax, x_, y_, w_, h_, color, ann.score())

        if text is not None:
            self._draw_text(ax, x, y, v, text, color, subtext=subtext)

        if self.show_decoding_order and hasattr(ann, 'decoding_order'):
            self._draw_decoding_order(ax, ann.decoding_order)

    @staticmethod
    def _draw_decoding_order(ax, decoding_order):
        for step_i, (jsi, jti, jsxyv, jtxyv) in enumerate(decoding_order):
            ax.plot([jsxyv[0], jtxyv[0]], [jsxyv[1], jtxyv[1]], '--', color='black')
            ax.text(0.5 * (jsxyv[0] + jtxyv[0]), 0.5 * (jsxyv[1] +jtxyv[1]),
                    '{}: {} -> {}'.format(step_i, jsi, jti), fontsize=8,
                    color='white', bbox={'facecolor': 'black', 'alpha': 0.5, 'linewidth': 0})


def quiver(ax, vector_field, intensity_field=None, step=1, threshold=0.5,
           xy_scale=1.0, uv_is_offset=False,
           reg_uncertainty=None, **kwargs):
    x, y, u, v, c, r = [], [], [], [], [], []
    for j in range(0, vector_field.shape[1], step):
        for i in range(0, vector_field.shape[2], step):
            if intensity_field is not None and intensity_field[j, i] < threshold:
                continue
            x.append(i * xy_scale)
            y.append(j * xy_scale)
            u.append(vector_field[0, j, i] * xy_scale)
            v.append(vector_field[1, j, i] * xy_scale)
            c.append(intensity_field[j, i] if intensity_field is not None else 1.0)
            r.append(reg_uncertainty[j, i] * xy_scale if reg_uncertainty is not None else None)
    x = np.array(x)
    y = np.array(y)
    u = np.array(u)
    v = np.array(v)
    c = np.array(c)
    r = np.array(r)
    s = np.argsort(c)
    if uv_is_offset:
        u += x
        v += y

    for uu, vv, rr in zip(u, v, r):
        if not rr:
            continue
        circle = matplotlib.patches.Circle(
            (uu, vv), rr / 2.0, zorder=11, linewidth=1, alpha=1.0,
            fill=False, color='orange')
        ax.add_artist(circle)

    return ax.quiver(x[s], y[s], u[s] - x[s], v[s] - y[s], c[s],
                     angles='xy', scale_units='xy', scale=1, zOrder=10, **kwargs)


def margins(ax, vector_field, intensity_field=None, step=1, threshold=0.5,
            xy_scale=1.0, uv_is_offset=False, **kwargs):
    x, y, u, v, r = [], [], [], [], []
    for j in range(0, vector_field.shape[1], step):
        for i in range(0, vector_field.shape[2], step):
            if intensity_field is not None and intensity_field[j, i] < threshold:
                continue
            x.append(i * xy_scale)
            y.append(j * xy_scale)
            u.append(vector_field[0, j, i] * xy_scale)
            v.append(vector_field[1, j, i] * xy_scale)
            r.append(vector_field[2:6, j, i] * xy_scale)
    x = np.array(x)
    y = np.array(y)
    u = np.array(u)
    v = np.array(v)
    r = np.array(r)
    if uv_is_offset:
        u -= x
        v -= y

    wedge_angles = [
        (0.0, 90.0),
        (90.0, 180.0),
        (270.0, 360.0),
        (180.0, 270.0),
    ]

    for xx, yy, uu, vv, rr in zip(x, y, u, v, r):
        for q_rr, (theta1, theta2) in zip(rr, wedge_angles):
            if not np.isfinite(q_rr):
                continue
            wedge = matplotlib.patches.Wedge(
                (xx + uu, yy + vv), q_rr, theta1, theta2,
                zorder=9, linewidth=1, alpha=0.5 / 16.0,
                fill=True, color='orange', **kwargs)
            ax.add_artist(wedge)


def arrows(ax, fourd, xy_scale=1.0, threshold=0.0, **kwargs):
    mask = np.min(fourd[:, 2], axis=0) >= threshold
    fourd = fourd[:, :, mask]
    (x1, y1), (x2, y2) = fourd[:, :2, :] * xy_scale
    c = np.min(fourd[:, 2], axis=0)
    s = np.argsort(c)
    return ax.quiver(x1[s], y1[s], (x2 - x1)[s], (y2 - y1)[s], c[s],
                     angles='xy', scale_units='xy', scale=1, zOrder=10, **kwargs)


def boxes(ax, scalar_field, *, intensity_field=None, regression_field=None,
          xy_scale=1.0, step=1, threshold=0.5,
          cmap='viridis_r', clim=(0.5, 1.0), **kwargs):
    x, y, s, c = [], [], [], []
    for j in range(0, scalar_field.shape[0], step):
        for i in range(0, scalar_field.shape[1], step):
            if intensity_field is not None and intensity_field[j, i] < threshold:
                continue
            x_offset, y_offset = 0.0, 0.0
            if regression_field is not None:
                x_offset = regression_field[0, j, i]
                y_offset = regression_field[1, j, i]
            x.append((i + x_offset) * xy_scale)
            y.append((j + y_offset) * xy_scale)
            s.append(scalar_field[j, i] * xy_scale)
            c.append(intensity_field[j, i] if intensity_field is not None else 1.0)

    cmap = matplotlib.cm.get_cmap(cmap)
    cnorm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    for xx, yy, ss, cc in zip(x, y, s, c):
        color = cmap(cnorm(cc))
        rectangle = matplotlib.patches.Rectangle(
            (xx - ss, yy - ss), ss * 2.0, ss * 2.0,
            color=color, zorder=10, linewidth=1, **kwargs)
        ax.add_artist(rectangle)


def circles(ax, scalar_field, intensity_field=None, xy_scale=1.0, step=1, threshold=0.5,
            cmap='viridis_r', clim=(0.5, 1.0), **kwargs):
    x, y, s, c = [], [], [], []
    for j in range(0, scalar_field.shape[0], step):
        for i in range(0, scalar_field.shape[1], step):
            if intensity_field is not None and intensity_field[j, i] < threshold:
                continue
            x.append(i * xy_scale)
            y.append(j * xy_scale)
            s.append(scalar_field[j, i] * xy_scale)
            c.append(intensity_field[j, i] if intensity_field is not None else 1.0)

    cmap = matplotlib.cm.get_cmap(cmap)
    cnorm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    for xx, yy, ss, cc in zip(x, y, s, c):
        color = cmap(cnorm(cc))
        circle = matplotlib.patches.Circle(
            (xx, yy), ss,
            color=color, zorder=10, linewidth=1, **kwargs)
        ax.add_artist(circle)


def white_screen(ax, alpha=0.9):
    ax.add_patch(
        plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, alpha=alpha,
                      facecolor='white')
    )


def cli(parser):
    group = parser.add_argument_group('show')
    group.add_argument('--show-box', default=False, action='store_true')
    group.add_argument('--show-joint-scales', default=False, action='store_true')
    group.add_argument('--show-joint-confidences', default=False, action='store_true')
    group.add_argument('--show-decoding-order', default=False, action='store_true')
    group.add_argument('--show-frontier-order', default=False, action='store_true')
    group.add_argument('--show-only-decoded-connections', default=False, action='store_true')


def configure(args):
    KeypointPainter.show_box = args.show_box
    KeypointPainter.show_joint_scales = args.show_joint_scales
    KeypointPainter.show_joint_confidences = args.show_joint_confidences
    KeypointPainter.show_decoding_order = args.show_decoding_order
    KeypointPainter.show_frontier_order = args.show_frontier_order
    KeypointPainter.show_only_decoded_connections = args.show_only_decoded_connections
