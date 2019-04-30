from contextlib import contextmanager

import numpy as np
from PIL import Image

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None


COCO_PERSON_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]]


@contextmanager
def canvas(fig_file=None, show=True, **kwargs):
    if 'figsize' not in kwargs:
        # kwargs['figsize'] = (15, 8)
        kwargs['figsize'] = (10, 6)
    fig, ax = plt.subplots(**kwargs)

    yield ax

    fig.set_tight_layout(True)
    if fig_file:
        fig.savefig(fig_file, dpi=200)  # , bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


@contextmanager
def image_canvas(image, fig_file=None, show=True, dpi_factor=1.0, fig_width=10.0, **kwargs):
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


class KeypointPainter(object):
    def __init__(self, *,
                 skeleton=None,
                 xy_scale=1.0, highlight=None, highlight_invisible=False,
                 show_box=True, linewidth=2, markersize=3,
                 color_connections=False,
                 solid_threshold=0.5):
        self.skeleton = skeleton or COCO_PERSON_SKELETON
        self.xy_scale = xy_scale
        self.highlight = highlight
        self.highlight_invisible = highlight_invisible
        self.show_box = show_box
        self.linewidth = linewidth
        self.markersize = markersize
        self.color_connections = color_connections
        self.solid_threshold = solid_threshold

    def _draw_skeleton(self, ax, x, y, v, *, color=None):
        if not np.any(v > 0):
            return

        if self.skeleton is not None:
            for ci, connection in enumerate(np.array(self.skeleton) - 1):
                c = color
                if self.color_connections:
                    c = matplotlib.cm.get_cmap('tab20')(ci / len(self.skeleton))
                if np.all(v[connection] > 0):
                    ax.plot(x[connection], y[connection],
                            linewidth=self.linewidth, color=c,
                            linestyle='dashed', dash_capstyle='round')
                if np.all(v[connection] > self.solid_threshold):
                    ax.plot(x[connection], y[connection],
                            linewidth=self.linewidth, color=c, solid_capstyle='round')

        # highlight invisible keypoints
        inv_color = 'k' if self.highlight_invisible else color

        ax.plot(x[v > 0], y[v > 0],
                'o', markersize=self.markersize,
                markerfacecolor=color, markeredgecolor=inv_color, markeredgewidth=2)
        ax.plot(x[v > self.solid_threshold], y[v > self.solid_threshold],
                'o', markersize=self.markersize,
                markerfacecolor=color, markeredgecolor=color, markeredgewidth=2)

        if self.highlight is not None:
            v_highlight = v[self.highlight]
            ax.plot(x[self.highlight][v_highlight > 0],
                    y[self.highlight][v_highlight > 0],
                    'o', markersize=self.markersize*2, markeredgewidth=2,
                    markerfacecolor=color, markeredgecolor=color)

    @staticmethod
    def _draw_box(ax, x, y, v, color, score=None):
        if not np.any(v > 0):
            return

        # keypoint bounding box
        x1, x2 = np.min(x[v > 0]), np.max(x[v > 0])
        y1, y2 = np.min(y[v > 0]), np.max(y[v > 0])
        if x2 - x1 < 5.0:
            x1 -= 2.0
            x2 += 2.0
        if y2 - y1 < 5.0:
            y1 -= 2.0
            y2 += 2.0
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, color=color))

        if score:
            ax.text(x1, y1, '{:.4f}'.format(score), fontsize=8, color=color)

    @staticmethod
    def _draw_text(ax, x, y, v, text, color):
        if not np.any(v > 0):
            return

        # keypoint bounding box
        x1, x2 = np.min(x[v > 0]), np.max(x[v > 0])
        y1, y2 = np.min(y[v > 0]), np.max(y[v > 0])
        if x2 - x1 < 5.0:
            x1 -= 2.0
            x2 += 2.0
        if y2 - y1 < 5.0:
            y1 -= 2.0
            y2 += 2.0

        ax.text(x1 + 2, y1 - 2, text, fontsize=8,
                color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0})

    @staticmethod
    def _draw_scales(ax, xs, ys, vs, color, scales):
        for x, y, v, scale in zip(xs, ys, vs, scales):
            if v == 0.0:
                continue
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    (x - scale, y - scale), 2 * scale, 2 * scale, fill=False, color=color))

    def keypoints(self, ax, keypoint_sets, *, scores=None, color=None, colors=None, texts=None):
        if keypoint_sets is None:
            return

        if color is None and self.color_connections:
            color = 'white'
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

            self._draw_skeleton(ax, x, y, v, color=color)
            if self.show_box:
                score = scores[i] if scores is not None else None
                self._draw_box(ax, x, y, v, color, score)

                if texts is not None:
                    self._draw_text(ax, x, y, v, texts[i], color)


    def annotations(self, ax, annotations, *,
                    color=None, colors=None, texts=None):
        if annotations is None:
            return

        if color is None and self.color_connections:
            color = 'white'
        if color is None and colors is None:
            colors = range(len(annotations))

        for i, ann in enumerate(annotations):
            kps = ann.data
            assert kps.shape[1] == 3
            x = kps[:, 0] * self.xy_scale
            y = kps[:, 1] * self.xy_scale
            v = kps[:, 2]

            if colors is not None:
                color = colors[i]

            if isinstance(color, (int, np.integer)):
                color = matplotlib.cm.get_cmap('tab20')((color % 20 + 0.05) / 20)

            self._draw_skeleton(ax, x, y, v, color=color)

            if ann.joint_scales is not None:
                self._draw_scales(ax, x, y, v, color, ann.joint_scales)

            if self.show_box:
                self._draw_box(ax, x, y, v, color, ann.score())

                if texts is not None:
                    self._draw_text(ax, x, y, v, texts[i], color)


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
        u -= x
        v -= y

    for xx, yy, uu, vv, _, rr in zip(x, y, u, v, c, r):
        if not rr:
            continue
        circle = matplotlib.patches.Circle(
            (xx + uu, yy + vv), rr / 2.0, zorder=10, linewidth=1, alpha=0.5)
        ax.add_artist(circle)

    return ax.quiver(x[s], y[s], u[s], v[s], c[s],
                     angles='xy', scale_units='xy', scale=1, zOrder=10, **kwargs)


def arrows(ax, fourd, xy_scale=1.0, threshold=0.0, **kwargs):
    mask = np.min(fourd[:, 2], axis=0) >= threshold
    fourd = fourd[:, :, mask]
    (x1, y1), (x2, y2) = fourd[:, :2, :] * xy_scale
    c = np.min(fourd[:, 2], axis=0)
    s = np.argsort(c)
    return ax.quiver(x1[s], y1[s], (x2 - x1)[s], (y2 - y1)[s], c[s],
                     angles='xy', scale_units='xy', scale=1, zOrder=10, **kwargs)


def white_screen(ax, alpha=0.9):
    ax.add_patch(
        plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, alpha=alpha,
                      facecolor='white')
    )
