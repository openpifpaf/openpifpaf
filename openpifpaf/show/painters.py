import logging

import numpy as np

from ..configurable import Configurable

try:
    import matplotlib
    import matplotlib.animation
    import matplotlib.collections
    import matplotlib.patches
except ImportError:
    matplotlib = None


LOG = logging.getLogger(__name__)


class DetectionPainter:
    def __init__(self, *, xy_scale=1.0):
        self.xy_scale = xy_scale

    def annotation(self, ax, ann, *, color=None, text=None, subtext=None):
        if color is None:
            color = 0
        if isinstance(color, (int, np.integer)):
            color = matplotlib.cm.get_cmap('tab20')((color % 20 + 0.05) / 20)

        if text is None:
            text = ann.category
            if getattr(ann, 'id_', None):
                text += ' ({})'.format(ann.id_)
        if subtext is None and ann.score:
            subtext = '{:.0%}'.format(ann.score)

        x, y, w, h = ann.bbox * self.xy_scale
        if w < 5.0:
            x -= 2.0
            w += 4.0
        if h < 5.0:
            y -= 2.0
            h += 4.0

        # draw box
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (x, y), w, h, fill=False, color=color, linewidth=1.0))

        # draw text
        ax.annotate(
            text,
            (x, y),
            fontsize=8,
            xytext=(5.0, 5.0),
            textcoords='offset points',
            color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0},
        )
        if subtext is not None:
            ax.annotate(
                subtext,
                (x, y),
                fontsize=5,
                xytext=(5.0, 18.0 + 3.0),
                textcoords='offset points',
                color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0},
            )


class CrowdPainter:
    def __init__(self, *, xy_scale=1.0):
        self.xy_scale = xy_scale

    @staticmethod
    def draw_polygon(ax, outlines, *, alpha=0.5, color='orange'):
        for outline in outlines:
            assert outline.shape[1] == 2

        patches = []
        for outline in outlines:
            polygon = matplotlib.patches.Polygon(
                outline[:, :2], color=color, facecolor=color, alpha=alpha)
            patches.append(polygon)
        ax.add_collection(matplotlib.collections.PatchCollection(patches, match_original=True))

    def annotation(self, ax, ann, *, color=None, text=None, subtext=None):
        if color is None:
            color = 0
        if isinstance(color, (int, np.integer)):
            color = matplotlib.cm.get_cmap('tab20')((color % 20 + 0.05) / 20)

        if text is None:
            text = '{} (crowd)'.format(ann.category)
            if getattr(ann, 'id_', None):
                text += ' ({})'.format(ann.id_)

        x, y, w, h = ann.bbox * self.xy_scale
        if w < 5.0:
            x -= 2.0
            w += 4.0
        if h < 5.0:
            y -= 2.0
            h += 4.0

        # draw box
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (x, y), w, h, fill=False, color=color, linewidth=1.0, linestyle='dotted'))

        # draw text
        ax.annotate(
            text,
            (x, y),
            fontsize=8,
            xytext=(5.0, 5.0),
            textcoords='offset points',
            color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0},
        )
        if subtext is not None:
            ax.annotate(
                subtext,
                (x, y),
                fontsize=5,
                xytext=(5.0, 18.0 + 3.0),
                textcoords='offset points',
                color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0},
            )


class KeypointPainter(Configurable):
    """Paint poses.

    The constructor can take any class attribute as parameter and
    overwrite the global default for that instance.

    Example to create a KeypointPainter with thick lines:
    >>> kp = KeypointPainter(line_width=48)
    """

    show_box = False
    show_joint_confidences = False
    show_joint_scales = False
    show_decoding_order = False
    show_frontier_order = False
    show_only_decoded_connections = False

    textbox_alpha = 0.5
    text_color = 'white'
    monocolor_connections = False
    line_width = None
    marker_size = None
    solid_threshold = 0.5
    font_size = 8

    def __init__(self, *,
                 xy_scale=1.0,
                 highlight=None,
                 highlight_invisible=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.xy_scale = xy_scale
        self.highlight = highlight
        self.highlight_invisible = highlight_invisible

        # set defaults for line_width and marker_size depending on monocolor
        if self.line_width is None:
            self.line_width = 2 if self.monocolor_connections else 6
        if self.marker_size is None:
            if self.monocolor_connections:
                self.marker_size = max(self.line_width + 1, int(self.line_width * 3.0))
            else:
                self.marker_size = max(1, int(self.line_width * 0.5))

        LOG.debug('color connections = %s, lw = %d, marker = %d',
                  self.monocolor_connections, self.line_width, self.marker_size)

    def _draw_skeleton(self, ax, x, y, v, *,
                       skeleton, skeleton_mask=None, color=None, alpha=1.0, **kwargs):
        if not np.any(v > 0):
            return

        if skeleton_mask is None:
            skeleton_mask = [True for _ in skeleton]
        assert len(skeleton) == len(skeleton_mask)

        # connections
        lines, line_colors, line_styles = [], [], []
        for ci, ((j1i, j2i), mask) in enumerate(zip(np.array(skeleton) - 1, skeleton_mask)):
            if not mask:
                continue
            c = color
            if not self.monocolor_connections:
                c = matplotlib.cm.get_cmap('tab20')((ci % 20 + 0.05) / 20)
            if v[j1i] > 0 and v[j2i] > 0:
                lines.append([(x[j1i], y[j1i]), (x[j2i], y[j2i])])
                line_colors.append(c)
                if v[j1i] > self.solid_threshold and v[j2i] > self.solid_threshold:
                    line_styles.append('solid')
                else:
                    line_styles.append('dashed')
        ax.add_collection(matplotlib.collections.LineCollection(
            lines, colors=line_colors,
            linewidths=kwargs.get('linewidth', self.line_width),
            linestyles=kwargs.get('linestyle', line_styles),
            capstyle='round',
            alpha=alpha,
        ))

        # joints
        ax.scatter(
            x[v > 0.0], y[v > 0.0], s=self.marker_size**2, marker='.',
            color=color if self.monocolor_connections else 'white',
            edgecolor='k' if self.highlight_invisible else None,
            zorder=2,
            alpha=alpha,
        )

        # highlight joints
        if self.highlight is not None:
            highlight_v = np.zeros_like(v)
            highlight_v[self.highlight] = 1
            highlight_v = np.logical_and(v, highlight_v)

            ax.scatter(
                x[highlight_v], y[highlight_v], s=self.marker_size**2, marker='.',
                color=color if self.monocolor_connections else 'white',
                edgecolor='k' if self.highlight_invisible else None,
                zorder=2,
                alpha=alpha,
            )

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

    @classmethod
    def _draw_text(cls, ax, x, y, v, text, color, *, subtext=None, alpha=1.0):
        if cls.font_size == 0:
            return
        if not np.any(v > 0):
            return

        coord_i = np.argsort(y[v > 0])
        if np.sum(v) >= 2 and y[v > 0][coord_i[1]] < y[v > 0][coord_i[0]] + 10:
            # second coordinate within 10 pixels
            f0 = 0.5 + 0.5 * (y[v > 0][coord_i[1]] - y[v > 0][coord_i[0]]) / 10.0
            coord_y = f0 * y[v > 0][coord_i[0]] + (1.0 - f0) * y[v > 0][coord_i[1]]
            coord_x = f0 * x[v > 0][coord_i[0]] + (1.0 - f0) * x[v > 0][coord_i[1]]
        else:
            coord_y = y[v > 0][coord_i[0]]
            coord_x = x[v > 0][coord_i[0]]

        bbox_config = {'facecolor': color, 'alpha': alpha * cls.textbox_alpha, 'linewidth': 0}
        ax.annotate(
            text,
            (coord_x, coord_y),
            fontsize=cls.font_size,
            xytext=(5.0, 5.0),
            textcoords='offset points',
            color=cls.text_color,
            bbox=bbox_config,
            alpha=alpha,
        )
        if subtext is not None:
            ax.annotate(
                subtext,
                (coord_x, coord_y),
                fontsize=cls.font_size * 5 // 8,
                xytext=(5.0, 18.0 + 3.0),
                textcoords='offset points',
                color=cls.text_color,
                bbox=bbox_config,
                alpha=alpha,
            )

    @staticmethod
    def _draw_scales(ax, xs, ys, vs, color, scales, alpha=1.0):
        for x, y, v, scale in zip(xs, ys, vs, scales):
            if v == 0.0:
                continue
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    (x - scale / 2, y - scale / 2), scale, scale,
                    fill=False, color=color, alpha=alpha))

    @classmethod
    def _draw_joint_confidences(cls, ax, xs, ys, vs, color):
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
                color=cls.text_color,
                bbox={'facecolor': color, 'alpha': 0.2, 'linewidth': 0, 'pad': 0.0},
            )

    def annotation(self, ax, ann, *, color=None, text=None, subtext=None, alpha=1.0):
        if color is None:
            color = 0
        if isinstance(color, (int, np.integer)):
            color = matplotlib.cm.get_cmap('tab20')((color % 20 + 0.05) / 20)

        text_is_score = False
        if text is None and hasattr(ann, 'id_'):
            text = '{}'.format(ann.id_)
        if text is None and getattr(ann, 'score', None):
            text = '{:.0%}'.format(ann.score)
            text_is_score = True
        if subtext is None and not text_is_score and getattr(ann, 'score', None):
            subtext = '{:.0%}'.format(ann.score)

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

        skeleton_mask = None
        if self.show_only_decoded_connections:
            decoded_connections = set((jsi, jti) for jsi, jti, _, __ in ann.decoding_order)
            skeleton_mask = [
                (s - 1, e - 1) in decoded_connections or (e - 1, s - 1) in decoded_connections
                for s, e in ann.skeleton
            ]

        self._draw_skeleton(ax, x, y, v, color=color,
                            skeleton=ann.skeleton, skeleton_mask=skeleton_mask, alpha=alpha)

        if self.show_joint_scales and ann.joint_scales is not None:
            self._draw_scales(ax, x, y, v, color, ann.joint_scales, alpha=alpha)

        if self.show_joint_confidences:
            self._draw_joint_confidences(ax, x, y, v, color)

        if self.show_box:
            x_, y_, w_, h_ = [v * self.xy_scale for v in ann.bbox()]
            self._draw_box(ax, x_, y_, w_, h_, color, ann.score())

        if text is not None:
            self._draw_text(ax, x, y, v, text, color, subtext=subtext, alpha=alpha)

        if self.show_decoding_order and hasattr(ann, 'decoding_order'):
            self._draw_decoding_order(ax, ann.decoding_order)

    @staticmethod
    def _draw_decoding_order(ax, decoding_order):
        for step_i, (jsi, jti, jsxyv, jtxyv) in enumerate(decoding_order):
            ax.plot([jsxyv[0], jtxyv[0]], [jsxyv[1], jtxyv[1]], '--', color='black')
            ax.text(0.5 * (jsxyv[0] + jtxyv[0]), 0.5 * (jsxyv[1] + jtxyv[1]),
                    '{}: {} -> {}'.format(step_i, jsi, jti), fontsize=8,
                    color='white', bbox={'facecolor': 'black', 'alpha': 0.5, 'linewidth': 0})
