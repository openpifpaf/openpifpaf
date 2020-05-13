import logging

import numpy as np

try:
    import matplotlib
    import matplotlib.animation
    import matplotlib.collections
    import matplotlib.patches
except ImportError:
    matplotlib = None


LOG = logging.getLogger(__name__)


def quiver(ax, vector_field, *,
           confidence_field=None, step=1, threshold=0.5,
           xy_scale=1.0, uv_is_offset=False,
           reg_uncertainty=None, **kwargs):
    x, y, u, v, c, r = [], [], [], [], [], []
    for j in range(0, vector_field.shape[1], step):
        for i in range(0, vector_field.shape[2], step):
            if confidence_field is not None and confidence_field[j, i] < threshold:
                continue
            x.append(i * xy_scale)
            y.append(j * xy_scale)
            u.append(vector_field[0, j, i] * xy_scale)
            v.append(vector_field[1, j, i] * xy_scale)
            c.append(confidence_field[j, i] if confidence_field is not None else 1.0)
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


def margins(ax, vector_field, *,
            confidence_field=None, step=1, threshold=0.5,
            xy_scale=1.0, uv_is_offset=False, **kwargs):
    x, y, u, v, r = [], [], [], [], []
    for j in range(0, vector_field.shape[1], step):
        for i in range(0, vector_field.shape[2], step):
            if confidence_field is not None and confidence_field[j, i] < threshold:
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


def boxes(ax, sigma_field, **kwargs):
    boxes_wh(ax, sigma_field * 2.0, sigma_field * 2.0, **kwargs)


def boxes_wh(ax, w_field, h_field, *, confidence_field=None, regression_field=None,
             xy_scale=1.0, step=1, threshold=0.5,
             regression_field_is_offset=False,
             cmap='viridis_r', clim=(0.5, 1.0), **kwargs):
    x, y, w, h, c = [], [], [], [], []
    for j in range(0, w_field.shape[0], step):
        for i in range(0, w_field.shape[1], step):
            if confidence_field is not None and confidence_field[j, i] < threshold:
                continue
            x_offset, y_offset = 0.0, 0.0
            if regression_field is not None:
                x_offset = regression_field[0, j, i]
                y_offset = regression_field[1, j, i]
                if not regression_field_is_offset:
                    x_offset -= i
                    y_offset -= j
            x.append((i + x_offset) * xy_scale)
            y.append((j + y_offset) * xy_scale)
            w.append(w_field[j, i] * xy_scale)
            h.append(h_field[j, i] * xy_scale)
            c.append(confidence_field[j, i] if confidence_field is not None else 1.0)

    cmap = matplotlib.cm.get_cmap(cmap)
    cnorm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    for xx, yy, ww, hh, cc in zip(x, y, w, h, c):
        color = cmap(cnorm(cc))
        rectangle = matplotlib.patches.Rectangle(
            (xx - ww / 2.0, yy - hh / 2.0), ww, hh,
            color=color, zorder=10, linewidth=1, **kwargs)
        ax.add_artist(rectangle)


def circles(ax, radius_field, *, confidence_field=None, regression_field=None,
            xy_scale=1.0, step=1, threshold=0.5,
            regression_field_is_offset=False,
            cmap='viridis_r', clim=(0.5, 1.0), **kwargs):
    x, y, s, c = [], [], [], []
    for j in range(0, radius_field.shape[0], step):
        for i in range(0, radius_field.shape[1], step):
            if confidence_field is not None and confidence_field[j, i] < threshold:
                continue
            x_offset, y_offset = 0.0, 0.0
            if regression_field is not None:
                x_offset = regression_field[0, j, i]
                y_offset = regression_field[1, j, i]
                if not regression_field_is_offset:
                    x_offset -= i
                    y_offset -= j
            x.append((i + x_offset) * xy_scale)
            y.append((j + y_offset) * xy_scale)
            s.append(radius_field[j, i] * xy_scale)
            c.append(confidence_field[j, i] if confidence_field is not None else 1.0)

    cmap = matplotlib.cm.get_cmap(cmap)
    cnorm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    for xx, yy, ss, cc in zip(x, y, s, c):
        color = cmap(cnorm(cc))
        circle = matplotlib.patches.Circle(
            (xx, yy), ss,
            color=color, zorder=10, linewidth=1, **kwargs)
        ax.add_artist(circle)
