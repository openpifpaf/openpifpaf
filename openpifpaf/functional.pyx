# cython: infer_types=True
cimport cython
from libc.math cimport exp, fabs, sqrt
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def scalar_square_add_constant(float[:, :] field, x_np, y_np, width_np, float[:] v):
    minx_np = np.round(x_np - width_np).astype(np.int)
    minx_np = np.clip(minx_np, 0, field.shape[1] - 1)
    miny_np = np.round(y_np - width_np).astype(np.int)
    miny_np = np.clip(miny_np, 0, field.shape[0] - 1)
    maxx_np = np.round(x_np + width_np).astype(np.int)
    maxx_np = np.clip(maxx_np + 1, minx_np + 1, field.shape[1])
    maxy_np = np.round(y_np + width_np).astype(np.int)
    maxy_np = np.clip(maxy_np + 1, miny_np + 1, field.shape[0])

    cdef long[:] minx = minx_np
    cdef long[:] miny = miny_np
    cdef long[:] maxx = maxx_np
    cdef long[:] maxy = maxy_np

    cdef Py_ssize_t i, xx, yy
    for i in range(minx.shape[0]):
        for xx in range(minx[i], maxx[i]):
            for yy in range(miny[i], maxy[i]):
                field[yy, xx] += v[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cumulative_average(float[:, :] ca, float[:, :] cw, x_np, y_np, width_np, float[:] v, float[:] w):
    minx_np = np.round(x_np - width_np).astype(np.int)
    minx_np = np.clip(minx_np, 0, ca.shape[1] - 1)
    miny_np = np.round(y_np - width_np).astype(np.int)
    miny_np = np.clip(miny_np, 0, ca.shape[0] - 1)
    maxx_np = np.round(x_np + width_np).astype(np.int)
    maxx_np = np.clip(maxx_np + 1, minx_np + 1, ca.shape[1])
    maxy_np = np.round(y_np + width_np).astype(np.int)
    maxy_np = np.clip(maxy_np + 1, miny_np + 1, ca.shape[0])

    cdef long[:] minx = minx_np
    cdef long[:] miny = miny_np
    cdef long[:] maxx = maxx_np
    cdef long[:] maxy = maxy_np

    cdef Py_ssize_t i, xx, yy
    for i in range(minx.shape[0]):
        if w[i] <= 0.0:
            continue
        for xx in range(minx[i], maxx[i]):
            for yy in range(miny[i], maxy[i]):
                ca[yy, xx] = (w[i] * v[i] + cw[yy, xx] * ca[yy, xx]) / (cw[yy, xx] + w[i])
                cw[yy, xx] += w[i]


cdef inline float approx_exp(float x):
    if x > 2.0 or x < -2.0:
        return 0.0
    x = 1.0 + x / 8.0
    x *= x
    x *= x
    x *= x
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def scalar_square_add_gauss(float[:, :] field, x_np, y_np, sigma_np, v_np, float truncate=2.0):
    sigma_np = np.maximum(1.0, sigma_np)
    width_np = np.maximum(1.0, truncate * sigma_np)
    minx_np = np.round(x_np - width_np).astype(np.int)
    minx_np = np.clip(minx_np, 0, field.shape[1] - 1)
    miny_np = np.round(y_np - width_np).astype(np.int)
    miny_np = np.clip(miny_np, 0, field.shape[0] - 1)
    maxx_np = np.round(x_np + width_np).astype(np.int)
    maxx_np = np.clip(maxx_np + 1, minx_np + 1, field.shape[1])
    maxy_np = np.round(y_np + width_np).astype(np.int)
    maxy_np = np.clip(maxy_np + 1, miny_np + 1, field.shape[0])

    cdef float[:] x = x_np
    cdef float[:] y = y_np
    cdef float[:] sigma = sigma_np
    cdef long[:] minx = minx_np
    cdef long[:] miny = miny_np
    cdef long[:] maxx = maxx_np
    cdef long[:] maxy = maxy_np
    cdef float[:] v = v_np

    cdef Py_ssize_t i, xx, yy
    cdef Py_ssize_t l = minx.shape[0]
    cdef float deltax2, deltay2
    cdef float vv
    cdef float cv, cx, cy, csigma2

    for i in range(l):
        csigma2 = sigma[i]**2
        cx = x[i]
        cy = y[i]
        cv = v[i]
        for xx in range(minx[i], maxx[i]):
            deltax2 = (xx - cx)**2
            for yy in range(miny[i], maxy[i]):
                deltay2 = (yy - cy)**2
                vv = cv * approx_exp(-0.5 * (deltax2 + deltay2) / csigma2)
                field[yy, xx] += vv


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def weiszfeld_nd(x_np, y_np, float[:] weights=None, float epsilon=1e-8, Py_ssize_t max_steps=20):
    """Weighted Weiszfeld algorithm."""
    if weights is None:
        weights = np.ones(x_np.shape[0])

    cdef float[:, :] x = x_np
    cdef float[:] y = y_np
    cdef float[:, :] weights_x = np.zeros_like(x)
    for i in range(weights_x.shape[0]):
        for j in range(weights_x.shape[1]):
            weights_x[i, j] = weights[i] * x[i, j]

    cdef float[:] prev_y = np.zeros_like(y)
    cdef float[:] y_top = np.zeros_like(y)
    cdef float y_bottom
    denom_np = np.zeros_like(weights)
    cdef float[:] denom = denom_np

    for s in range(max_steps):
        prev_y[:] = y

        for i in range(denom.shape[0]):
            denom[i] = sqrt((x[i][0] - prev_y[0])**2 + (x[i][1] - prev_y[1])**2) + epsilon

        y_top[:] = 0.0
        y_bottom = 0.0
        for j in range(denom.shape[0]):
            y_top[0] += weights_x[j, 0] / denom[j]
            y_top[1] += weights_x[j, 1] / denom[j]
            y_bottom += weights[j] / denom[j]
        y[0] = y_top[0] / y_bottom
        y[1] = y_top[1] / y_bottom

        if fabs(y[0] - prev_y[0]) + fabs(y[1] - prev_y[1]) < 1e-2:
            return y_np, denom_np

    return y_np, denom_np


@cython.boundscheck(False)
@cython.wraparound(False)
def paf_mask_center(float[:, :] paf_field, float x, float y, float sigma=1.0):
    mask_np = np.zeros((paf_field.shape[1],), dtype=np.uint8)
    cdef unsigned char[:] mask = mask_np

    for i in range(mask.shape[0]):
        mask[i] = (
            paf_field[1, i] > x - sigma * paf_field[3, i] and
            paf_field[1, i] < x + sigma * paf_field[3, i] and
            paf_field[2, i] > y - sigma * paf_field[3, i] and
            paf_field[2, i] < y + sigma * paf_field[3, i]
        )

    return mask_np != 0


@cython.boundscheck(False)
@cython.wraparound(False)
def paf_center(float[:, :] paf_field, float x, float y, float sigma=1.0):
    result_np = np.empty_like(paf_field)
    cdef float[:, :] result = result_np
    cdef unsigned int result_i = 0
    cdef bint take

    for i in range(paf_field.shape[1]):
        take = (
            paf_field[1, i] > x - sigma * paf_field[3, i] and
            paf_field[1, i] < x + sigma * paf_field[3, i] and
            paf_field[2, i] > y - sigma * paf_field[3, i] and
            paf_field[2, i] < y + sigma * paf_field[3, i]
        )
        if not take:
            continue

        result[:, result_i] = paf_field[:, i]
        result_i += 1

    return result_np[:, :result_i]
