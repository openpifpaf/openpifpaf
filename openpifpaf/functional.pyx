# cython: infer_types=True
cimport cython
from libc.math cimport exp, fabs, sqrt
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def scalar_square_add_constant(double[:, :] field, x_np, y_np, width_np, double[:] v):
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

    # for minxx, minyy, maxxx, maxyy, vv in zip(minx, miny, maxx, maxy, v):
    for i in range(minx.shape[0]):
        for xx in range(minx[i], maxx[i]):
            for yy in range(miny[i], maxy[i]):
                field[yy, xx] += v[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def scalar_square_add_gauss(double[:, :] field, x_np, y_np, sigma_np, v_np, double truncate=2.0):
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

    cdef double[:] x = x_np
    cdef double[:] y = y_np
    cdef double[:] sigma = sigma_np
    cdef long[:] minx = minx_np
    cdef long[:] miny = miny_np
    cdef long[:] maxx = maxx_np
    cdef long[:] maxy = maxy_np
    cdef double[:] v = v_np

    cdef Py_ssize_t i, xx, yy
    cdef Py_ssize_t l = minx.shape[0]
    cdef double deltax, deltay
    cdef double vv

    # for minxx, minyy, maxxx, maxyy, vv in zip(minx, miny, maxx, maxy, v):
    for i in range(l):
        for xx in range(minx[i], maxx[i]):
            deltax = xx - x[i]
            for yy in range(miny[i], maxy[i]):
                deltay = yy - y[i]
                vv = v[i] * exp(-0.5 * (deltax**2 + deltay**2) / sigma[i]**2)
                field[yy, xx] += vv


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def weiszfeld_nd(x_np, y_np, double[:] weights=None, double epsilon=1e-8, Py_ssize_t max_steps=20):
    """Weighted Weiszfeld algorithm."""
    if weights is None:
        weights = np.ones(x_np.shape[0])

    cdef double[:, :] x = x_np
    cdef double[:] y = y_np
    cdef double[:, :] weights_x = np.zeros_like(x)
    for i in range(weights_x.shape[0]):
        for j in range(weights_x.shape[1]):
            weights_x[i, j] = weights[i] * x[i, j]

    cdef double[:] prev_y = np.zeros_like(y)
    cdef double[:] y_top = np.zeros_like(y)
    cdef double y_bottom
    denom_np = np.zeros_like(weights)
    cdef double[:] denom = denom_np

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
def paf_mask_center(double[:, :] paf_field, double x, double y, double sigma=1.0):
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
def paf_center(double[:, :] paf_field, double x, double y, double sigma=1.0):
    result_np = np.empty_like(paf_field)
    cdef double[:, :] result = result_np
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
