# cython: infer_types=True
cimport cython
from libc.math cimport exp, fabs, sqrt, fmin, fmax
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void scalar_square_add_constant(float[:, :] field, float[:] x, float[:] y, float[:] width, float[:] v) nogil:
    cdef long minx, miny, maxx, maxy
    cdef Py_ssize_t i, xx, yy
    cdef float cx, cy, cv, cwidth

    for i in range(x.shape[0]):
        cx = x[i]
        cy = y[i]
        cv = v[i]
        cwidth = width[i]

        minx = (<long>clip(cx - cwidth, 0, field.shape[1] - 1))
        maxx = (<long>clip(cx + cwidth, minx + 1, field.shape[1]))
        miny = (<long>clip(cy - cwidth, 0, field.shape[0] - 1))
        maxy = (<long>clip(cy + cwidth, miny + 1, field.shape[0]))
        for xx in range(minx, maxx):
            for yy in range(miny, maxy):
                field[yy, xx] += cv


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void cumulative_average(float[:, :] cuma, float[:, :] cumw, float[:] x, float[:] y, float[:] width, float[:] v, float[:] w) nogil:
    cdef long minx, miny, maxx, maxy
    cdef float cv, cw, cx, cy, cwidth
    cdef Py_ssize_t i, xx, yy

    for i in range(x.shape[0]):
        cw = w[i]
        if cw <= 0.0:
            continue

        cv = v[i]
        cx = x[i]
        cy = y[i]
        cwidth = width[i]

        minx = (<long>clip(cx - cwidth, 0, cuma.shape[1] - 1))
        maxx = (<long>clip(cx + cwidth, minx + 1, cuma.shape[1]))
        miny = (<long>clip(cy - cwidth, 0, cuma.shape[0] - 1))
        maxy = (<long>clip(cy + cwidth, miny + 1, cuma.shape[0]))
        for xx in range(minx, maxx):
            for yy in range(miny, maxy):
                cuma[yy, xx] = (cw * cv + cumw[yy, xx] * cuma[yy, xx]) / (cumw[yy, xx] + cw)
                cumw[yy, xx] += cw


cdef inline float approx_exp(float x) nogil:
    if x > 2.0 or x < -2.0:
        return 0.0
    x = 1.0 + x / 8.0
    x *= x
    x *= x
    x *= x
    return x


cdef inline float clip(float v, float minv, float maxv) nogil:
    return fmax(minv, fmin(maxv, v))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void scalar_square_add_gauss(float[:, :] field, float[:] x, float[:] y, float[:] sigma, float[:] v, float truncate=2.0) nogil:
    cdef Py_ssize_t i, xx, yy
    cdef float vv, deltax2, deltay2
    cdef float cv, cx, cy, csigma, csigma2
    cdef long minx, miny, maxx, maxy

    for i in range(x.shape[0]):
        csigma = sigma[i]
        csigma2 = csigma * csigma
        cx = x[i]
        cy = y[i]
        cv = v[i]

        minx = (<long>clip(cx - truncate * csigma, 0, field.shape[1] - 1))
        maxx = (<long>clip(cx + truncate * csigma, minx + 1, field.shape[1]))
        miny = (<long>clip(cy - truncate * csigma, 0, field.shape[0] - 1))
        maxy = (<long>clip(cy + truncate * csigma, miny + 1, field.shape[0]))
        for xx in range(minx, maxx):
            deltax2 = (xx - cx)**2
            for yy in range(miny, maxy):
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
def scalar_values(float[:, :] field, float[:] x, float[:] y, float default=-1):
    values_np = np.full((x.shape[0],), default, dtype=np.float32)
    cdef float[:] values = values_np
    cdef float maxx = <float>field.shape[1] - 1, maxy = <float>field.shape[0] - 1

    for i in range(values.shape[0]):
        if x[i] < 0.0 or y[i] < 0.0 or x[i] > maxx or y[i] > maxy:
            continue

        values[i] = field[<Py_ssize_t>y[i], <Py_ssize_t>x[i]]

    return values_np


@cython.boundscheck(False)
@cython.wraparound(False)
def scalar_value(float[:, :] field, float x, float y, float default=-1):
    if x < 0.0 or y < 0.0 or x > field.shape[1] - 1 or y > field.shape[0] - 1:
        return default

    return field[<Py_ssize_t>y, <Py_ssize_t>x]


@cython.boundscheck(False)
@cython.wraparound(False)
def scalar_value_clipped(float[:, :] field, float x, float y):
    x = clip(x, 0.0, field.shape[1] - 1)
    y = clip(y, 0.0, field.shape[0] - 1)
    return field[<Py_ssize_t>y, <Py_ssize_t>x]


@cython.boundscheck(False)
@cython.wraparound(False)
def scalar_nonzero(unsigned char[:, :] field, float x, float y, unsigned char default=0):
    if x < 0.0 or y < 0.0 or x > field.shape[1] - 1 or y > field.shape[0] - 1:
        return default

    return field[<Py_ssize_t>y, <Py_ssize_t>x]


@cython.boundscheck(False)
@cython.wraparound(False)
def scalar_nonzero_clipped(unsigned char[:, :] field, float x, float y):
    x = clip(x, 0.0, field.shape[1] - 1)
    y = clip(y, 0.0, field.shape[0] - 1)
    return field[<Py_ssize_t>y, <Py_ssize_t>x]


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
