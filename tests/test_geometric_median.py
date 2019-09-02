import numpy as np
import scipy.interpolate

from openpifpaf.decoder.utils import weiszfeld_nd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def test_robust_1d():
    data = np.array([-10, 3, 5, 7, 8, 50, 5000])
    x = np.linspace(0, 10, 51)
    diff = np.expand_dims(data, 0) - np.expand_dims(x, 1)
    print(diff.shape)
    f = np.sum(np.abs(diff), axis=1)

    if plt is not None:
        plt.figure()
        plt.plot(x, f, 'x-')
        plt.xlabel('$y$')
        plt.tight_layout()
        # plt.show()

    assert x[np.argmin(f)] == 7.0


def weiszfeld_1d(x, last_y):
    return np.sum(x / np.abs(x - last_y)) / np.sum(1.0 / np.abs(x - last_y))


def test_iterative_1d():
    data = np.array([-10, 3, 5, 7, 8, 50, 5000])
    ys = np.linspace(0, 10, 51)
    diff = np.expand_dims(data, 0) - np.expand_dims(ys, 1)
    f = np.sum(np.abs(diff), axis=1)

    y = [2.0]
    for _ in range(10):
        y.append(weiszfeld_1d(data, y[-1]))
    vs = scipy.interpolate.interp1d(ys, f)(y)

    # iterations
    if plt is not None:
        plt.figure()
        plt.plot(y, vs, 'x-')
        plt.xlabel('$y$')
        plt.tight_layout()
        # plt.show()


def test_iterative_nd():
    data = np.array([[-10], [3], [5], [7], [8], [50], [5000]])
    y_init = np.mean(data, axis=0)
    y, _ = weiszfeld_nd(data, y_init)
    print(y_init, y)
    assert 6.99 < y < 7.01


def test_iterative_weighted_nd():
    data = np.array([[-10], [3], [5], [7], [8], [50], [100], [5000]])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    y_init = [np.sum(data * np.expand_dims(weights, -1)) / np.sum(weights)]
    y, _ = weiszfeld_nd(data, y_init, weights=weights)
    print(y_init, y)
    assert 6.99 < y[-1] < 7.01
