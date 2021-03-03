import time

import numpy as np
import PIL.Image
import pytest
import scipy.ndimage

try:
    import cv2
except ImportError:
    cv2 = None


@pytest.mark.parametrize('resample', [PIL.Image.BILINEAR, PIL.Image.BICUBIC])
@pytest.mark.xfail
def test_pil_resize(resample):
    d_in = np.array([[0, 10, 20, 30, 40, 50]], dtype=np.uint8)
    image = PIL.Image.fromarray(d_in, mode='L')

    w, _ = image.size
    target_w = (w - 1) * 2 + 1
    image = image.resize((target_w, 1), resample=resample)

    d_out = np.asarray(image)
    print(d_out)
    assert np.all(d_in == d_out[0, ::2])


@pytest.mark.parametrize('order', [0, 1, 2, 3])
def test_scipy_zoom(order):
    d_in = np.array([[0, 10, 20, 30, 40, 50]], dtype=np.uint8)

    w = d_in.shape[1]
    target_w = (w - 1) * 2 + 1
    d_out = scipy.ndimage.zoom(d_in, (1, target_w / w), order=order)

    print(d_out)
    assert np.all(d_in == d_out[0, ::2])


def test_opencv_faster_than_pil():
    image = PIL.Image.new('RGB', (640, 640))

    pil_start = time.perf_counter()
    for _ in range(5):
        image.resize((500, 500))
    pil_time = time.perf_counter() - pil_start

    cv_start = time.perf_counter()
    for _ in range(5):
        im_np = np.asarray(image)
        PIL.Image.fromarray(cv2.resize(im_np, (500, 500)))
    cv_time = time.perf_counter() - cv_start

    print(f'pil: {pil_time}, cv: {cv_time}')
    assert cv_time < 0.5 * pil_time
