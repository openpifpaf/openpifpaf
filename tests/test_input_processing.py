import numpy as np
import PIL
import torchvision


def test_resize():
    np_image = np.zeros((5, 5))
    np_image[2, 2] = 1.0

    image = PIL.Image.fromarray(np_image)
    image = torchvision.transforms.functional.resize(image, (10, 10))

    np_result = np.asarray(image)
    assert np.all(np_result[:5] == np_result[:4:-1])  # symmetric
    assert np.all(np_result[:, :5] == np_result[:, :4:-1])  # symmetric


def test_resize_bicubic():
    np_image = np.zeros((5, 5))
    np_image[2, 2] = 1.0

    image = PIL.Image.fromarray(np_image)
    image = torchvision.transforms.functional.resize(image, (10, 10), PIL.Image.BICUBIC)

    np_result = np.asarray(image)
    assert np.all(np_result[:5] == np_result[:4:-1])  # symmetric
    assert np.all(np_result[:, :5] == np_result[:, :4:-1])  # symmetric
