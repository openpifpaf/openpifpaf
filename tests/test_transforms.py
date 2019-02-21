import numpy as np
import PIL

from openpifpaf.transforms import SquareCrop


def rescale_from_12_to_6(im_np, x, y):
    print(im_np[0, :, 0])
    im = PIL.Image.fromarray(im_np)

    anns = [{
        'keypoints': [[x, y, 2.0]],
        'bbox': [0.0, 0.0, 12.0, 12.0],
        'segmentation': None,
    }]

    transform = SquareCrop(6, min_scale=1.0)

    return transform(im, anns)


def test_1():
    im_in = np.zeros((12, 12, 3), dtype=np.uint8)
    im_in[0, [5, 6]] = 255

    im, anns, _ = rescale_from_12_to_6(im_in, 5.5, 0)
    im = np.asarray(im)

    print(anns)
    assert anns[0]['bbox'].tolist() == [0.0, 0.0, 6.0, 6.0]
    assert anns[0]['keypoints'][0, 0] == 2.5
    assert anns[0]['keypoints'][0, 1] == -0.25


    print(im[0, :, 0], im.shape)
    assert im[0, :3, 0].tolist() == im[0, :2:-1, 0].tolist()  # symmetric
