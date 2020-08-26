import numpy as np
import PIL
import pytest

from openpifpaf import transforms


def apply_transform(im_np, anns, transform=None):
    im = PIL.Image.fromarray(im_np)

    transform_list = [transforms.NormalizeAnnotations()]
    if transform is not None:
        transform_list.append(transform)

    im_transformed, anns_transformed, meta = transforms.Compose(transform_list)(im, anns, None)
    im_transformed_np = np.asarray(im_transformed)

    return im_transformed_np, anns_transformed, meta


def single_pixel_transform(x, y, transform, image_wh=(13, 11)):
    im = np.zeros((image_wh[1], image_wh[0], 3), dtype=np.uint8)
    im[y, x, :] = 255

    anns = [{
        'keypoints': [(x, y, 2.0)],
        'bbox': [0.0, 0.0, image_wh[0], image_wh[1]],
    }]

    im_transformed, anns_transformed, _ = apply_transform(im, anns, transform)

    image_yx = np.unravel_index(
        np.argmax(im_transformed[:, :, 0]),
        shape=im_transformed[:, :, 0].shape,
    )

    return (
        [image_yx[1], image_yx[0]],
        # pylint: disable=unsubscriptable-object
        anns_transformed[0]['keypoints'][0][:2].tolist(),
    )


def test_rescale_absolute(x=5, y=5):
    image_xy, keypoint_xy = single_pixel_transform(
        x, y, transforms.RescaleAbsolute(7), image_wh=(11, 11))
    print(image_xy, keypoint_xy)
    assert image_xy == keypoint_xy


def test_crop(x=4, y=6):
    image_xy, keypoint_xy = single_pixel_transform(x, y, transforms.Crop(7), (9, 11))
    print(image_xy, keypoint_xy)
    assert image_xy == keypoint_xy


def test_pad(x=4, y=6):
    image_xy, keypoint_xy = single_pixel_transform(x, y, transforms.CenterPad(17))
    print(image_xy, keypoint_xy)
    assert image_xy == keypoint_xy


@pytest.mark.parametrize('x', range(10))
def test_rotateby90(x, y=6):
    transform = transforms.Compose([
        transforms.SquarePad(),
        transforms.RotateBy90(),
    ])
    image_xy, keypoint_xy = single_pixel_transform(x, y, transform)
    print(image_xy, keypoint_xy)
    assert image_xy == pytest.approx(keypoint_xy)
