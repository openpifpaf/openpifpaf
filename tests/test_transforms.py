import numpy as np
import PIL

from openpifpaf import transforms


def rescale_from_12_to_6(im_np, x, y):
    print(im_np[0, :, 0])
    im = PIL.Image.fromarray(im_np)

    anns = [{
        'keypoints': [[x, y, 2.0]],
        'bbox': [0.0, 0.0, 12.0, 12.0],
        'segmentation': None,
    }]

    transform = transforms.Compose([
        transforms.NormalizeAnnotations(),
        transforms.RescaleAbsolute(6),
        transforms.EVAL_TRANSFORM,
    ])

    return transform(im, anns, None)


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
        anns_transformed[0]['keypoints'][0][:2].tolist(),
    )


def test_crop(x=4, y=6):
    image_xy, keypoint_xy = single_pixel_transform(x, y, transforms.Crop(7))
    print(image_xy, keypoint_xy)
    assert image_xy == keypoint_xy


def test_pad(x=4, y=6):
    image_xy, keypoint_xy = single_pixel_transform(x, y, transforms.CenterPad(17))
    print(image_xy, keypoint_xy)
    assert image_xy == keypoint_xy


def test_rotateby90(x=4, y=6):
    transform = transforms.Compose([
        transforms.SquarePad(),
        transforms.RotateBy90(),
    ])
    image_xy, keypoint_xy = single_pixel_transform(x, y, transform)
    print(image_xy, keypoint_xy)
    assert image_xy == keypoint_xy
