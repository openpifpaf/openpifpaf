import numpy as np
import torch
import openpifpaf.network.nets
import openpifpaf.utils
from openpifpaf.data import COCO_KEYPOINTS, COCO_PERSON_SKELETON


def localize(x):
    black = torch.zeros((3, 321, 321))
    im = black.clone()
    im[:, 0, x] = 1000.0

    model = openpifpaf.network.nets.factory_from_scratch(
        'resnet50block5', ['pif', 'paf'],
        pretrained=False,
    )
    model.eval()

    decode = openpifpaf.decoder.PifPaf(8, keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON)
    processor = openpifpaf.decoder.Processor(model, decode)
    pif_ref = processor.fields(torch.unsqueeze(black, 0))[0][0]
    pif = processor.fields(torch.unsqueeze(im, 0))[0][0]

    # intensity only, first field, first row
    pif_ref = pif_ref[0][0][0]
    pif = pif[0][0][0]
    assert len(pif_ref) == 21  # (306 + 14 (padding)) / 16
    assert len(pif) == len(pif_ref)

    active_pif = np.nonzero(pif_ref - pif)
    return active_pif[0].tolist()


def test_pixel_to_field_left():
    assert localize(0) == [0, 1, 2, 3, 4, 5, 6]


def test_pixel_to_field_center():
    assert localize(160) == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def test_pixel_to_field_right():
    assert localize(320) == [14, 15, 16, 17, 18, 19, 20]
