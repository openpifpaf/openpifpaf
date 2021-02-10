import numpy as np
import torch
import openpifpaf


def localize(x):
    openpifpaf.plugin.register()
    openpifpaf.plugins.coco.CocoKp.upsample_stride = 1
    datamodule = openpifpaf.datasets.factory('cocokp')
    openpifpaf.network.basenetworks.Resnet.pretrained = False

    black = torch.zeros((3, 321, 321))
    im = black.clone()
    im[:, 0, x] = 1000.0

    model, _ = openpifpaf.network.Factory(base_name='resnet18').factory(
        head_metas=datamodule.head_metas)
    model.eval()

    decode = openpifpaf.decoder.factory(datamodule.head_metas)
    cif_ref = decode.fields_batch(model, torch.unsqueeze(black, 0))[0][0]
    cif = decode.fields_batch(model, torch.unsqueeze(im, 0))[0][0]

    # intensity only, first field, first row
    cif_ref = cif_ref[0][0][0]
    cif = cif[0][0][0]
    assert len(cif_ref) == 21  # (321 - 1) / 16 + 1
    assert len(cif) == len(cif_ref)

    active_cif = np.nonzero(cif_ref - cif)
    return active_cif[0].tolist()


def test_pixel_to_field_left():
    assert localize(0) == [0, 1, 2, 3, 4, 5, 6]


def test_pixel_to_field_center():
    assert localize(160) == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def test_pixel_to_field_right():
    assert localize(320) == [14, 15, 16, 17, 18, 19, 20]
