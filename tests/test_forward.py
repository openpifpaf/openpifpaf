import torch

import openpifpaf


def test_forward():
    openpifpaf.plugins.coco.CocoKp.upsample_stride = 1
    datamodule = openpifpaf.datasets.factory('cocokp')
    openpifpaf.network.basenetworks.Resnet.pretrained = False
    model, _ = openpifpaf.network.Factory(base_name='resnet18').factory(
        head_metas=datamodule.head_metas)

    dummy_image_batch = torch.zeros((1, 3, 241, 321))
    cif, caf = model(dummy_image_batch)
    assert cif.shape == (1, 17, 5, 16, 21)
    assert caf.shape == (1, 19, 8, 16, 21)


def test_forward_upsample():
    openpifpaf.plugins.coco.CocoKp.upsample_stride = 2
    datamodule = openpifpaf.datasets.factory('cocokp')
    openpifpaf.network.basenetworks.Resnet.pretrained = False
    model, _ = openpifpaf.network.Factory(base_name='resnet18').factory(
        head_metas=datamodule.head_metas)
    openpifpaf.plugins.coco.CocoKp.upsample_stride = 1  # reset

    dummy_image_batch = torch.zeros((1, 3, 241, 321))
    cif, caf = model(dummy_image_batch)
    assert cif.shape == (1, 17, 5, 31, 41)
    assert caf.shape == (1, 19, 8, 31, 41)
