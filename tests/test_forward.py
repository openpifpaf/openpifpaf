import torch

import openpifpaf


def test_forward():
    openpifpaf.datasets.CocoKp.upsample_stride = 1
    datamodule = openpifpaf.datasets.factory('cocokp')
    openpifpaf.network.basenetworks.Resnet.pretrained = False
    model, _ = openpifpaf.network.factory(
        base_name='resnet18',
        head_metas=datamodule.head_metas,
    )

    dummy_image_batch = torch.zeros((1, 3, 241, 321))
    cif, caf, _ = model(dummy_image_batch)
    assert cif.shape == (1, 17, 5, 16, 21)
    assert caf.shape == (1, 19, 9, 16, 21)


def test_forward_dense():
    openpifpaf.datasets.CocoKp.upsample_stride = 1
    datamodule = openpifpaf.datasets.factory('cocokp')
    openpifpaf.network.basenetworks.Resnet.pretrained = False
    model, _ = openpifpaf.network.factory(
        base_name='resnet18',
        head_metas=datamodule.head_metas,
        dense_coupling=1.0,
    )

    dummy_image_batch = torch.zeros((1, 3, 241, 321))
    cif, caf = model(dummy_image_batch)
    assert cif.shape == (1, 17, 5, 16, 21)
    assert caf.shape == (1, 19 + 25, 9, 16, 21)


def test_forward_headquad():
    openpifpaf.datasets.CocoKp.upsample_stride = 2
    datamodule = openpifpaf.datasets.factory('cocokp')
    openpifpaf.network.basenetworks.Resnet.pretrained = False
    model, _ = openpifpaf.network.factory(
        base_name='resnet18',
        head_metas=datamodule.head_metas,
    )

    dummy_image_batch = torch.zeros((1, 3, 241, 321))
    cif, caf, _ = model(dummy_image_batch)
    assert cif.shape == (1, 17, 5, 31, 41)
    assert caf.shape == (1, 19, 9, 31, 41)
