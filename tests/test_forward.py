import torch

import openpifpaf


def test_forward():
    openpifpaf.network.heads.CompositeFieldFused.quad = 0
    model, _ = openpifpaf.network.factory(
        base_name='resnet18',
        head_names=['cif', 'caf', 'caf25'],
        pretrained=False,
    )

    dummy_image_batch = torch.zeros((1, 3, 241, 321))
    cif, caf = model(dummy_image_batch)
    assert cif.shape == (1, 17, 5, 16, 21)
    assert caf.shape == (1, 19, 9, 16, 21)


def test_forward_dense():
    openpifpaf.network.heads.CompositeFieldFused.quad = 0
    model, _ = openpifpaf.network.factory(
        base_name='resnet18',
        head_names=['cif', 'caf', 'caf25'],
        dense_connections=True,
        pretrained=False,
    )

    dummy_image_batch = torch.zeros((1, 3, 241, 321))
    cif, caf = model(dummy_image_batch)
    assert cif.shape == (1, 17, 5, 16, 21)
    assert caf.shape == (1, 19 + 25, 9, 16, 21)


def test_forward_headquad():
    openpifpaf.network.heads.CompositeFieldFused.quad = 1
    model, _ = openpifpaf.network.factory(
        base_name='resnet18',
        head_names=['cif', 'caf', 'caf25'],
        pretrained=False,
    )

    dummy_image_batch = torch.zeros((1, 3, 241, 321))
    cif, caf = model(dummy_image_batch)
    assert cif.shape == (1, 17, 5, 31, 41)
    assert caf.shape == (1, 19, 9, 31, 41)
