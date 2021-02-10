import numpy as np
import torch

import openpifpaf


def test_forward():
    openpifpaf.plugin.register()
    openpifpaf.plugins.coco.CocoKp.upsample_stride = 1
    datamodule = openpifpaf.datasets.factory('cocokp')
    openpifpaf.network.basenetworks.Resnet.pretrained = False
    model, _ = openpifpaf.network.Factory(base_name='resnet18').factory(
        head_metas=datamodule.head_metas)

    dummy_image_batch = torch.zeros((1, 3, 241, 321))
    cif, caf = model(dummy_image_batch)
    assert cif.shape == (1, 17, 5, 16, 21)
    assert caf.shape == (1, 19, 9, 16, 21)


def test_forward_upsample():
    openpifpaf.plugin.register()
    openpifpaf.plugins.coco.CocoKp.upsample_stride = 2
    datamodule = openpifpaf.datasets.factory('cocokp')
    openpifpaf.network.basenetworks.Resnet.pretrained = False
    model, _ = openpifpaf.network.Factory(base_name='resnet18').factory(
        head_metas=datamodule.head_metas)

    dummy_image_batch = torch.zeros((1, 3, 241, 321))
    cif, caf = model(dummy_image_batch)
    assert cif.shape == (1, 17, 5, 31, 41)
    assert caf.shape == (1, 19, 9, 31, 41)


def test_forward_noinplace():
    openpifpaf.plugin.register()
    openpifpaf.plugins.coco.CocoKp.upsample_stride = 2
    datamodule = openpifpaf.datasets.factory('cocokp')
    openpifpaf.network.basenetworks.Resnet.pretrained = False
    model, _ = openpifpaf.network.Factory(base_name='resnet18').factory(
        head_metas=datamodule.head_metas)

    dummy_image_batch = torch.zeros((1, 3, 241, 321))

    with torch.no_grad():
        openpifpaf.network.heads.CompositeField3.inplace_ops = True
        ref_cif, ref_caf = model(dummy_image_batch)

        openpifpaf.network.heads.CompositeField3.inplace_ops = False
        cif, caf = model(dummy_image_batch)

    np.testing.assert_allclose(ref_cif.numpy(), cif.numpy())
    np.testing.assert_allclose(ref_caf.numpy(), caf.numpy())

    # back to default
    openpifpaf.network.heads.CompositeField3.inplace_ops = True
