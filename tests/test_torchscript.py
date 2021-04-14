import pytest
import torch

import openpifpaf


@pytest.mark.xfail
def test_torchscript_script():
    openpifpaf.network.heads.CompositeField3.inplace_ops = False

    datamodule = openpifpaf.datasets.factory('cocokp')
    model, _ = openpifpaf.network.Factory(
        base_name='shufflenetv2k16',
    ).factory(head_metas=datamodule.head_metas)
    with torch.no_grad():
        torch.jit.script(model)


def test_torchscript_trace():
    openpifpaf.network.heads.CompositeField3.inplace_ops = False

    datamodule = openpifpaf.datasets.factory('cocokp')
    model, _ = openpifpaf.network.Factory(
        base_name='shufflenetv2k16',
    ).factory(head_metas=datamodule.head_metas)
    with torch.no_grad():
        torch.jit.trace(model, torch.empty((1, 3, 81, 81)))


@pytest.mark.xfail
def test_torchscript_decoder():
    datamodule = openpifpaf.datasets.factory('cocokp')
    decoder = openpifpaf.decoder.factory(datamodule.head_metas)
    with torch.no_grad():
        torch.jit.script(decoder)
