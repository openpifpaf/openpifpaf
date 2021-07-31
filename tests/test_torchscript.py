import os

import pytest
import torch

import openpifpaf
import openpifpaf.export_torchscript


@pytest.mark.xfail
def test_torchscript_script():
    openpifpaf.network.heads.CompositeField3.inplace_ops = False
    openpifpaf.network.heads.CompositeField4.inplace_ops = False

    datamodule = openpifpaf.datasets.factory('cocokp')
    model, _ = openpifpaf.network.Factory(
        base_name='shufflenetv2k16',
    ).factory(head_metas=datamodule.head_metas)
    with torch.inference_mode():
        torch.jit.script(model)


def test_torchscript_trace():
    openpifpaf.network.heads.CompositeField3.inplace_ops = False
    openpifpaf.network.heads.CompositeField4.inplace_ops = False

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
    with torch.inference_mode():
        torch.jit.script(decoder)


def test_torchscript_exportable(tmpdir):
    outfile = str(tmpdir.join('openpifpaf-shufflenetv2k16.torchscript.pt'))
    assert not os.path.exists(outfile)

    datamodule = openpifpaf.datasets.factory('cocokp')
    model, _ = openpifpaf.network.Factory(
        base_name='shufflenetv2k16',
    ).factory(head_metas=datamodule.head_metas)
    openpifpaf.export_torchscript.apply(model, outfile)
    assert os.path.exists(outfile)
