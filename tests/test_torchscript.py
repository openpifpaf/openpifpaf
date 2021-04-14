import os

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


class ModuleWithCifHrOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.cifhr = torch.zeros((17, 300, 400))

    def forward(self, x):
        torch.ops.my_ops.cif_hr_accumulate_op(self.cifhr, x, 8, 0.1, 16, 0.0, 1.0)
        return x


def test_torchscript_cifhrop(tmpdir):
    openpifpaf.plugin.register()

    outfile = str(tmpdir.join('cifhrop.pt'))
    assert not os.path.exists(outfile)

    model = ModuleWithCifHrOp()

    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, outfile)
    assert os.path.exists(outfile)
