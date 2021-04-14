import os
import sys

import pytest
import torch

import openpifpaf
import openpifpaf.export_coreml


@pytest.mark.skipif(not sys.platform.startswith('darwin'), reason='coreml export only on macos')
def test_coreml_exportable(tmpdir):
    outfile = str(tmpdir.join('openpifpaf-shufflenetv2k16.coreml.mlmodel'))
    assert not os.path.exists(outfile)

    datamodule = openpifpaf.datasets.factory('cocokp')
    model, _ = openpifpaf.network.Factory(
        base_name='shufflenetv2k16',
    ).factory(head_metas=datamodule.head_metas)
    openpifpaf.export_coreml.apply(model, outfile)
    assert os.path.exists(outfile)


class ModuleWithOccupancy(openpifpaf.network.HeadNetwork):
    def __init__(self, meta, in_features):
        super().__init__(meta, in_features)
        self.occupancy = torch.classes.my_classes.Occupancy([17, 100, 120], 1.0, 0.1)

    def forward(self, *args):
        x = args[0]
        return x


@pytest.mark.skipif(not sys.platform.startswith('darwin'), reason='coreml export only on macos')
def test_coreml_torchscript(tmpdir):
    openpifpaf.plugin.register()

    outfile = str(tmpdir.join('occupancy.coreml.mlmodel'))
    assert not os.path.exists(outfile)

    datamodule = openpifpaf.datasets.factory('cocokp')
    model, _ = openpifpaf.network.Factory(
        base_name='shufflenetv2k16',
    ).factory(head_metas=datamodule.head_metas)
    model.set_head_nets([
        ModuleWithOccupancy(model.head_metas[0], model.base_net.out_features),
        model.head_nets[1],
    ])

    openpifpaf.export_coreml.apply(model, outfile)
    assert os.path.exists(outfile)


class ModuleWithCifHr(openpifpaf.network.HeadNetwork):
    def __init__(self, meta, in_features):
        super().__init__(meta, in_features)
        self.cifhr = torch.classes.my_classes.CifHr([17, 25, 30], 8)

    def forward(self, *args):
        x = args[0]
        with torch.no_grad():
            self.cifhr.accumulate(x, 8, 0.0, 1.0)
        return x


@pytest.mark.skipif(not sys.platform.startswith('darwin'), reason='coreml export only on macos')
@pytest.mark.xfail
def test_coreml_torchscript_cifhr(tmpdir):
    openpifpaf.plugin.register()

    outfile = str(tmpdir.join('cifhr.coreml.mlmodel'))
    assert not os.path.exists(outfile)

    datamodule = openpifpaf.datasets.factory('cocokp')
    model, _ = openpifpaf.network.Factory(
        base_name='shufflenetv2k16',
    ).factory(head_metas=datamodule.head_metas)
    model.set_head_nets([
        ModuleWithCifHr(model.head_metas[0], model.base_net.out_features),
        model.head_nets[1],
    ])

    openpifpaf.export_coreml.apply(model, outfile)
    assert os.path.exists(outfile)
