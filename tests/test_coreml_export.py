import os

import numpy as np
import onnxruntime
import pytest
import torch

import openpifpaf
import openpifpaf.export_coreml


def test_onnx_exportable(tmpdir):
    openpifpaf.plugin.register()

    outfile = str(tmpdir.join('openpifpaf-shufflenetv2k16.coreml.mlmodel'))
    assert not os.path.exists(outfile)

    datamodule = openpifpaf.datasets.factory('cocokp')
    model, _ = openpifpaf.network.Factory(
        base_name='shufflenetv2k16',
    ).factory(head_metas=datamodule.head_metas)
    openpifpaf.export_coreml.apply(model, outfile)
    assert os.path.exists(outfile)
