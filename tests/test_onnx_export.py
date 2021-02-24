import os
import sys

import numpy as np
import onnxruntime
import pytest
import torch

import openpifpaf
import openpifpaf.export_onnx


@pytest.mark.skipif(sys.platform.startswith('darwin'), reason='onnx simplify crashes')
def test_onnx_exportable(tmpdir):
    openpifpaf.plugin.register()

    outfile = str(tmpdir.join('openpifpaf-shufflenetv2k16.onnx'))
    assert not os.path.exists(outfile)

    datamodule = openpifpaf.datasets.factory('cocokp')
    model, _ = openpifpaf.network.Factory(
        base_name='shufflenetv2k16',
    ).factory(head_metas=datamodule.head_metas)
    openpifpaf.export_onnx.apply(model, outfile, verbose=False)
    assert os.path.exists(outfile)
    openpifpaf.export_onnx.check(outfile)

    # openpifpaf.export_onnx.polish(outfile, outfile + '.polished')
    # assert os.path.exists(outfile + '.polished')

    openpifpaf.export_onnx.simplify(outfile, outfile + '.simplified')
    assert os.path.exists(outfile + '.simplified')


@pytest.mark.skipif(sys.platform.startswith('darwin'), reason='onnx simplify crashes')
def test_onnxruntime(tmpdir):
    """Export an onnx model and test outputs.

    This test predicts the outputs of a model with standard OpenPifPaf
    and using onnxruntime from an exported ONNX graph.
    """
    openpifpaf.plugin.register()

    onnx_model_file = str(tmpdir.join('openpifpaf-shufflenetv2k16.onnx'))
    assert not os.path.exists(onnx_model_file)

    # create model
    openpifpaf.plugins.coco.CocoKp.upsample_stride = 2  # create a model with PixelShuffle
    datamodule = openpifpaf.datasets.factory('cocokp')
    model, _ = openpifpaf.network.Factory(
        base_name='shufflenetv2k16',
    ).factory(head_metas=datamodule.head_metas)
    print(model)

    # export to onnx file
    openpifpaf.export_onnx.apply(model, onnx_model_file, verbose=False)

    # pytorch prediction
    dummy_input = torch.randn(1, 3, 97, 129, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred_pytorch = model(dummy_input)

    # onnxruntime prediction
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = onnxruntime.InferenceSession(onnx_model_file, so)

    input_name = ort_session.get_inputs()[0].name
    cif_name = ort_session.get_outputs()[0].name
    caf_name = ort_session.get_outputs()[1].name
    pred_onnx = ort_session.run([cif_name, caf_name], {input_name: dummy_input.numpy()})

    # compare shapes
    assert pred_pytorch[0].shape == pred_onnx[0].shape
    assert pred_pytorch[1].shape == pred_onnx[1].shape

    # compare values
    np.testing.assert_allclose(pred_pytorch[0].numpy(), pred_onnx[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(pred_pytorch[1].numpy(), pred_onnx[1], rtol=1e-03, atol=1e-05)
