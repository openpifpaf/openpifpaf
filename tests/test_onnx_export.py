import os
import sys

import numpy as np
import pytest
import torch

import openpifpaf
import openpifpaf.export_onnx

try:
    import onnxruntime
except ImportError:
    onnxruntime = None


@pytest.mark.skipif(onnxruntime is None, reason='onnxruntime unavailable')
def test_dummy(tmpdir):
    outfile = str(tmpdir.join('openpifpaf-dummy.onnx'))
    assert not os.path.exists(outfile)

    model = torch.nn.ReLU()
    openpifpaf.export_onnx.apply(model, outfile, verbose=False,
                                 skip_image_check=True, output_names=["dummy"])
    assert os.path.exists(outfile)
    openpifpaf.export_onnx.check(outfile)


@pytest.mark.skipif(onnxruntime is None, reason='onnxruntime unavailable')
def test_only_base(tmpdir):
    outfile = str(tmpdir.join('openpifpaf-dummy.onnx'))
    assert not os.path.exists(outfile)

    datamodule = openpifpaf.datasets.factory('cocokp')
    model, _ = openpifpaf.network.Factory(
        base_name='shufflenetv2k16',
    ).factory(head_metas=datamodule.head_metas)
    openpifpaf.export_onnx.apply(model.base_net, outfile, verbose=False,
                                 skip_image_check=True, output_names=["dummy"])
    assert os.path.exists(outfile)
    openpifpaf.export_onnx.check(outfile)


@pytest.mark.skipif(onnxruntime is None, reason='onnxruntime unavailable')
def test_cifdet(tmpdir):
    outfile = str(tmpdir.join('openpifpaf-dummy.onnx'))
    assert not os.path.exists(outfile)

    datamodule = openpifpaf.datasets.factory('cocodet')
    model, _ = openpifpaf.network.Factory(
        base_name='shufflenetv2k16',
    ).factory(head_metas=datamodule.head_metas)
    openpifpaf.export_onnx.apply(model.base_net, outfile, verbose=False,
                                 skip_image_check=True, output_names=["cifdet"])
    assert os.path.exists(outfile)
    openpifpaf.export_onnx.check(outfile)


@pytest.mark.skipif(onnxruntime is None, reason='onnxruntime unavailable')
def test_cif_only(tmpdir):
    outfile = str(tmpdir.join('openpifpaf-shufflenetv2k16.onnx'))
    assert not os.path.exists(outfile)

    datamodule = openpifpaf.datasets.factory('cocokp')
    model, _ = openpifpaf.network.Factory(
        base_name='shufflenetv2k16',
    ).factory(head_metas=datamodule.head_metas)
    openpifpaf.export_onnx.apply(model.head_nets[0], outfile,
                                 channels=1392, input_h=1, input_w=1, verbose=False,
                                 skip_image_check=True, output_names=["cif"])
    assert os.path.exists(outfile)
    openpifpaf.export_onnx.check(outfile)


@pytest.mark.skipif(onnxruntime is None, reason='onnxruntime unavailable')
def test_caf_only(tmpdir):
    outfile = str(tmpdir.join('openpifpaf-shufflenetv2k16.onnx'))
    assert not os.path.exists(outfile)

    datamodule = openpifpaf.datasets.factory('cocokp')
    model, _ = openpifpaf.network.Factory(
        base_name='shufflenetv2k16',
    ).factory(head_metas=datamodule.head_metas)
    openpifpaf.export_onnx.apply(model.head_nets[1], outfile,
                                 channels=1392, input_h=1, input_w=1, verbose=False,
                                 skip_image_check=True, output_names=["caf"])
    assert os.path.exists(outfile)
    openpifpaf.export_onnx.check(outfile)


@pytest.mark.skipif(onnxruntime is None, reason='onnxruntime unavailable')
def test_shufflenetv2k16(tmpdir):
    outfile = str(tmpdir.join('openpifpaf-shufflenetv2k16.onnx'))
    assert not os.path.exists(outfile)

    datamodule = openpifpaf.datasets.factory('cocokp')
    model, _ = openpifpaf.network.Factory(
        base_name='shufflenetv2k16',
    ).factory(head_metas=datamodule.head_metas)
    openpifpaf.export_onnx.apply(model, outfile, verbose=False)
    assert os.path.exists(outfile)
    openpifpaf.export_onnx.check(outfile)


@pytest.mark.skip(reason='simplifier is unstable')
@pytest.mark.skipif(sys.version_info >= (3, 9),
                    reason='onnx-simplifier requires py<3.9')
def test_onnx_simplify(tmpdir):
    outfile = str(tmpdir.join('openpifpaf-shufflenetv2k16.onnx'))
    assert not os.path.exists(outfile)

    datamodule = openpifpaf.datasets.factory('cocokp')
    model, _ = openpifpaf.network.Factory(
        base_name='shufflenetv2k16',
    ).factory(head_metas=datamodule.head_metas)
    openpifpaf.export_onnx.apply(model, outfile, verbose=False)
    assert os.path.exists(outfile)
    openpifpaf.export_onnx.check(outfile)

    openpifpaf.export_onnx.simplify(outfile, outfile + '.simplified')
    assert os.path.exists(outfile + '.simplified')


@pytest.mark.skipif(onnxruntime is None, reason='onnxruntime unavailable')
@pytest.mark.parametrize('test_batch_dim', [1, 2])
def test_onnxruntime(tmpdir, test_batch_dim):
    """Export an onnx model and test outputs.

    This test predicts the outputs of a model with standard OpenPifPaf
    and using onnxruntime from an exported ONNX graph.
    """
    if test_batch_dim == 2 and torch.__version__.startswith('1.7'):
        pytest.skip()

    onnx_model_file = str(tmpdir.join('openpifpaf-shufflenetv2k16.onnx'))
    assert not os.path.exists(onnx_model_file)

    # create model
    openpifpaf.plugins.coco.CocoKp.upsample_stride = 2  # create a model with PixelShuffle
    datamodule = openpifpaf.datasets.factory('cocokp')
    model, _ = openpifpaf.network.Factory(
        base_name='shufflenetv2k16',
    ).factory(head_metas=datamodule.head_metas)
    print(model)
    openpifpaf.plugins.coco.CocoKp.upsample_stride = 1  # reset

    # export to onnx file
    openpifpaf.export_onnx.apply(model, onnx_model_file, verbose=False)

    # pytorch prediction
    dummy_input = torch.randn(test_batch_dim, 3, 97, 129, dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
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
