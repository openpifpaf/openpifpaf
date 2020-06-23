import os
import pytest
import torch

import openpifpaf
import openpifpaf.export_onnx


@pytest.mark.skipif(not torch.__version__.startswith('1.5'), reason='only PyTorch 1.5')
def test_onnx_exportable(tmpdir):
    outfile = str(tmpdir.join('openpifpaf-shufflenetv2k16w.onnx'))
    assert not os.path.exists(outfile)

    model, _ = openpifpaf.network.factory(
        base_name='shufflenetv2k16w',
        head_names=['cif', 'caf', 'caf25'],
        pretrained=False,
    )
    openpifpaf.export_onnx.apply(model, outfile, verbose=False)
    assert os.path.exists(outfile)
    openpifpaf.export_onnx.check(outfile)

    openpifpaf.export_onnx.polish(outfile, outfile + '.polished')
    assert os.path.exists(outfile + '.polished')

    openpifpaf.export_onnx.simplify(outfile, outfile + '.simplified')
    assert os.path.exists(outfile + '.simplified')
