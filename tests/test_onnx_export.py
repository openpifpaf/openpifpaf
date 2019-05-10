import os

import openpifpaf.export_onnx


def test_onnx_exportable(tmpdir):
    outfile = tmpdir.join('openpifpaf-resnet50.onnx')
    openpifpaf.export_onnx.apply(checkpoint='resnet50', outfile=outfile)
    assert os.path.exists(outfile)
