import os

import openpifpaf.export_onnx


def test_onnx_exportable(tmp_path):
    outfile = os.path.join(tmp_path, 'openpifpaf-resnet50.onnx')
    openpifpaf.export_onnx.apply(checkpoint='resnet50', outfile=outfile)
    assert os.path.exists(outfile)
