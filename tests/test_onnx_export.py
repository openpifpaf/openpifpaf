import os

import openpifpaf.export_onnx


def test_onnx_exportable(tmpdir):
    outfile = str(tmpdir.join('openpifpaf-resnet50.onnx'))
    assert not os.path.exists(outfile)

    openpifpaf.export_onnx.apply('resnet50', outfile, verbose=False)
    assert os.path.exists(outfile)
