import os

import openpifpaf.export_onnx


def test_onnx_exportable():
    openpifpaf.export_onnx.main()
    assert os.path.exists('openpifpaf-resnet50.onnx')
