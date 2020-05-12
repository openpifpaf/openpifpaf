import os

import openpifpaf
import openpifpaf.export_onnx


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
