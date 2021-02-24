import os
import sys

import pytest

import openpifpaf
import openpifpaf.export_coreml


@pytest.mark.skipif(not sys.platform.startswith('darwin'), reason='coreml export only on macos')
def test_coreml_exportable(tmpdir):
    openpifpaf.plugin.register()

    outfile = str(tmpdir.join('openpifpaf-shufflenetv2k16.coreml.mlmodel'))
    assert not os.path.exists(outfile)

    datamodule = openpifpaf.datasets.factory('cocokp')
    model, _ = openpifpaf.network.Factory(
        base_name='shufflenetv2k16',
    ).factory(head_metas=datamodule.head_metas)
    openpifpaf.export_coreml.apply(model, outfile)
    assert os.path.exists(outfile)
