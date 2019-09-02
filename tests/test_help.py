import os
import subprocess
import sys

import pytest


PYTHON = 'python3' if sys.platform != 'win32' else 'python'


@pytest.mark.parametrize('module_name', [
    'predict',
    'train',
    'logs',
    pytest.mark.skipif(sys.platform == 'win32', reason='does not run on windows')('webcam'),
    'eval_coco',
    'export_onnx',
    'migrate',
])
def test_predict(module_name):
    out_file = 'docs/cli-help-{}.txt'.format(module_name)
    with open(out_file, 'w') as f:
        subprocess.run([
            PYTHON, '-m', 'openpifpaf.{}'.format(module_name),
            '--help',
        ], stdout=f)

    assert os.path.getsize(out_file) > 0
