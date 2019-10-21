import os
import subprocess
import sys

import pytest


PYTHON = 'python3' if sys.platform != 'win32' else 'python'


MODULE_NAMES = [
    'predict',
    'train',
    'logs',
    'eval_coco',
    'export_onnx',
    'migrate',
]


if sys.platform != 'win32':
    MODULE_NAMES.append('webcam')


@pytest.mark.parametrize('module_name', MODULE_NAMES)
def test_predict(module_name):
    out_file = 'docs/cli-help-{}.txt'.format(module_name)
    with open(out_file, 'w') as f:
        subprocess.run([
            PYTHON, '-m', 'openpifpaf.{}'.format(module_name),
            '--help',
        ], stdout=f, check=True)

    assert os.path.getsize(out_file) > 0
