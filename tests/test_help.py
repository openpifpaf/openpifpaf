import os
import subprocess

import pytest


@pytest.mark.parametrize('module_name', [
    'predict',
    'train',
    'logs',
    'webcam',
    'eval_coco',
    'export_onnx',
    'migrate',
])
def test_predict(module_name):
    out_file = 'docs/cli-help-{}.txt'.format(module_name)
    with open(out_file, 'w') as f:
        subprocess.run([
            'python', '-m', 'openpifpaf.{}'.format(module_name),
            '--help',
        ], stdout=f)

    assert os.path.getsize(out_file) > 0
