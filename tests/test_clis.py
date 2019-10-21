import datetime
import os
import subprocess
import sys

import pytest


PYTHON = 'python3' if sys.platform != 'win32' else 'python'


@pytest.mark.parametrize('batch_size', [1, 2])
def test_predict(batch_size):
    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    test_hash = 'test-clis-{}'.format(now)
    os.makedirs(test_hash)

    subprocess.run([
        PYTHON, '-m', 'openpifpaf.predict',
        '--checkpoint=shufflenetv2x1',
        '--batch-size={}'.format(batch_size),
        '--loader-workers=0',
        '--output-types', 'json',
        '-o', test_hash,
        'docs/coco/000000081988.jpg',
    ], check=True)

    assert os.path.exists(test_hash + '/000000081988.jpg.pifpaf.json')


@pytest.mark.skipif(sys.platform == 'win32', reason='does not run on windows')
def test_webcam():
    subprocess.run([
        PYTHON, '-m', 'openpifpaf.webcam',
        '--source=docs/coco/000000081988.jpg',
    ], check=True)
