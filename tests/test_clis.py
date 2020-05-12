import os
import subprocess
import sys

import pytest


PYTHON = 'python3' if sys.platform != 'win32' else 'python'


@pytest.mark.parametrize('batch_size', [1, 2])
def test_predict(batch_size, tmpdir):
    subprocess.run([
        PYTHON, '-m', 'openpifpaf.predict',
        '--checkpoint=shufflenetv2k16w',
        '--batch-size={}'.format(batch_size),
        '--loader-workers=0',
        '--json-output', str(tmpdir),
        '--long-edge=181',
        'docs/coco/000000081988.jpg',
    ], check=True)

    assert os.path.exists(os.path.join(tmpdir, '000000081988.jpg.predictions.json'))


@pytest.mark.skipif(sys.platform == 'win32', reason='does not run on windows')
def test_video(tmpdir):
    subprocess.run([
        PYTHON, '-m', 'openpifpaf.video',
        '--checkpoint=shufflenetv2k16w',
        '--source=docs/coco/000000081988.jpg',
        '--json-output={}'.format(os.path.join(tmpdir, 'video.json')),
    ], check=True)
