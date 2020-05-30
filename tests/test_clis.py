import os
import subprocess
import sys

import pytest


PYTHON = 'python3' if sys.platform != 'win32' else 'python'


@pytest.mark.parametrize('batch_size,with_debug', [(1, False), (2, False), (1, True)])
def test_predict(batch_size, with_debug, tmpdir):
    """Test predict cli.

    with_debug makes sure that debugging works in this environment.
    For example, at some point, --debug required matplotlib which was unintentional.
    """

    if batch_size > 1 and sys.platform.startswith('win'):
        pytest.skip('multiprocess decoding not supported on windows')

    cmd = [
        PYTHON, '-m', 'openpifpaf.predict',
        '--checkpoint=shufflenetv2k16w',
        '--batch-size={}'.format(batch_size),
        '--loader-workers=0',
        '--json-output', str(tmpdir),
        '--long-edge=181',
        'docs/coco/000000081988.jpg',
    ]
    if with_debug:
        cmd.append('--debug')

    subprocess.run(cmd, check=True)
    assert os.path.exists(os.path.join(tmpdir, '000000081988.jpg.predictions.json'))


@pytest.mark.skipif(sys.platform == 'win32', reason='does not run on windows')
@pytest.mark.parametrize('with_debug', [False, True])
def test_video(with_debug, tmpdir):
    cmd = [
        PYTHON, '-m', 'openpifpaf.video',
        '--checkpoint=shufflenetv2k16w',
        '--source=docs/coco/000000081988.jpg',
        '--json-output={}'.format(os.path.join(tmpdir, 'video.json')),
    ]
    if with_debug:
        cmd.append('--debug')

    subprocess.run(cmd, check=True)
    assert os.path.exists(os.path.join(tmpdir, 'video.json'))
