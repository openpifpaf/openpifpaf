import datetime
import os
import subprocess

import pytest


@pytest.mark.parametrize('batch_size', [1, 2])
def test_predict(batch_size):
    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    test_hash = 'test-clis-{}'.format(now)
    os.makedirs(test_hash)

    subprocess.run([
        'python', '-m', 'openpifpaf.predict',
        '--checkpoint=shufflenetv2x1',
        '--batch-size={}'.format(batch_size),
        '-o', test_hash,
        'docs/coco/000000081988.jpg',
    ])

    assert os.path.exists(test_hash + '/000000081988.jpg.pifpaf.json')
