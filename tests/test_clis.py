import datetime
import os
import subprocess


def test_predict():
    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    test_hash = 'test-clis-{}'.format(now)
    os.makedirs(test_hash)

    subprocess.run([
        'python', '-m', 'openpifpaf.predict',
        'docs/coco/000000081988.jpg',
        '-o', test_hash,
    ])

    assert os.path.exists(test_hash + '/000000081988.jpg.pifpaf.json')
