import subprocess

import pytest


@pytest.mark.parametrize('module_name', ['predict', 'train', 'logs', 'webcam', 'eval_coco'])
def test_predict(module_name):
    with open('docs/cli-help-{}.txt'.format(module_name), 'w') as f:
        subprocess.run([
            'python', '-m', 'openpifpaf.{}'.format(module_name),
            '--help',
        ], stdout=f)
