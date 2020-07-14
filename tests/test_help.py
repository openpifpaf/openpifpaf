import os
import subprocess
import sys

import pytest

from openpifpaf import __version__


PYTHON = 'python3' if sys.platform != 'win32' else 'python'


MODULE_NAMES = [
    'predict',
    'train',
    'logs',
    'eval_coco',
    'export_onnx',
    'migrate',
    'count_ops',
    'benchmark',
]


if sys.platform != 'win32':
    MODULE_NAMES.append('video')


@pytest.mark.parametrize('module_name', MODULE_NAMES)
def test_help(module_name):
    out_file = 'docs/cli-help-{}.txt'.format(module_name)
    with open(out_file, 'w') as f:
        subprocess.run([
            PYTHON, '-m', 'openpifpaf.{}'.format(module_name),
            '--help',
        ], stdout=f, check=True)

    assert os.path.getsize(out_file) > 0


@pytest.mark.parametrize('module_name', MODULE_NAMES)
def test_version(module_name):
    output = subprocess.check_output([
        PYTHON, '-m', 'openpifpaf.{}'.format(module_name),
        '--version',
    ])
    cli_version = output.decode().strip().replace('.dirty', '')

    assert cli_version == 'OpenPifPaf {}'.format(__version__.replace('.dirty', ''))


@pytest.mark.parametrize('module_name', MODULE_NAMES)
def test_usage(module_name):
    output = subprocess.check_output([
        PYTHON, '-m', 'openpifpaf.{}'.format(module_name),
        '--help',
    ])

    assert output.decode().startswith('usage: python3 -m openpifpaf.{}'.format(module_name))
