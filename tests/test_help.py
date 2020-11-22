import subprocess
import sys

import pytest

from openpifpaf import __version__


PYTHON = 'python3' if sys.platform != 'win32' else 'python'


MODULE_NAMES = [
    'predict',
    'train',
    'logs',
    'eval',
    'export_onnx',
    'migrate',
    'count_ops',
    'benchmark',
]


if sys.platform != 'win32':
    MODULE_NAMES.append('video')


@pytest.mark.parametrize('module_name', MODULE_NAMES)
def test_help(module_name):
    help_text = subprocess.check_output([
        PYTHON, '-m', 'openpifpaf.{}'.format(module_name),
        '--help',
    ])

    assert len(help_text) > 10


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
