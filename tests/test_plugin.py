import json
import os
import subprocess
import sys

import pytest


PYTHON = 'python3' if sys.platform != 'win32' else 'python'


def test_plugin_autoregistered():
    cmd = [
        PYTHON, '-c',
        'import openpifpaf; assert "testplugin" in openpifpaf.DATAMODULES',
    ]

    cwd = os.path.dirname(__file__)
    subprocess.run(cmd, check=True, cwd=cwd)


def test_plugin_importable():
    cmd = [
        PYTHON, '-c',
        'import openpifpaf_testplugin; print("hello")',
    ]

    cwd = os.path.dirname(__file__)
    subprocess.run(cmd, check=True, cwd=cwd)
