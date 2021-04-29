import os
import subprocess
import sys


PYTHON = 'python3' if sys.platform != 'win32' else 'python'


def test_plugin_autoregistered():
    cmd = [
        PYTHON, '-c',
        'import openpifpaf; assert "testplugin" in openpifpaf.DATAMODULES',
    ]

    cwd = os.path.dirname(__file__)
    subprocess.run(cmd, check=True, cwd=cwd)


def test_plugin_importable():
    """Plugin is imported first.

    This triggers registration of all plugins except the current one.
    """

    cmd = [
        PYTHON, '-c',
        (
            'import openpifpaf_testplugin; '
            'import openpifpaf; '
            'assert "testplugin" not in openpifpaf.DATAMODULES; '
            'openpifpaf_testplugin.register(); '
            'assert "testplugin" in openpifpaf.DATAMODULES; '
        ),
    ]

    cwd = os.path.dirname(__file__)
    subprocess.run(cmd, check=True, cwd=cwd)
