import argparse
from collections import defaultdict
import os
import subprocess
import sys

import openpifpaf


PYTHON = 'python3' if sys.platform != 'win32' else 'python'


def test_autoregistered():
    cmd = [
        PYTHON, '-c',
        'import openpifpaf; assert "testplugin" in openpifpaf.DATAMODULES',
    ]

    cwd = os.path.dirname(__file__)
    subprocess.run(cmd, check=True, cwd=cwd)


def test_importable():
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


def test_cli_respects_namespace():
    conflicts = defaultdict(list)

    for name, module in openpifpaf.datasets.DATAMODULES.items():
        if name.startswith('test'):
            continue

        print(name, module)
        parser = argparse.ArgumentParser()
        module.cli(parser)
        arguments = parser.parse_args([])

        for arg, _ in arguments._get_kwargs():  # pylint: disable=protected-access
            # allow --cocokp-... in cocokp module
            if arg.startswith(name + '_'):
                continue

            # allow --coco-eval... in cocokp module
            if name.startswith(arg.partition('_')[0]):
                continue

            conflicts[name].append(arg)

    if conflicts:
        raise Exception(' '.join(
            f'Cli arguments {args} defined in the "{name}" module must start with "{name}_".'
            for name, args in conflicts.items()
        ))
