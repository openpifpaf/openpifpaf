import os
import subprocess

import pytest
import torch


@pytest.mark.slow
def test_build(tmpdir):
    assert not os.path.exists(tmpdir.join('openpifpaf-image'))
    assert not os.path.exists(tmpdir.join('openpifpaf-video'))
    assert not os.path.exists(tmpdir.join('libopenpifpafstatic.a'))

    cpp_folder = os.path.join(os.path.dirname(__file__), '..', 'cpp')
    configure_cmd = [
        'cmake',
        '-DCMAKE_PREFIX_PATH={}'.format(torch.utils.cmake_prefix_path),
        cpp_folder,
    ]
    print(configure_cmd)
    subprocess.run(configure_cmd, cwd=tmpdir, check=True)

    build_cmd = [
        'cmake', '--build', '.', '-j4', '--config', 'Release'
    ]
    print(build_cmd)
    subprocess.run(build_cmd, cwd=tmpdir, check=True)

    assert os.path.exists(tmpdir.join('openpifpaf-image'))
    assert os.path.exists(tmpdir.join('openpifpaf-video'))
    assert os.path.exists(tmpdir.join('libopenpifpafstatic.a'))
