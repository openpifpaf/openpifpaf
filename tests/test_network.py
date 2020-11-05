import os

import openpifpaf


def test_local_checkpoint():
    # make sure model is cached
    _, __ = openpifpaf.network.factory(checkpoint='shufflenetv2k16')

    local_path = openpifpaf.network.local_checkpoint_path('shufflenetv2k16')
    assert local_path is not None
    assert os.path.exists(local_path)
