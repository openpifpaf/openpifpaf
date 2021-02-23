import os

import openpifpaf


def test_local_checkpoint():
    # make sure model is cached
    _, __ = openpifpaf.network.Factory(checkpoint='shufflenetv2k16').factory()

    local_path = openpifpaf.network.local_checkpoint_path('shufflenetv2k16')
    assert local_path is not None
    assert os.path.exists(local_path)


def test_nossl_checkpoints():
    """There are sometimes issues on Windows with SSL certificates, so avoid
    using SSL checkpoints."""
    openpifpaf.plugin.register()

    for url in openpifpaf.CHECKPOINT_URLS.values():
        if not isinstance(url, str):
            continue
        assert not url.startswith('https')
