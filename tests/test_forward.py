import torch

import openpifpaf


def test_forward():
    model, _ = openpifpaf.network.factory(
        base_name='resnet18',
        head_names=['pif', 'paf'],
        pretrained=False,
    )

    dummy_image_batch = torch.zeros((1, 3, 240, 320))
    pif, paf = model(dummy_image_batch)
    assert pif[0].shape == (1, 17, 15, 20)
    assert paf[0].shape == (1, 19, 15, 20)
