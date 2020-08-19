import torch

import openpifpaf


def test_relative_scale_loss():
    loss = openpifpaf.network.losses.ScaleLoss(1.0)
    x = torch.ones((4,)).log()
    t = torch.ones((4,))
    loss_values = loss(x, t)
    assert loss_values.numpy().tolist() == [0, 0, 0, 0]


def test_relative_scale_loss_masked():
    loss = openpifpaf.network.losses.ScaleLoss(1.0)
    x = torch.ones((4,)).log()
    t = torch.ones((4,))

    mask = x > 1.0
    x = torch.masked_select(x, mask)
    t = torch.masked_select(t, mask)

    loss_values = loss(x, t)
    assert loss_values.sum().numpy() == 0.0
