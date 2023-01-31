import numpy as np
import torch

import openpifpaf


def test_relative_scale_loss():
    loss = openpifpaf.network.losses.components.Scale([0], [0])
    x = torch.ones((1, 1, 1, 4, 1))
    x = torch.log(torch.exp(x) - 1)  # inverse softplus
    t = torch.ones((1, 1, 1, 4, 1))

    loss_values = loss(x, t)
    np.testing.assert_allclose(loss_values.numpy(), [0, 0, 0, 0], atol=1e-7)


def test_relative_scale_loss_masked():
    loss = openpifpaf.network.losses.components.Scale([0], [0])
    x = torch.ones((1, 1, 1, 4, 1))
    x = torch.log(torch.exp(x) - 1)  # inverse softplus
    t = torch.ones((1, 1, 1, 4, 1))

    mask = x > 0.5  # select none
    t[mask] = np.nan

    loss_values = loss(x, t)
    assert loss_values.sum().numpy() == 0.0
