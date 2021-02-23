import numpy as np
import torch

import openpifpaf


def test_relative_scale_loss():
    loss = openpifpaf.network.losses.components.Scale()
    x = torch.ones((4,))
    x = torch.log(torch.exp(x) - 1)  # inverse softplus
    t = torch.ones((4,))
    loss_values = loss(x, t)
    np.testing.assert_allclose(loss_values.numpy(), [0, 0, 0, 0], atol=1e-7)


def test_relative_scale_loss_masked():
    loss = openpifpaf.network.losses.components.Scale()
    x = torch.ones((4,))
    x = torch.log(torch.exp(x) - 1)  # inverse softplus
    t = torch.ones((4,))

    mask = x > 1.0
    x = torch.masked_select(x, mask)
    t = torch.masked_select(t, mask)

    loss_values = loss(x, t)
    assert loss_values.sum().numpy() == 0.0
