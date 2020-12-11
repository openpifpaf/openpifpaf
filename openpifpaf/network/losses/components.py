import argparse
import logging

import torch

LOG = logging.getLogger(__name__)


class Bce(torch.nn.Module):
    background_weight = 1.0
    focal_gamma = 1.0

    def __init__(self, *, detach_focal=False):
        super().__init__()
        self.detach_focal = detach_focal

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Bce Loss')
        group.add_argument('--background-weight', default=cls.background_weight, type=float,
                           help='BCE weight where ground truth is background')
        group.add_argument('--focal-gamma', default=cls.focal_gamma, type=float,
                           help='when > 0.0, use focal loss with the given gamma')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.background_weight = args.background_weight
        cls.focal_gamma = args.focal_gamma

    def forward(self, x, t):  # pylint: disable=arguments-differ
        t_zeroone = t.clone()
        t_zeroone[t_zeroone > 0.0] = 1.0
        # x = torch.clamp(x, -20.0, 20.0)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            x, t_zeroone, reduction='none')
        # bce = torch.clamp(bce, 0.02, 5.0)  # 0.02 -> -3.9, 0.01 -> -4.6, 0.001 -> -7, 0.0001 -> -9
        bce = torch.clamp_min(bce, 0.02)

        if self.focal_gamma != 0.0:
            pt = torch.exp(-bce)
            focal = (1.0 - pt)**self.focal_gamma
            if self.detach_focal:
                focal = focal.detach()
            bce = focal * bce

        weight_mask = t_zeroone != t
        bce[weight_mask] = bce[weight_mask] * t[weight_mask]

        if self.background_weight != 1.0:
            bg_weight = torch.ones_like(t, requires_grad=False)
            bg_weight[t == 0] *= self.background_weight
            bce = bce * bg_weight

        return bce


class ScaleLoss(torch.nn.Module):
    def __init__(self, b, *,
                 clip=None,
                 relative=False, relative_eps=0.1):
        super().__init__()
        self.b = b
        self.clip = clip
        self.relative = relative
        self.relative_eps = relative_eps

    def forward(self, logs, t):  # pylint: disable=arguments-differ
        loss = torch.nn.functional.l1_loss(
            torch.nn.functional.softplus(logs),
            t,
            reduction='none',
        )
        if self.clip is not None:
            loss = torch.clamp(loss, self.clip[0], self.clip[1])

        denominator = self.b
        if self.relative:
            denominator = self.b * (self.relative_eps + t)
        loss = loss / denominator

        return loss


def laplace_loss(x1, x2, b, t1, t2, bmin, *, weight=None, norm_clip=None):
    """Loss based on Laplace Distribution.

    Loss for a single two-dimensional vector (x1, x2) with radial
    spread b and true (t1, t2) vector.
    """

    # left derivative of sqrt at zero is not defined, so prefer torch.norm():
    # https://github.com/pytorch/pytorch/issues/2421
    # norm = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
    # norm = (torch.stack((x1, x2)) - torch.stack((t1, t2))).norm(dim=0)
    # norm = (
    #     torch.nn.functional.l1_loss(x1, t1, reduction='none')
    #     + torch.nn.functional.l1_loss(x2, t2, reduction='none')
    # )
    # While torch.norm is a special treatment at zero, it does produce
    # large gradients for tiny values (as it should).
    # Similar to BatchNorm, we introduce a physically irrelevant epsilon
    # that stabilizes the gradients for small norms.
    norm = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2 + torch.clamp_min(bmin**2, 0.0001))
    if norm_clip is not None:
        norm = torch.clamp(norm, norm_clip[0], norm_clip[1])

    # constrain range of logb
    # low range constraint: prevent strong confidence when overfitting
    # high range constraint: force some data dependence
    # logb = 3.0 * torch.tanh(logb / 3.0)
    b = torch.nn.functional.softplus(b)
    b = torch.max(b, bmin)

    # ln(2) = 0.694
    losses = torch.log(b) + norm / b
    if weight is not None:
        losses = losses * weight
    return losses


def l1_loss(x1, x2, _, t1, t2, weight=None):
    """L1 loss.

    Loss for a single two-dimensional vector (x1, x2)
    true (t1, t2) vector.
    """
    losses = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
    if weight is not None:
        losses = losses * weight
    return losses


def logl1_loss(logx, t, **kwargs):
    """Swap in replacement for functional.l1_loss."""
    return torch.nn.functional.l1_loss(
        logx, torch.log(t), **kwargs)


class SmoothL1:
    r_smooth = 0.0

    def __init__(self, *, scale_required=True):
        self.scale = None
        self.scale_required = scale_required

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Bce Loss')
        group.add_argument('--r-smooth', type=float, default=cls.r_smooth,
                           help='r_{smooth} for SmoothL1 regressions')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.r_smooth = args.r_smooth

    def __call__(self, x1, x2, _, t1, t2, weight=None):
        """L1 loss.

        Loss for a single two-dimensional vector (x1, x2)
        true (t1, t2) vector.
        """
        if self.scale_required and self.scale is None:
            raise Exception
        if self.scale is None:
            self.scale = 1.0

        r = self.r_smooth * self.scale
        d = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
        smooth_regime = d < r

        smooth_loss = 0.5 / r[smooth_regime] * d[smooth_regime] ** 2
        linear_loss = d[smooth_regime == 0] - (0.5 * r[smooth_regime == 0])
        losses = torch.cat((smooth_loss, linear_loss))

        if weight is not None:
            losses = losses * weight

        self.scale = None
        return torch.sum(losses)
