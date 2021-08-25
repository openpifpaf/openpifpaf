import argparse
import logging
import math

import torch

LOG = logging.getLogger(__name__)


class SoftClamp(torch.nn.Module):
    def __init__(self, max_value):
        super().__init__()
        self.max_value = max_value

    def forward(self, x):
        # Backprop rule pre-multiplies by input. Therefore, for a constant
        # gradient above the max bce threshold, need to divide by the input.
        # Just like gradient-clipping, but inline:
        above_max = x > self.max_value
        # x[above_max] /= th[above_max].detach() / self.max_value
        x[above_max] = self.max_value + torch.log(1 - self.max_value + x[above_max])

        return x


class Bce(torch.nn.Module):
    background_weight = 1.0
    focal_alpha = 0.5
    focal_gamma = 1.0
    focal_detach = False
    focal_clamp = True
    soft_clamp_value = 5.0

    # choose low value for force-complete-pose and Focal loss modification
    background_clamp = -15.0

    # 0.02 -> -3.9, 0.01 -> -4.6, 0.001 -> -7, 0.0001 -> -9
    min_bce = 0.0  # 1e-6 corresponds to x~=14, 1e-10 -> 20

    def __init__(self, **kwargs):
        super().__init__()
        for n, v in kwargs.items():
            assert hasattr(self, n)
            setattr(self, n, v)

        self.soft_clamp = None
        if self.soft_clamp_value:
            self.soft_clamp = SoftClamp(self.soft_clamp_value)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Bce Loss')
        group.add_argument('--background-weight', default=cls.background_weight, type=float,
                           help='BCE weight where ground truth is background')
        group.add_argument('--focal-alpha', default=cls.focal_alpha, type=float,
                           help='scale parameter of focal loss')
        group.add_argument('--focal-gamma', default=cls.focal_gamma, type=float,
                           help='use focal loss with the given gamma')
        assert not cls.focal_detach
        group.add_argument('--focal-detach', default=False, action='store_true')
        assert cls.focal_clamp
        group.add_argument('--no-focal-clamp', dest='focal_clamp',
                           default=True, action='store_false')
        group.add_argument('--bce-min', default=cls.min_bce, type=float,
                           help='gradient clipped below')
        group.add_argument('--bce-soft-clamp', default=cls.soft_clamp_value, type=float,
                           help='soft clamp for BCE')
        group.add_argument('--bce-background-clamp', default=cls.background_clamp, type=float,
                           help='background clamp for BCE')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.background_weight = args.background_weight
        cls.focal_alpha = args.focal_alpha
        cls.focal_gamma = args.focal_gamma
        cls.focal_detach = args.focal_detach
        cls.focal_clamp = args.focal_clamp
        cls.min_bce = args.bce_min
        cls.soft_clamp_value = args.bce_soft_clamp
        cls.background_clamp = args.bce_background_clamp

    def forward(self, x, t):  # pylint: disable=arguments-differ
        t_zeroone = t.clone()
        t_zeroone[t_zeroone > 0.0] = 1.0
        # x = torch.clamp(x, -20.0, 20.0)
        if self.background_clamp is not None:
            bg_clamp_mask = (t_zeroone == 0.0) & (x < self.background_clamp)
            x[bg_clamp_mask] = self.background_clamp
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            x, t_zeroone, reduction='none')
        # torch.clamp_max_(bce, 10.0)
        if self.soft_clamp is not None:
            bce = self.soft_clamp(bce)
        if self.min_bce > 0.0:
            torch.clamp_min_(bce, self.min_bce)

        if self.focal_gamma != 0.0:
            p = torch.sigmoid(x)
            pt = p * t_zeroone + (1 - p) * (1 - t_zeroone)
            # Above code is more stable than deriving pt from bce: pt = torch.exp(-bce)

            if self.focal_clamp and self.min_bce > 0.0:
                pt_threshold = math.exp(-self.min_bce)
                torch.clamp_max_(pt, pt_threshold)

            focal = 1.0 - pt
            if self.focal_gamma != 1.0:
                focal = (focal + 1e-4)**self.focal_gamma

            if self.focal_detach:
                focal = focal.detach()

            bce = focal * bce

        if self.focal_alpha == 0.5:
            bce = 0.5 * bce
        elif self.focal_alpha >= 0.0:
            alphat = self.focal_alpha * t_zeroone + (1 - self.focal_alpha) * (1 - t_zeroone)
            bce = alphat * bce

        weight_mask = t_zeroone != t
        bce[weight_mask] = bce[weight_mask] * t[weight_mask]

        if self.background_weight != 1.0:
            bg_weight = torch.ones_like(t, requires_grad=False)
            bg_weight[t == 0] *= self.background_weight
            bce = bce * bg_weight

        return bce


class BceDistance(Bce):
    def forward(self, x, t):
        t_sign = t.clone()
        t_sign[t > 0.0] = 1.0
        t_sign[t <= 0.0] = -1.0

        # construct target location relative to x but without backpropagating through x
        x_detached = x.detach()
        focal_loss_modification = 1.0
        p_bar = 1.0 / (1.0 + torch.exp(t_sign * x_detached))
        if self.focal_alpha:
            focal_loss_modification *= self.focal_alpha
        if self.focal_gamma == 1.0:
            p = 1.0 - p_bar

            # includes simplifications for numerical stability
            # neg_ln_p = torch.log(1 + torch.exp(-t_sign * x_detached))
            # even more stability:
            neg_ln_p = torch.nn.functional.softplus(-t_sign * x_detached)

            focal_loss_modification = focal_loss_modification * (p_bar + p * neg_ln_p)
        elif self.focal_gamma > 0.0:
            p = 1.0 - p_bar
            neg_ln_p = torch.nn.functional.softplus(-t_sign * x_detached)

            focal_loss_modification = focal_loss_modification * (
                p_bar ** self.focal_gamma
                + self.focal_gamma * p_bar ** (self.focal_gamma - 1.0) * p * neg_ln_p
            )
        elif self.focal_gamma == 0.0:
            pass
        else:
            raise NotImplementedError
        target = x_detached + t_sign * p_bar * focal_loss_modification

        # construct distance with x that backpropagates gradients
        d = x - target

        # background clamp
        if self.background_clamp:
            d[(x_detached < self.background_clamp) & (t_sign == -1.0)] = 0.0

        return d


class BceL2(BceDistance):
    def forward(self, x, t):
        d = super().forward(x, t)
        return torch.nn.functional.smooth_l1_loss(d, torch.zeros_like(d), reduction='none')


class Scale(torch.nn.Module):
    b = 1.0
    log_space = False
    relative = True
    relative_eps = 0.1
    clip = None
    soft_clamp_value = 5.0

    def __init__(self):
        super().__init__()

        self.soft_clamp = None
        if self.soft_clamp_value:
            self.soft_clamp = SoftClamp(self.soft_clamp_value)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Scale Loss')
        group.add_argument('--b-scale', default=cls.b, type=float,
                           help='Laplace width b for scale loss')
        assert not cls.log_space
        group.add_argument('--scale-log', default=False, action='store_true')
        group.add_argument('--scale-soft-clamp', default=cls.soft_clamp_value, type=float,
                           help='soft clamp for scale')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.b = args.b_scale
        cls.log_space = args.scale_log
        if args.scale_log:
            cls.relative = False
        cls.soft_clamp_value = args.scale_soft_clamp

    def forward(self, x, t):  # pylint: disable=arguments-differ
        assert not (self.log_space and self.relative)

        x = torch.nn.functional.softplus(x)
        d = torch.nn.functional.l1_loss(
            x if not self.log_space else torch.log(x),
            t if not self.log_space else torch.log(t),
            reduction='none',
        )
        if self.clip is not None:
            d = torch.clamp(d, self.clip[0], self.clip[1])

        denominator = self.b
        if self.relative:
            denominator = self.b * (self.relative_eps + t)
        d = d / denominator

        if self.soft_clamp is not None:
            d = self.soft_clamp(d)

        loss = torch.nn.functional.smooth_l1_loss(d, torch.zeros_like(d), reduction='none')
        return loss


class ScaleDistance(Scale):
    def forward(self, x, t):
        x = torch.nn.functional.softplus(x)
        d = 1.0 / self.b * (x - t)
        d[torch.isnan(d)] = 0.0
        return d


class Laplace(torch.nn.Module):
    """Loss based on Laplace Distribution.

    Loss for a single two-dimensional vector (x1, x2) with radial
    spread b and true (t1, t2) vector.
    """

    weight = None
    norm_clip = None
    soft_clamp_value = 5.0

    def __init__(self):
        super().__init__()

        self.soft_clamp = None
        if self.soft_clamp_value:
            self.soft_clamp = SoftClamp(self.soft_clamp_value)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Laplace Loss')
        group.add_argument('--laplace-soft-clamp', default=cls.soft_clamp_value, type=float,
                           help='soft clamp for Laplace')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.soft_clamp_value = args.laplace_soft_clamp

    def forward(self, x1, x2, logb, t1, t2, bmin):
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
        # norm = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2 + torch.clamp_min(bmin**2, 0.0001))
        # norm = torch.stack((x1 - t1, x2 - t2, torch.clamp_min(bmin, 0.01))).norm(dim=0)
        norm = torch.stack((x1 - t1, x2 - t2, bmin)).norm(dim=0)
        if self.norm_clip is not None:
            norm = torch.clamp(norm, self.norm_clip[0], self.norm_clip[1])

        # constrain range of logb
        # logb = logb + 3.0  # shift logb such that weight decay relaxes to large b
        # low range constraint: prevent strong confidence when overfitting
        # high range constraint: force some data dependence
        logb = 3.0 * torch.tanh(logb / 3.0)
        # b = torch.nn.functional.softplus(b)
        # b = torch.max(b, bmin)
        # b_plus_bmin = torch.nn.functional.softplus(b) + bmin
        # b_plus_bmin = 20.0 * torch.sigmoid(b / 20.0) + bmin
        # logb = -3.0 + torch.nn.functional.softplus(logb + 3.0)
        # log_bmin = torch.log(bmin)
        # logb = log_bmin + torch.nn.functional.softplus(logb - log_bmin)

        # ln(2) = 0.694
        # losses = torch.log(b_plus_bmin) + norm / b_plus_bmin
        scaled_norm = norm * torch.exp(-logb)
        if self.soft_clamp is not None:
            scaled_norm = self.soft_clamp(scaled_norm)
        losses = logb + scaled_norm
        if self.weight is not None:
            losses = losses * self.weight
        return losses


class RegressionDistance:
    @staticmethod
    def __call__(x_regs, t_regs, t_sigma_min, t_scales):
        d = x_regs - t_regs
        d = torch.cat([d, t_sigma_min], dim=2)

        # L2 distance for coordinate pair
        d_shape = d.shape
        d = d.reshape(d_shape[0], d_shape[1], -1, 3, d_shape[-2], d_shape[-1])
        d[torch.isnan(d)] = float('inf')
        d = torch.linalg.norm(d, ord=2, dim=3)
        d[~torch.isfinite(d)] = 0.0

        # 68% inside of t_sigma
        if t_scales.shape[2]:
            t_sigma = 0.5 * t_scales
            t_sigma[torch.isnan(t_sigma)] = 0.5  # assume a sigma when not given
        elif t_regs.shape[2] == 4:
            # two regressions without scales is detection, i.e. second
            # regression targets are width and height
            t_scales = torch.linalg.norm(0.5 * t_regs[:, :, 2:], ord=2, dim=2, keepdim=True)
            t_sigma = 0.5 * t_scales
            t_sigma[torch.isnan(t_sigma)] = 5.0  # assume a sigma when not given
            t_sigma = torch.repeat_interleave(t_sigma, 2, dim=2)
        else:
            t_sigma = 0.5
        d = 1.0 / t_sigma * d

        return d


class RegressionLoss:
    @staticmethod
    def __call__(x_regs, t_regs, t_sigma_min, t_scales):
        """Only t_regs is guaranteed to be valid.
        Imputes t_sigma_min and and t_scales with guesses.
        """
        x_regs = x_regs.reshape(t_regs.shape[0], t_regs.shape[1] // 2, 2)
        t_regs = t_regs.reshape(t_regs.shape[0], t_regs.shape[1] // 2, 2)
        t_sigma_min = t_sigma_min.unsqueeze(-1)
        t_scales = t_scales.unsqueeze(-1)

        d = x_regs - t_regs
        t_sigma_min_imputed = t_sigma_min.clone()
        t_sigma_min_imputed[torch.isnan(t_sigma_min)] = 0.1
        d = torch.cat([d, t_sigma_min_imputed], dim=2)

        # L2 distance for coordinate pair
        d = torch.linalg.norm(d, ord=2, dim=2, keepdim=True)

        # 68% inside of t_sigma
        if t_scales.shape[1]:
            t_sigma = 0.5 * t_scales
            t_sigma[torch.isnan(t_sigma)] = 0.5
        elif t_regs.shape[1] == 2:
            # Two regressions without scales is detection, i.e. second
            # regression targets are width and height.
            # Assume a regression sigma of 10% of the scale.
            t_scales = torch.linalg.norm(t_regs[:, 1:2], ord=2, dim=2, keepdim=True)
            t_sigma = 0.1 * t_scales
            t_sigma = torch.repeat_interleave(t_sigma, 2, dim=1)
        else:
            t_sigma = 0.5
        d = 1.0 / t_sigma * d

        return d


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
