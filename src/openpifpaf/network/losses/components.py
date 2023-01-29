import argparse
import logging
import math
import typing as t

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


class Base(torch.nn.Module):
    def __init__(self, xi: t.List[int], ti: t.List[int]):
        super().__init__()
        self.xi = xi
        self.ti = ti

    def forward(self, x, t):
        return x[:, :, :, :, self.xi], t[:, :, :, :, self.ti]


class Bce(Base):
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

    def __init__(self, xi: t.List[int], ti: t.List[int], **kwargs):
        super().__init__(xi, ti)
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
        x, t = super().forward(x, t)

        t_zeroone = t.clone()[t >= 0.0]
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
        x, t = super().forward(x, t)

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
        if self.soft_clamp is not None:
            d = self.soft_clamp(d)

        return d


class BceL2(BceDistance):
    def forward(self, x, t):
        d = super().forward(x, t)
        l = torch.nn.functional.smooth_l1_loss(d, torch.zeros_like(d), reduction='none')

        mask_valid = t[:, :, :, :, self.ti] >= 0.0
        mask_foreground = t[mask_valid] > 0.0
        x_logs2 = x[:, :, :, :, 0][mask_valid][mask_foreground]
        x_logs2 = 3.0 * torch.tanh(x_logs2 / 3.0)

        # modify loss for uncertainty
        l[mask_foreground] = 0.5 * l[mask_foreground] * torch.exp(-x_logs2) + 0.5 * x_logs2

        return l


class Scale(Base):
    b = 1.0
    log_space = False
    relative = True
    relative_eps = 0.1
    clip = None
    soft_clamp_value = 5.0

    def __init__(self, xi: t.List[int], ti: t.List[int]):
        super().__init__(xi, ti)

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
        x, t = super().forward(x, t)

        scale_mask = torch.isfinite(t)
        x = x[scale_mask]
        t = t[scale_mask]

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


class Regression(Base):
    soft_clamp_value = 5.0

    def __init__(self, xi: t.List[int], ti: t.List[int], *, sigma_from_scale: float = 0.5):
        super().__init__(xi, ti)
        self.sigma_from_scale = sigma_from_scale

        self.soft_clamp = None
        if self.soft_clamp_value:
            self.soft_clamp = SoftClamp(self.soft_clamp_value)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Regression loss')
        group.add_argument('--regression-soft-clamp', default=cls.soft_clamp_value, type=float,
                           help='soft clamp for scale')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.soft_clamp_value = args.scale_soft_clamp

    def forward(self, x_all, t_all):
        """Only t_regs is guaranteed to be valid.
        Imputes t_sigma_min and and t_scales with guesses.
        """
        x, t = super().forward(x_all, t_all)

        x_regs = x[:, :, :, :, 0:2]
        x_scales = x[:, :, :, :, 2:3]
        t_regs = t[:, :, :, :, 0:2]
        t_sigma_min = t[:, :, :, :, 2:3]
        t_scales = t[:, :, :, :, 3:4]

        finite = torch.isfinite(t_regs)
        reg_mask = torch.all(finite, dim=4)

        x_regs = x_regs[reg_mask]
        x_scales = x_scales[reg_mask].unsqueeze(-1)
        t_regs = t_regs[reg_mask]
        t_sigma_min = t_sigma_min[reg_mask].unsqueeze(-1)
        t_scales = t_scales[reg_mask].unsqueeze(-1)

        # impute t_scales_reg with predicted values
        t_scales = t_scales.clone()
        invalid_t_scales = torch.isnan(t_scales)
        t_scales[invalid_t_scales] = \
            torch.nn.functional.softplus(x_scales.detach()[invalid_t_scales])

        d = x_regs - t_regs
        t_sigma_min_imputed = t_sigma_min.clone()
        t_sigma_min_imputed[torch.isnan(t_sigma_min)] = 0.1
        d = torch.cat([d, t_sigma_min_imputed], dim=2)

        # L2 distance for coordinate pair
        d = torch.linalg.norm(d, ord=2, dim=2, keepdim=True)

        # 68% inside of t_sigma; assume t_scales represents 95%
        t_sigma = self.sigma_from_scale * t_scales
        l = 1.0 / t_sigma * d

        if self.soft_clamp is not None:
            d = self.soft_clamp(d)

        return d
        # uncertainty modification
        x_logs2 = x_all[:, :, :, :, 0][reg_mask]
        x_logs2 = 3.0 * torch.tanh(x_logs2 / 3.0)
        # We want sigma = b*0.5. Therefore, log_b = 0.5 * log_s2 + log2
        x_logb = 0.5 * x_logs2 + 0.69314
        l = l * torch.exp(-x_logb) + x_logb

        return l
