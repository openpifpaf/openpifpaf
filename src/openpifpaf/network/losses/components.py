import argparse
import logging
from typing import List

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
    def __init__(self, xi: List[int], ti: List[int]):
        super().__init__()
        self.xi = xi
        self.ti = ti

    def forward(self, x_all, t_all):
        return x_all[:, :, :, :, self.xi], t_all[:, :, :, :, self.ti]


class Bce(Base):
    focal_alpha = 0.5
    focal_gamma = 1.0
    soft_clamp_value = 5.0

    # choose low value for force-complete-pose and Focal loss modification
    background_clamp = -15.0

    def __init__(self, xi: List[int], ti: List[int], weights=None, **kwargs):
        super().__init__(xi, ti)
        self.weights = weights

        for n, v in kwargs.items():
            assert hasattr(self, n)
            setattr(self, n, v)

        self.soft_clamp = None
        if self.soft_clamp_value:
            self.soft_clamp = SoftClamp(self.soft_clamp_value)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Bce Loss')
        group.add_argument('--focal-alpha', default=cls.focal_alpha, type=float,
                           help='scale parameter of focal loss')
        group.add_argument('--focal-gamma', default=cls.focal_gamma, type=float,
                           help='use focal loss with the given gamma')
        group.add_argument('--bce-soft-clamp', default=cls.soft_clamp_value, type=float,
                           help='soft clamp for BCE')
        group.add_argument('--bce-background-clamp', default=cls.background_clamp, type=float,
                           help='background clamp for BCE')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.focal_alpha = args.focal_alpha
        cls.focal_gamma = args.focal_gamma
        cls.soft_clamp_value = args.bce_soft_clamp
        cls.background_clamp = args.bce_background_clamp

    def forward(self, x_all, t_all):
        x, t = super().forward(x_all, t_all)

        mask = t >= 0.0
        t = t[mask]
        x = x[mask]

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
        l = torch.nn.functional.smooth_l1_loss(x, target, reduction='none')

        # background clamp
        if self.background_clamp:
            l[(x_detached < self.background_clamp) & (t_sign == -1.0)] = 0.0
        if self.soft_clamp is not None:
            l = self.soft_clamp(l)

        mask_foreground = t > 0.0
        x_logs2 = x_all[:, :, :, :, 0:1][mask][mask_foreground]
        x_logs2 = 3.0 * torch.tanh(x_logs2 / 3.0)

        # modify loss for uncertainty
        l[mask_foreground] = 0.5 * l[mask_foreground] * torch.exp(-x_logs2) + 0.5 * x_logs2

        # modify loss for weighting
        if self.weights is not None:
            full_weights = torch.empty_like(t_all[:, :, :, :, 0:1])
            full_weights[:] = self.weights
            l = full_weights[mask] * l

        return l


class Scale(Base):
    b = 1.0
    log_space = False
    relative = True
    relative_eps = 0.1
    clip = None
    soft_clamp_value = 5.0

    def __init__(self, xi: List[int], ti: List[int], weights=None, **kwargs):
        super().__init__(xi, ti)
        self.weights = weights

        for n, v in kwargs.items():
            assert hasattr(self, n)
            setattr(self, n, v)

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

    def forward(self, x_all, t_all):  # pylint: disable=arguments-differ
        x, t = super().forward(x_all, t_all)

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

        # modify loss for weighting
        if self.weights is not None:
            full_weights = torch.empty_like(t_all[:, :, :, :, 0:1])
            full_weights[:] = self.weights
            loss = full_weights[scale_mask] * loss

        return loss


class Regression(Base):
    soft_clamp_value = 5.0

    def __init__(
        self,
        xi: List[int],
        ti: List[int],
        weights=None,
        *,
        sigma_from_scale: float = 0.5,
        scale_from_wh: bool = False,
    ):
        super().__init__(xi, ti)
        self.weights = weights
        self.sigma_from_scale = sigma_from_scale
        self.scale_from_wh = scale_from_wh

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
        if self.scale_from_wh:
            x_scales = torch.linalg.norm(x[:, :, :, :, 2:4], ord=2, dim=4, keepdim=True)
            t_scales = torch.linalg.norm(t[:, :, :, :, 3:5], ord=2, dim=4, keepdim=True)

        finite = torch.isfinite(t_regs)
        reg_mask = torch.all(finite, dim=4)

        x_regs = x_regs[reg_mask]
        x_scales = x_scales[reg_mask]
        t_regs = t_regs[reg_mask]
        t_sigma_min = t_sigma_min[reg_mask]
        t_scales = t_scales[reg_mask]

        # impute t_scales_reg with predicted values
        t_scales = t_scales.clone()
        invalid_t_scales = torch.isnan(t_scales)
        t_scales[invalid_t_scales] = \
            torch.nn.functional.softplus(x_scales.detach()[invalid_t_scales])

        d = x_regs - t_regs
        t_sigma_min_imputed = t_sigma_min.clone()
        t_sigma_min_imputed[torch.isnan(t_sigma_min)] = 0.1
        d = torch.cat([d, t_sigma_min_imputed], dim=1)

        # L2 distance for coordinate pair
        d = torch.linalg.norm(d, ord=2, dim=1, keepdim=True)

        # 68% inside of t_sigma
        t_sigma = self.sigma_from_scale * t_scales
        l = 1.0 / t_sigma * d

        if self.soft_clamp is not None:
            l = self.soft_clamp(l)

        # uncertainty modification
        x_logs2 = x_all[:, :, :, :, 0:1][reg_mask]
        x_logs2 = 3.0 * torch.tanh(x_logs2 / 3.0)
        # We want sigma = b*0.5. Therefore, log_b = 0.5 * log_s2 + log2
        x_logb = 0.5 * x_logs2 + 0.69314
        l = l * torch.exp(-x_logb) + x_logb

        # modify loss for weighting
        if self.weights is not None:
            full_weights = torch.empty_like(t_all[:, :, :, :, 0:1])
            full_weights[:] = self.weights
            l = full_weights[reg_mask] * l

        return l
