import argparse
import logging

import torch

from .. import heads
from . import components

LOG = logging.getLogger(__name__)


class CompositeLoss(torch.nn.Module):
    prescale = 1.0
    regression_loss = components.Laplace()
    bce_total_soft_clamp = None

    def __init__(self, head_net: heads.CompositeField3):
        super().__init__()
        self.n_vectors = head_net.meta.n_vectors
        self.n_scales = head_net.meta.n_scales

        LOG.debug('%s: n_vectors = %d, n_scales = %d',
                  head_net.meta.name, self.n_vectors, self.n_scales)

        self.confidence_loss = components.Bce()
        self.scale_losses = torch.nn.ModuleList([
            components.Scale() for _ in range(self.n_scales)])
        self.field_names = (
            ['{}.{}.c'.format(head_net.meta.dataset, head_net.meta.name)]
            + ['{}.{}.vec{}'.format(head_net.meta.dataset, head_net.meta.name, i + 1)
               for i in range(self.n_vectors)]
            + ['{}.{}.scales{}'.format(head_net.meta.dataset, head_net.meta.name, i + 1)
               for i in range(self.n_scales)]
        )

        w = head_net.meta.training_weights
        self.weights = None
        if w is not None:
            self.weights = torch.ones([1, head_net.meta.n_fields, 1, 1], requires_grad=False)
            self.weights[0, :, 0, 0] = torch.Tensor(w)
        LOG.debug("The weights for the keypoints are %s", self.weights)
        self.bce_blackout = None
        self.previous_losses = None

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Composite Loss')
        group.add_argument('--loss-prescale', default=cls.prescale, type=float)
        group.add_argument('--regression-loss', default='laplace',
                           choices=['smoothl1', 'l1', 'laplace'],
                           help='type of regression loss')
        group.add_argument('--bce-total-soft-clamp', default=cls.bce_total_soft_clamp,
                           type=float,
                           help='per feature clamp value applied to the total')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.prescale = args.loss_prescale

        if args.regression_loss == 'smoothl1':
            cls.regression_loss = components.SmoothL1()
        elif args.regression_loss == 'l1':
            cls.regression_loss = staticmethod(components.l1_loss)
        elif args.regression_loss == 'laplace':
            cls.regression_loss = components.Laplace()
        elif args.regression_loss is None:
            cls.regression_loss = components.Laplace()
        else:
            raise Exception('unknown regression loss type {}'.format(args.regression_loss))

        cls.bce_total_soft_clamp = args.bce_total_soft_clamp

    def _confidence_loss(self, x_confidence, t_confidence):
        # TODO assumes one confidence
        x_confidence = x_confidence[:, :, 0]
        t_confidence = t_confidence[:, :, 0]

        bce_masks = torch.isnan(t_confidence).bitwise_not_()
        if not torch.any(bce_masks):
            return None

        batch_size = t_confidence.shape[0]
        n_fields = t_confidence.shape[1]
        n_features = t_confidence.numel()
        LOG.debug('batch size = %d, n fields = %d, n_features = %d',
                  batch_size, n_fields, n_features)

        if self.bce_blackout:
            x_confidence = x_confidence[:, self.bce_blackout]
            bce_masks = bce_masks[:, self.bce_blackout]
            t_confidence = t_confidence[:, self.bce_blackout]

        LOG.debug('BCE: x = %s, target = %s, mask = %s',
                  x_confidence.shape, t_confidence.shape, bce_masks.shape)
        bce_target = torch.masked_select(t_confidence, bce_masks)
        x_confidence = torch.masked_select(x_confidence, bce_masks)
        ce_loss = self.confidence_loss(x_confidence, bce_target)
        if self.prescale != 1.0:
            ce_loss = ce_loss * self.prescale
        if self.weights is not None:
            weight = torch.ones_like(t_confidence, requires_grad=False)
            weight[:] = self.weights
            weight = torch.masked_select(weight, bce_masks)
            ce_loss = ce_loss * weight

        ce_loss = ce_loss.sum()
        if self.bce_total_soft_clamp is not None:
            total_clamp_value = self.bce_total_soft_clamp * n_features / n_fields
            LOG.debug('summed ce loss = %s, soft clamp = %f', ce_loss, total_clamp_value)
            ce_loss = components.SoftClamp(total_clamp_value)(ce_loss)

        ce_loss = ce_loss / batch_size

        return ce_loss

    def _localization_loss(self, x_regs, t_regs):
        assert x_regs.shape[2] == self.n_vectors * 3
        assert t_regs.shape[2] == self.n_vectors * 3
        batch_size = t_regs.shape[0]

        reg_losses = []
        if self.weights is not None:
            weight = torch.ones_like(t_regs[:, :, 0], requires_grad=False)
            weight[:] = self.weights
        for i in range(self.n_vectors):
            reg_masks = torch.isnan(t_regs[:, :, i * 2]).bitwise_not_()
            loss = self.regression_loss(
                torch.masked_select(x_regs[:, :, i * 2 + 0], reg_masks),
                torch.masked_select(x_regs[:, :, i * 2 + 1], reg_masks),
                torch.masked_select(x_regs[:, :, self.n_vectors * 2 + i], reg_masks),
                torch.masked_select(t_regs[:, :, i * 2 + 0], reg_masks),
                torch.masked_select(t_regs[:, :, i * 2 + 1], reg_masks),
                torch.masked_select(t_regs[:, :, self.n_vectors * 2 + i], reg_masks),
            )
            if self.prescale != 1.0:
                loss = loss * self.prescale
            if self.weights is not None:
                loss = loss * torch.masked_select(weight, reg_masks)
            reg_losses.append(loss.sum() / batch_size)

        return reg_losses

    def _scale_losses(self, x_scales, t_scales):
        assert x_scales.shape[2] == t_scales.shape[2] == len(self.scale_losses)

        batch_size = t_scales.shape[0]
        losses = []
        if self.weights is not None:
            weight = torch.ones_like(t_scales[:, :, 0], requires_grad=False)
            weight[:] = self.weights
        for i, sl in enumerate(self.scale_losses):
            mask = torch.isnan(t_scales[:, :, i]).bitwise_not_()
            loss = sl(
                torch.masked_select(x_scales[:, :, i], mask),
                torch.masked_select(t_scales[:, :, i], mask),
            )
            if self.prescale != 1.0:
                loss = loss * self.prescale
            if self.weights is not None:
                loss = loss * torch.masked_select(weight, mask)
            losses.append(loss.sum() / batch_size)

        return losses

    def forward(self, x, t):
        LOG.debug('loss for %s', self.field_names)

        if t is None:
            return [None for _ in range(1 + self.n_vectors + self.n_scales)]
        assert x.shape[2] == 1 + self.n_vectors * 3 + self.n_scales
        assert t.shape[2] == 1 + self.n_vectors * 3 + self.n_scales

        # x = x.double()
        x_confidence = x[:, :, 0:1]
        x_regs = x[:, :, 1:1 + self.n_vectors * 3]
        x_scales = x[:, :, 1 + self.n_vectors * 3:]

        # t = t.double()
        t_confidence = t[:, :, 0:1]
        t_regs = t[:, :, 1:1 + self.n_vectors * 3]
        t_scales = t[:, :, 1 + self.n_vectors * 3:]

        ce_loss = self._confidence_loss(x_confidence, t_confidence)
        reg_losses = self._localization_loss(x_regs, t_regs)
        scale_losses = self._scale_losses(x_scales, t_scales)

        all_losses = [ce_loss] + reg_losses + scale_losses
        if not all(torch.isfinite(l).item() if l is not None else True for l in all_losses):
            raise Exception('found a loss that is not finite: {}, prev: {}'
                            ''.format(all_losses, self.previous_losses))
        self.previous_losses = [float(l.item()) if l is not None else None for l in all_losses]

        return all_losses


class CompositeLaplace(torch.nn.Module):
    prescale = 1.0
    bce_total_soft_clamp = None
    soft_clamp_value = 5.0

    def __init__(self, head_net: heads.CompositeField3):
        super().__init__()
        self.n_vectors = head_net.meta.n_vectors
        self.n_scales = head_net.meta.n_scales

        LOG.debug('%s: n_vectors = %d, n_scales = %d',
                  head_net.meta.name, self.n_vectors, self.n_scales)

        self.field_names = (
            '{}.{}.c'.format(head_net.meta.dataset, head_net.meta.name),
            '{}.{}.vec'.format(head_net.meta.dataset, head_net.meta.name),
            '{}.{}.scales'.format(head_net.meta.dataset, head_net.meta.name),
        )
        self.distance_loss_fn = torch.nn.SmoothL1Loss(reduction='none')
        self.distance_loss_fn_l1 = torch.nn.L1Loss(reduction='none')
        self.distance_loss = lambda d: self.distance_loss_fn(d, torch.zeros_like(d))
        self.distance_loss_l1 = lambda d: self.distance_loss_fn_l1(d, torch.zeros_like(d))
        self.soft_clamp = None
        if self.soft_clamp_value:
            self.soft_clamp = components.SoftClamp(self.soft_clamp_value)

        w = head_net.meta.training_weights
        self.weights = None
        if w is not None:
            self.weights = torch.ones([1, head_net.meta.n_fields, 1, 1], requires_grad=False)
            self.weights[0, :, 0, 0] = torch.Tensor(w)
        LOG.debug("The weights for the keypoints are %s", self.weights)
        self.bce_blackout = None
        self.previous_losses = None

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Composite Laplace')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.prescale = args.loss_prescale

        if args.regression_loss == 'smoothl1':
            cls.regression_loss = components.SmoothL1()
        elif args.regression_loss == 'l1':
            cls.regression_loss = staticmethod(components.l1_loss)
        elif args.regression_loss == 'laplace':
            cls.regression_loss = components.Laplace()
        elif args.regression_loss is None:
            cls.regression_loss = components.Laplace()
        else:
            raise Exception('unknown regression loss type {}'.format(args.regression_loss))

        cls.bce_total_soft_clamp = args.bce_total_soft_clamp

    def _confidence_distance(self, x_confidence, t_confidence):
        t_sign = t_confidence.clone()
        t_sign[t_confidence > 0.0] = 1.0
        t_sign[t_confidence <= 0.0] = -1.0
        # construct target location relative to x but without backpropagating through x
        x = x_confidence.detach()

        focal_loss_modification = 1.0
        p_bar = 1.0 / (1.0 + torch.exp(t_sign * x))
        if components.Bce.focal_alpha:
            focal_loss_modification *= components.Bce.focal_alpha
        if components.Bce.focal_gamma == 1.0:
            p = 1.0 - p_bar

            # includes simplifications for numerical stability
            # neg_ln_p = torch.log(1 + torch.exp(-t_sign * x))
            # even more stability:
            neg_ln_p = torch.nn.functional.softplus(-t_sign * x)

            focal_loss_modification = focal_loss_modification * (p_bar + p * neg_ln_p)
        elif components.Bce.focal_gamma == 0.0:
            pass
        else:
            raise NotImplementedError
        target = x + t_sign * p_bar * focal_loss_modification

        # construct distance with x that backpropagates gradients
        d = x_confidence - target

        # background clamp
        if components.Bce.background_clamp:
            d[(x < components.Bce.background_clamp) & (t_sign == -1.0)] = 0.0

        # nan target
        d[torch.isnan(t_confidence)] = 0.0

        return d

    def _reg_distance(self, x_regs, t_regs, t_sigma_min, t_scales):
        d = x_regs - t_regs
        d = torch.cat([d, t_sigma_min], dim=2)
        d[torch.isnan(d)] = 0.0

        # L2 distance for coordinate pair
        d_shape = d.shape
        d = d.reshape(d_shape[0], d_shape[1], -1, 3, d_shape[-2], d_shape[-1])
        d = torch.linalg.norm(d, ord=2, dim=3)

        # 68% inside of t_sigma
        if t_scales.shape[2]:
            t_sigma = 0.5 * t_scales
            t_sigma[torch.isnan(t_sigma)] = 0.5  # assume a sigma when not given
        elif t_regs.shape[2] == 4:
            # two regressions without scales is detection, i.e. second
            # regression targets are width and height
            t_scales = torch.linalg.norm(0.3 * t_regs[:, :, 2:], ord=2, dim=2, keepdim=True)
            t_sigma = 0.5 * t_scales
            t_sigma[torch.isnan(t_sigma)] = 3.0  # assume a sigma when not given
            t_sigma = torch.repeat_interleave(t_sigma, 2, dim=2)
        else:
            t_sigma = 0.5
        d = 1.0 / t_sigma * d

        return d

    def _scale_distance(self, x_scales, t_scales):
        d = 0.1 * (x_scales - t_scales)
        d[torch.isnan(d)] = 0.0
        return d

    def forward(self, x, t):
        LOG.debug('loss for %s', self.field_names)

        if t is None:
            return [None for _ in range(1 + self.n_vectors + self.n_scales)]
        assert x.shape[2] == 1 + self.n_vectors * 3 + self.n_scales
        assert t.shape[2] == 1 + self.n_vectors * 3 + self.n_scales

        # x = x.double()
        x_confidence = x[:, :, 0:1]
        x_regs = x[:, :, 1:1 + self.n_vectors * 2]
        x_logs2 = x[:, :, 1 + self.n_vectors * 2]
        x_scales = x[:, :, 1 + self.n_vectors * 3:]

        # t = t.double()
        t_confidence = t[:, :, 0:1]
        t_regs = t[:, :, 1:1 + self.n_vectors * 2]
        t_sigma_min = t[:, :, 1 + self.n_vectors * 2:1 + self.n_vectors * 3]
        t_scales = t[:, :, 1 + self.n_vectors * 3:]

        d_confidence = self._confidence_distance(x_confidence, t_confidence)
        d_reg = self._reg_distance(x_regs, t_regs, t_sigma_min, t_scales)
        d_scale = self._scale_distance(x_scales, t_scales)

        l_confidence = self.distance_loss(d_confidence)
        l_reg = self.distance_loss_l1(d_reg)
        l_scale = self.distance_loss(d_scale)

        # softclamp
        if self.soft_clamp is not None:
            l_confidence = self.soft_clamp(l_confidence)
            l_reg = self.soft_clamp(l_reg)
            l_scale = self.soft_clamp(l_scale)

        reg_mask = torch.isfinite(torch.sum(t_sigma_min, dim=2))
        x_logs2 = 3.0 * torch.tanh(x_logs2[reg_mask] / 3.0)
        l_confidence = l_confidence[:, :, 0]
        l_confidence[reg_mask] = l_confidence[reg_mask] * torch.exp(-x_logs2) + 0.5 * x_logs2
        for i in range(l_reg.shape[2]):
            # We want b=sigma. Therefore, log_b = 0.5 * log_s2
            x_logb = 0.5 * x_logs2
            l_reg_component = l_reg[:, :, i]
            l_reg_component[reg_mask] = l_reg_component[reg_mask] * torch.exp(-x_logb) + x_logb
        # l_scale[reg_mask] = l_scale[reg_mask] * torch.exp(-x_logs) + x_logs

        batch_size = t.shape[0]
        losses = [
            torch.sum(l_confidence) / batch_size,
            torch.sum(l_reg) / batch_size,
            torch.sum(l_scale) / batch_size,
        ]

        if not all(torch.isfinite(l).item() if l is not None else True for l in losses):
            raise Exception('found a loss that is not finite: {}, prev: {}'
                            ''.format(losses, self.previous_losses))
        self.previous_losses = [float(l.item()) if l is not None else None for l in losses]

        return losses
