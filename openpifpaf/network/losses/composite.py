import argparse
import logging

import torch

from . import components

LOG = logging.getLogger(__name__)


class CompositeLossByComponent(torch.nn.Module):
    """Default loss until v0.12"""

    prescale = 1.0
    regression_loss = components.Laplace()
    bce_total_soft_clamp = None

    def __init__(self, head_meta):
        super().__init__()
        self.n_vectors = head_meta.n_vectors
        self.n_scales = head_meta.n_scales

        LOG.debug('%s: n_vectors = %d, n_scales = %d',
                  head_meta.name, self.n_vectors, self.n_scales)

        self.confidence_loss = components.Bce()
        self.scale_losses = torch.nn.ModuleList([
            components.Scale() for _ in range(self.n_scales)])
        self.field_names = (
            ['{}.{}.c'.format(head_meta.dataset, head_meta.name)]
            + ['{}.{}.vec{}'.format(head_meta.dataset, head_meta.name, i + 1)
               for i in range(self.n_vectors)]
            + ['{}.{}.scales{}'.format(head_meta.dataset, head_meta.name, i + 1)
               for i in range(self.n_scales)]
        )

        w = head_meta.training_weights
        self.weights = None
        if w is not None:
            self.weights = torch.ones([1, head_meta.n_fields, 1, 1], requires_grad=False)
            self.weights[0, :, 0, 0] = torch.Tensor(w)
        LOG.debug("The weights for the keypoints are %s", self.weights)
        self.bce_blackout = None
        self.previous_losses = None

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Composite Loss by Components')
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


class CompositeLoss(torch.nn.Module):
    """Default loss since v0.13"""

    soft_clamp_value = 5.0

    def __init__(self, head_meta):
        super().__init__()
        self.n_confidences = head_meta.n_confidences
        self.n_vectors = head_meta.n_vectors
        self.n_scales = head_meta.n_scales

        LOG.debug('%s: n_vectors = %d, n_scales = %d',
                  head_meta.name, self.n_vectors, self.n_scales)

        self.field_names = (
            '{}.{}.c'.format(head_meta.dataset, head_meta.name),
            '{}.{}.vec'.format(head_meta.dataset, head_meta.name),
            '{}.{}.scales'.format(head_meta.dataset, head_meta.name),
        )

        self.bce_loss = components.BceL2()
        self.reg_loss = components.RegressionLoss()
        self.scale_loss = components.Scale()

        self.soft_clamp = None
        if self.soft_clamp_value:
            self.soft_clamp = components.SoftClamp(self.soft_clamp_value)

        self.weights = None
        if head_meta.training_weights is not None:
            assert len(head_meta.training_weights) == head_meta.n_fields
            self.weights = torch.Tensor(head_meta.training_weights).reshape(1, -1, 1, 1, 1)

        LOG.debug("The weights for the keypoints are %s", self.weights)
        self.bce_blackout = None
        self.previous_losses = None

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Composite Loss')
        group.add_argument('--soft-clamp', default=cls.soft_clamp_value, type=float,
                           help='soft clamp')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.soft_clamp_value = args.soft_clamp

    # pylint: disable=too-many-statements
    def forward(self, x, t):
        LOG.debug('loss for %s', self.field_names)

        if t is None:
            return [None, None, None]
        assert x.shape[2] == 1 + self.n_confidences + self.n_vectors * 2 + self.n_scales
        assert t.shape[2] == self.n_confidences + self.n_vectors * 3 + self.n_scales

        # determine foreground and background masks based on ground truth
        t = torch.transpose(t, 2, 4)
        finite = torch.isfinite(t)
        t_confidence_raw = t[:, :, :, :, 0:self.n_confidences]
        bg_mask = torch.all(t_confidence_raw == 0.0, dim=4)
        c_mask = torch.all(t_confidence_raw > 0.0, dim=4)
        reg_mask = torch.all(finite[:, :, :, :, self.n_confidences:1 + self.n_vectors * 2], dim=4)
        scale_mask = torch.all(finite[:, :, :, :, self.n_confidences + self.n_vectors * 3:], dim=4)

        # extract masked ground truth
        t_confidence_bg = t[bg_mask][:, 0:self.n_confidences]
        t_confidence = t[c_mask][:, 0:self.n_confidences]
        t_regs = t[reg_mask][:, self.n_confidences:1 + self.n_vectors * 2]
        t_sigma_min = t[reg_mask][
            :,
            self.n_confidences + self.n_vectors * 2:self.n_confidences + self.n_vectors * 3
        ]
        t_scales_reg = t[reg_mask][:, self.n_confidences + self.n_vectors * 3:]
        t_scales = t[scale_mask][:, self.n_confidences + self.n_vectors * 3:]

        # extract masked predictions
        x = torch.transpose(x, 2, 4)
        x_confidence_bg = x[bg_mask][:, 1:1 + self.n_confidences]
        x_logs2_c = x[c_mask][:, 0:1]
        x_confidence = x[c_mask][:, 1:1 + self.n_confidences]
        x_logs2_reg = x[reg_mask][:, 0:1]
        x_regs = x[reg_mask][:, 1 + self.n_confidences:1 + self.n_confidences + self.n_vectors * 2]
        # x_logs2_scale = x[scale_mask][:, 0:1]
        x_scales_reg = x[reg_mask][:, 1 + self.n_confidences + self.n_vectors * 2:]
        x_scales = x[scale_mask][:, 1 + self.n_confidences + self.n_vectors * 2:]

        # impute t_scales_reg with predicted values
        t_scales_reg = t_scales_reg.clone()
        invalid_t_scales_reg = torch.isnan(t_scales_reg)
        t_scales_reg[invalid_t_scales_reg] = \
            torch.nn.functional.softplus(x_scales_reg.detach()[invalid_t_scales_reg])

        l_confidence_bg = self.bce_loss(x_confidence_bg, t_confidence_bg)
        l_confidence = self.bce_loss(x_confidence, t_confidence)
        l_reg = self.reg_loss(x_regs, t_regs, t_sigma_min, t_scales_reg)
        l_scale = self.scale_loss(x_scales, t_scales)

        # softclamp
        if self.soft_clamp is not None:
            l_confidence_bg = self.soft_clamp(l_confidence_bg)
            l_confidence = self.soft_clamp(l_confidence)
            l_reg = self.soft_clamp(l_reg)
            l_scale = self.soft_clamp(l_scale)

        # --- composite uncertainty
        # c
        x_logs2_c = 3.0 * torch.tanh(x_logs2_c / 3.0)
        l_confidence = 0.5 * l_confidence * torch.exp(-x_logs2_c) + 0.5 * x_logs2_c
        # reg
        x_logs2_reg = 3.0 * torch.tanh(x_logs2_reg / 3.0)
        # We want sigma = b*0.5. Therefore, log_b = 0.5 * log_s2 + log2
        x_logb = 0.5 * x_logs2_reg + 0.69314
        reg_factor = torch.exp(-x_logb)
        x_logb = x_logb.unsqueeze(1)
        reg_factor = reg_factor.unsqueeze(1)
        if self.n_vectors > 1:
            x_logb = torch.repeat_interleave(x_logb, self.n_vectors, 1)
            reg_factor = torch.repeat_interleave(reg_factor, self.n_vectors, 1)
        l_reg = l_reg * reg_factor + x_logb
        # scale
        # scale_factor = torch.exp(-x_logs2)
        # for i in range(self.n_scales):
        #     l_scale_component = l_scale[:, i]
        #     l_scale_component = l_scale_component * scale_factor + 0.5 * x_logs2

        if self.weights is not None:
            full_weights = torch.empty_like(t_confidence_raw)
            full_weights[:] = self.weights
            l_confidence_bg = full_weights[bg_mask] * l_confidence_bg
            l_confidence = full_weights[c_mask] * l_confidence
            l_reg = full_weights.unsqueeze(-1)[reg_mask] * l_reg
            l_scale = full_weights[scale_mask] * l_scale

        batch_size = t.shape[0]
        losses = [
            (torch.sum(l_confidence_bg) + torch.sum(l_confidence)) / batch_size,
            torch.sum(l_reg) / batch_size,
            torch.sum(l_scale) / batch_size,
        ]

        if not all(torch.isfinite(l).item() if l is not None else True for l in losses):
            raise Exception('found a loss that is not finite: {}, prev: {}'
                            ''.format(losses, self.previous_losses))
        self.previous_losses = [float(l.item()) if l is not None else None for l in losses]

        return losses
