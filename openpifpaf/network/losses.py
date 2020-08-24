"""Losses."""

import logging
import torch

from . import heads

LOG = logging.getLogger(__name__)


class Bce(torch.nn.Module):
    def __init__(self, *, focal_gamma=0.0, detach_focal=False):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.detach_focal = detach_focal

    def forward(self, x, t):  # pylint: disable=arguments-differ
        t_zeroone = t.clone()
        t_zeroone[t_zeroone > 0.0] = 1.0
        # x = torch.clamp(x, -20.0, 20.0)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            x, t_zeroone, reduction='none')
        bce = torch.clamp(bce, 0.02, 5.0)  # 0.02 -> -3.9, 0.01 -> -4.6, 0.001 -> -7, 0.0001 -> -9

        if self.focal_gamma != 0.0:
            pt = torch.exp(-bce)
            focal = (1.0 - pt)**self.focal_gamma
            if self.detach_focal:
                focal = focal.detach()
            bce = focal * bce

        weight_mask = t_zeroone != t
        bce[weight_mask] = bce[weight_mask] * t[weight_mask]

        return bce


class ScaleLoss(torch.nn.Module):
    def __init__(self, b, *, low_clip=0.0, relative=False):
        super().__init__()
        self.b = b
        self.low_clip = low_clip
        self.relative = relative

    def forward(self, logs, t):  # pylint: disable=arguments-differ
        loss = torch.nn.functional.l1_loss(
            torch.exp(logs),
            t,
            reduction='none',
        )
        loss = torch.clamp(loss, self.low_clip, 5.0)

        loss = loss / self.b
        if self.relative:
            loss = loss / (1.0 + t)

        return loss


def laplace_loss(x1, x2, logb, t1, t2, *, weight=None, norm_low_clip=0.0):
    """Loss based on Laplace Distribution.

    Loss for a single two-dimensional vector (x1, x2) with radial
    spread b and true (t1, t2) vector.
    """

    # left derivative of sqrt at zero is not defined, so prefer torch.norm():
    # https://github.com/pytorch/pytorch/issues/2421
    # norm = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
    norm = (torch.stack((x1, x2)) - torch.stack((t1, t2))).norm(dim=0)
    norm = torch.clamp(norm, norm_low_clip, 5.0)

    # constrain range of logb
    # low range constraint: prevent strong confidence when overfitting
    # high range constraint: force some data dependence
    # logb = 3.0 * torch.tanh(logb / 3.0)
    logb = torch.clamp_min(logb, -3.0)

    # ln(2) = 0.694
    losses = logb + (norm + 0.1) * torch.exp(-logb)
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


def margin_loss(x1, x2, t1, t2, max_r1, max_r2, max_r3, max_r4):
    x = torch.stack((x1, x2))
    t = torch.stack((t1, t2))

    max_r = torch.min((torch.stack(max_r1, max_r2, max_r3, max_r4)), axis=0)
    m0 = torch.isfinite(max_r)
    x = x[:, m0]
    t = t[:, m0]
    max_r = max_r[m0]

    # m1 = (x - t).norm(p=1, dim=0) > max_r
    # x = x[:, m1]
    # t = t[:, m1]
    # max_r = max_r[m1]

    norm = (x - t).norm(dim=0)
    m2 = norm > max_r

    return torch.sum(norm[m2] - max_r[m2])


def quadrant(xys):
    q = torch.zeros((xys.shape[1],), dtype=torch.long)
    q[xys[0, :] < 0.0] += 1
    q[xys[1, :] < 0.0] += 2
    return q


def quadrant_margin_loss(x1, x2, t1, t2, max_r1, max_r2, max_r3, max_r4):
    x = torch.stack((x1, x2))
    t = torch.stack((t1, t2))

    diffs = x - t
    qs = quadrant(diffs)
    norms = diffs.norm(dim=0)

    m1 = norms[qs == 0] > max_r1[qs == 0]
    m2 = norms[qs == 1] > max_r2[qs == 1]
    m3 = norms[qs == 2] > max_r3[qs == 2]
    m4 = norms[qs == 3] > max_r4[qs == 3]

    return (
        torch.sum(norms[qs == 0][m1] - max_r1[qs == 0][m1]) +
        torch.sum(norms[qs == 1][m2] - max_r2[qs == 1][m2]) +
        torch.sum(norms[qs == 2][m3] - max_r3[qs == 2][m3]) +
        torch.sum(norms[qs == 3][m4] - max_r4[qs == 3][m4])
    )


class SmoothL1Loss():
    r_smooth = 0.0

    def __init__(self, *, scale_required=True):
        self.scale = None
        self.scale_required = scale_required

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


class MultiHeadLoss(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas):
        super().__init__()

        if not lambdas:
            lambdas = [1.0 for l in losses for _ in l.field_names]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas

        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss: %s, %s', self.field_names, self.lambdas)

    def forward(self, head_fields, head_targets):  # pylint: disable=arguments-differ
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        assert self.task_sparsity_weight == 0.0  # TODO implement
        flat_head_losses = [ll
                            for l, f, t in zip(self.losses, head_fields, head_targets)
                            for ll in l(f, t)]

        assert len(self.lambdas) == len(flat_head_losses)
        loss_values = [lam * l
                       for lam, l in zip(self.lambdas, flat_head_losses)
                       if l is not None]
        total_loss = sum(loss_values) if loss_values else None

        return total_loss, flat_head_losses


class MultiHeadLossAutoTuneKendall(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas, *, sparse_task_parameters=None, tune=None):
        """Auto-tuning multi-head loss.

        Uses idea from "Multi-Task Learning Using Uncertainty to Weigh Losses
        for Scene Geometry and Semantics" by Kendall, Gal and Cipolla.

        Individual losses must not be negative for Kendall's prescription.

        In the common setting, use lambdas of zero and one to deactivate and
        activate the tasks you want to train. Less common, if you have
        secondary tasks, you can reduce their importance by choosing a
        lambda value between zero and one.
        """
        super().__init__()

        if not lambdas:
            lambdas = [1.0 for l in losses for _ in l.field_names]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas
        self.sparse_task_parameters = sparse_task_parameters
        self.tune = tune

        self.log_sigmas = torch.nn.Parameter(
            torch.zeros((len(lambdas),), dtype=torch.float64),
            requires_grad=True,
        )

        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss with autotune: %s', self.field_names)
        assert len(self.field_names) == len(self.lambdas)
        assert len(self.field_names) == len(self.log_sigmas)

        if self.tune is None:
            def tune_from_name(name):
                if '.vec' in name:
                    return 'none'
                if '.scale' in name:
                    return 'laplace'
                return 'gauss'
            self.tune = [
                tune_from_name(n)
                for l in self.losses for n in l.field_names
            ]
        LOG.info('tune config: %s', self.tune)

    def batch_meta(self):
        return {'mtl_sigmas': [round(float(s), 3) for s in self.log_sigmas.exp()]}

    def forward(self, *args):
        head_fields, head_targets = args
        LOG.debug('losses = %d, fields = %d, targets = %d',
                  len(self.losses), len(head_fields), len(head_targets))
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        flat_head_losses = [ll
                            for l, f, t in zip(self.losses, head_fields, head_targets)
                            for ll in l(f, t)]

        assert len(self.lambdas) == len(flat_head_losses)
        assert len(self.log_sigmas) == len(flat_head_losses)
        constrained_log_sigmas = 3.0 * torch.tanh(self.log_sigmas / 3.0)
        def tuned_loss(tune, log_sigma, loss):
            if tune == 'none':
                return loss
            if tune == 'laplace':
                # this is the negative ln of a Laplace with
                # ln(2) = 0.694
                return 0.694 + log_sigma + loss * torch.exp(-log_sigma)
            if tune == 'gauss':
                # this is the negative ln of a Gaussian with
                # ln(sqrt(2pi)) = 0.919
                return 0.919 + log_sigma + loss * 0.5 * torch.exp(-2.0 * log_sigma)
            raise Exception('unknown tune: {}'.format(tune))
        loss_values = [
            lam * tuned_loss(t, log_sigma, l)
            for lam, t, log_sigma, l in zip(
                self.lambdas, self.tune, constrained_log_sigmas, flat_head_losses)
            if l is not None
        ]
        total_loss = sum(loss_values) if loss_values else None

        if self.task_sparsity_weight and self.sparse_task_parameters is not None:
            head_sparsity_loss = sum(
                # torch.norm(param, p=1)
                param.abs().max(dim=1)[0].clamp(min=1e-6).sum()
                for param in self.sparse_task_parameters
            )
            LOG.debug('l1 head sparsity loss = %f (total = %f)', head_sparsity_loss, total_loss)
            total_loss = total_loss + self.task_sparsity_weight * head_sparsity_loss

        return total_loss, flat_head_losses


class MultiHeadLossAutoTuneVariance(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas, *, sparse_task_parameters=None):
        """Auto-tuning multi-head loss based on loss-variance.

        In the common setting, use lambdas of zero and one to deactivate and
        activate the tasks you want to train. Less common, if you have
        secondary tasks, you can reduce their importance by choosing a
        lambda value between zero and one.
        """
        super().__init__()

        if not lambdas:
            lambdas = [1.0 for l in losses for _ in l.field_names]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas
        self.sparse_task_parameters = sparse_task_parameters

        self.epsilons = torch.ones((len(lambdas),), dtype=torch.float64)
        # choose a prime number for the buffer length:
        # for multiple tasks, prevents that always the same buffer_index is
        # skipped which would mean that some nan values will remain forever
        self.buffer = torch.full((len(lambdas), 53), float('nan'), dtype=torch.float64)
        self.buffer_index = -1

        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss with autotune: %s', self.field_names)
        assert len(self.field_names) == len(self.lambdas)
        assert len(self.field_names) == len(self.epsilons)

    def batch_meta(self):
        return {'mtl_sigmas': [round(float(s), 3) for s in self.epsilons]}

    def forward(self, *args):
        head_fields, head_targets = args
        LOG.debug('losses = %d, fields = %d, targets = %d',
                  len(self.losses), len(head_fields), len(head_targets))
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        flat_head_losses = [ll
                            for l, f, t in zip(self.losses, head_fields, head_targets)
                            for ll in l(f, t)]

        self.buffer_index = (self.buffer_index + 1) % self.buffer.shape[1]
        for i, ll in enumerate(flat_head_losses):
            if not hasattr(ll, 'data'):  # e.g. when ll is None
                continue
            self.buffer[i, self.buffer_index] = ll.data
        self.epsilons = torch.sqrt(
            torch.mean(self.buffer**2, dim=1)
            - torch.sum(self.buffer, dim=1)**2 / self.buffer.shape[1]**2
        )
        self.epsilons[torch.isnan(self.epsilons)] = 10.0
        self.epsilons = self.epsilons.clamp(0.01, 100.0)

        # normalize sum 1/eps to the number of sub-losses
        LOG.debug('eps before norm: %s', self.epsilons)
        self.epsilons = self.epsilons * torch.sum(1.0 / self.epsilons) / self.epsilons.shape[0]
        LOG.debug('eps after norm: %s', self.epsilons)

        assert len(self.lambdas) == len(flat_head_losses)
        assert len(self.epsilons) == len(flat_head_losses)
        loss_values = [
            lam * l / eps
            for lam, eps, l in zip(self.lambdas, self.epsilons, flat_head_losses)
            if l is not None
        ]
        total_loss = sum(loss_values) if loss_values else None

        if self.task_sparsity_weight and self.sparse_task_parameters is not None:
            head_sparsity_loss = sum(
                # torch.norm(param, p=1)
                param.abs().max(dim=1)[0].clamp(min=1e-6).sum()
                for param in self.sparse_task_parameters
            )
            LOG.debug('l1 head sparsity loss = %f (total = %f)', head_sparsity_loss, total_loss)
            total_loss = total_loss + self.task_sparsity_weight * head_sparsity_loss

        return total_loss, flat_head_losses


class CompositeLoss(torch.nn.Module):
    background_weight = 1.0
    focal_gamma = 1.0
    b_scale = 1.0
    margin = False

    def __init__(self, head_net: heads.CompositeField, regression_loss):
        super().__init__()
        self.n_vectors = head_net.meta.n_vectors
        self.n_scales = head_net.meta.n_scales

        LOG.debug('%s: n_vectors = %d, n_scales = %d, margin = %s',
                  head_net.meta.name, self.n_vectors, self.n_scales, self.margin)

        self.confidence_loss = Bce(focal_gamma=self.focal_gamma, detach_focal=True)
        self.regression_loss = regression_loss or laplace_loss
        self.scale_losses = torch.nn.ModuleList([ScaleLoss(self.b_scale, low_clip=0.0)
                                                 for _ in range(self.n_scales)])
        self.field_names = (
            ['{}.c'.format(head_net.meta.name)] +
            ['{}.vec{}'.format(head_net.meta.name, i + 1) for i in range(self.n_vectors)] +
            ['{}.scales{}'.format(head_net.meta.name, i + 1) for i in range(self.n_scales)]
        )
        if self.margin:
            self.field_names += ['{}.margin{}'.format(head_net.meta.name, i + 1)
                                 for i in range(self.n_vectors)]

        self.bce_blackout = None
        self.previous_losses = None

    def _confidence_loss(self, x_confidence, target_confidence):
        bce_masks = torch.isnan(target_confidence).bitwise_not_()
        if not torch.any(bce_masks):
            return None

        # TODO assumes one confidence
        x_confidence = x_confidence[:, :, 0]

        batch_size = x_confidence.shape[0]
        LOG.debug('batch size = %d', batch_size)

        if self.bce_blackout:
            x_confidence = x_confidence[:, self.bce_blackout]
            bce_masks = bce_masks[:, self.bce_blackout]
            target_confidence = target_confidence[:, self.bce_blackout]

        LOG.debug('BCE: x = %s, target = %s, mask = %s',
                  x_confidence.shape, target_confidence.shape, bce_masks.shape)
        bce_target = torch.masked_select(target_confidence, bce_masks)
        x_confidence = torch.masked_select(x_confidence, bce_masks)
        ce_loss = self.confidence_loss(x_confidence, bce_target)
        if self.background_weight != 1.0:
            bce_weight = torch.ones_like(bce_target, requires_grad=False)
            bce_weight[bce_target == 0] *= self.background_weight
            ce_loss = ce_loss * bce_weight

        ce_loss = ce_loss.sum() / batch_size

        return ce_loss

    def _localization_loss(self, x_regs, x_logbs, target_regs):
        batch_size = target_regs[0].shape[0]

        reg_losses = []
        for i, target_reg in enumerate(target_regs):
            reg_masks = torch.isnan(target_reg[:, :, 0]).bitwise_not_()
            if not torch.any(reg_masks):
                reg_losses.append(None)
                continue

            reg_losses.append(self.regression_loss(
                torch.masked_select(x_regs[:, :, i, 0], reg_masks),
                torch.masked_select(x_regs[:, :, i, 1], reg_masks),
                torch.masked_select(x_logbs[:, :, i], reg_masks),
                torch.masked_select(target_reg[:, :, 0], reg_masks),
                torch.masked_select(target_reg[:, :, 1], reg_masks),
                norm_low_clip=0.0,
            ).sum() / batch_size)

        return reg_losses

    def _scale_losses(self, x_scales, target_scales):
        assert x_scales.shape[2] == len(target_scales)

        batch_size = x_scales.shape[0]
        return [
            sl(
                torch.masked_select(x_scales[:, :, i], torch.isnan(target_scale).bitwise_not_()),
                torch.masked_select(target_scale, torch.isnan(target_scale).bitwise_not_()),
            ).sum() / batch_size
            for i, (sl, target_scale) in enumerate(zip(self.scale_losses, target_scales))
        ]

    def _margin_losses(self, x_regs, target_regs, *, target_confidence):
        if not self.margin:
            return []

        reg_masks = target_confidence > 0.5
        if not torch.any(reg_masks):
            return [None for _ in target_regs]

        batch_size = reg_masks.shape[0]
        margin_losses = []
        for x_reg, target_reg in zip(x_regs, target_regs):
            margin_losses.append(quadrant_margin_loss(
                torch.masked_select(x_reg[:, :, 0], reg_masks),
                torch.masked_select(x_reg[:, :, 1], reg_masks),
                torch.masked_select(target_reg[:, :, 0], reg_masks),
                torch.masked_select(target_reg[:, :, 1], reg_masks),
                torch.masked_select(target_reg[:, :, 2], reg_masks),
                torch.masked_select(target_reg[:, :, 3], reg_masks),
                torch.masked_select(target_reg[:, :, 4], reg_masks),
                torch.masked_select(target_reg[:, :, 5], reg_masks),
            ) / (100.0 * batch_size))
        return margin_losses

    def forward(self, *args):
        LOG.debug('loss for %s', self.field_names)

        x, t = args

        x = [xx.double() for xx in x]
        t = [tt.double() for tt in t]

        x_confidence, x_regs, x_logbs, x_scales = x

        assert len(t) == 1 + self.n_vectors + self.n_scales
        running_t = iter(t)
        target_confidence = next(running_t)
        target_regs = [next(running_t) for _ in range(self.n_vectors)]
        target_scales = [next(running_t) for _ in range(self.n_scales)]

        ce_loss = self._confidence_loss(x_confidence, target_confidence)
        reg_losses = self._localization_loss(x_regs, x_logbs, target_regs)
        scale_losses = self._scale_losses(x_scales, target_scales)
        margin_losses = self._margin_losses(x_regs, target_regs,
                                            target_confidence=target_confidence)

        all_losses = [ce_loss] + reg_losses + scale_losses + margin_losses
        if not all(torch.isfinite(l).item() if l is not None else True for l in all_losses):
            raise Exception('found a loss that is not finite: {}, prev: {}'
                            ''.format(all_losses, self.previous_losses))
        self.previous_losses = [float(l.item()) if l is not None else None for l in all_losses]

        return all_losses


def cli(parser):
    group = parser.add_argument_group('losses')
    group.add_argument('--lambdas', default=None, type=float, nargs='+',
                       help='prefactor for head losses')
    group.add_argument('--r-smooth', type=float, default=SmoothL1Loss.r_smooth,
                       help='r_{smooth} for SmoothL1 regressions')
    group.add_argument('--regression-loss', default='laplace',
                       choices=['smoothl1', 'smootherl1', 'l1', 'laplace'],
                       help='type of regression loss')
    group.add_argument('--background-weight', default=CompositeLoss.background_weight, type=float,
                       help='BCE weight where ground truth is background')
    group.add_argument('--b-scale', default=CompositeLoss.b_scale, type=float,
                       help='Laplace width b for scale loss')
    group.add_argument('--focal-gamma', default=CompositeLoss.focal_gamma, type=float,
                       help='when > 0.0, use focal loss with the given gamma')
    group.add_argument('--margin-loss', default=False, action='store_true',
                       help='[experimental]')
    group.add_argument('--auto-tune-mtl', default=False, action='store_true',
                       help=('[experimental] use Kendall\'s prescription for '
                             'adjusting the multitask weight'))
    group.add_argument('--auto-tune-mtl-variance', default=False, action='store_true',
                       help=('[experimental] use Variance prescription for '
                             'adjusting the multitask weight'))
    assert MultiHeadLoss.task_sparsity_weight == MultiHeadLossAutoTuneKendall.task_sparsity_weight
    assert MultiHeadLoss.task_sparsity_weight == MultiHeadLossAutoTuneVariance.task_sparsity_weight
    group.add_argument('--task-sparsity-weight',
                       default=MultiHeadLoss.task_sparsity_weight, type=float,
                       help='[experimental]')


def configure(args):
    # apply for CompositeLoss
    CompositeLoss.background_weight = args.background_weight
    CompositeLoss.focal_gamma = args.focal_gamma
    CompositeLoss.b_scale = args.b_scale
    CompositeLoss.margin = args.margin_loss

    # MultiHeadLoss
    MultiHeadLoss.task_sparsity_weight = args.task_sparsity_weight
    MultiHeadLossAutoTuneKendall.task_sparsity_weight = args.task_sparsity_weight
    MultiHeadLossAutoTuneVariance.task_sparsity_weight = args.task_sparsity_weight

    # SmoothL1
    SmoothL1Loss.r_smooth = args.r_smooth


def factory_from_args(args, head_nets):
    return factory(
        head_nets,
        args.lambdas,
        reg_loss_name=args.regression_loss,
        device=args.device,
        auto_tune_mtl_kendall=args.auto_tune_mtl,
        auto_tune_mtl_variance=args.auto_tune_mtl_variance,
    )


# pylint: disable=too-many-branches
def factory(head_nets, lambdas, *,
            reg_loss_name=None, device=None,
            auto_tune_mtl_kendall=False, auto_tune_mtl_variance=False):
    if isinstance(head_nets[0], (list, tuple)):
        return [factory(hn, lam,
                        reg_loss_name=reg_loss_name,
                        device=device)
                for hn, lam in zip(head_nets, lambdas)]

    if reg_loss_name == 'smoothl1':
        reg_loss = SmoothL1Loss()
    elif reg_loss_name == 'l1':
        reg_loss = l1_loss
    elif reg_loss_name == 'laplace':
        reg_loss = laplace_loss
    elif reg_loss_name is None:
        reg_loss = laplace_loss
    else:
        raise Exception('unknown regression loss type {}'.format(reg_loss_name))

    sparse_task_parameters = None
    if MultiHeadLoss.task_sparsity_weight:
        sparse_task_parameters = []
        for head_net in head_nets:
            if getattr(head_net, 'sparse_task_parameters', None) is not None:
                sparse_task_parameters += head_net.sparse_task_parameters
            elif isinstance(head_net, heads.CompositeFieldFused):
                sparse_task_parameters.append(head_net.conv.weight)
            else:
                raise Exception('unknown l1 parameters for given head: {} ({})'
                                ''.format(head_net.meta.name, type(head_net)))

    losses = [CompositeLoss(head_net, reg_loss) for head_net in head_nets]
    if auto_tune_mtl_kendall:
        loss = MultiHeadLossAutoTuneKendall(losses, lambdas,
                                            sparse_task_parameters=sparse_task_parameters)
    elif auto_tune_mtl_variance:
        loss = MultiHeadLossAutoTuneVariance(losses, lambdas,
                                             sparse_task_parameters=sparse_task_parameters)
    else:
        loss = MultiHeadLoss(losses, lambdas)

    if device is not None:
        loss = loss.to(device)

    return loss
