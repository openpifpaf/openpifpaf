"""Losses."""

import logging
import torch

from . import heads


from torch.autograd import Variable
import torch.nn.functional as F

import os.path
from os import path

from .panoptic_losses import RegularCE, OhemCE, DeepLabCE

from torch import nn
L1Loss = nn.L1Loss
MSELoss = nn.MSELoss
CrossEntropyLoss = nn.CrossEntropyLoss

LOG = logging.getLogger(__name__)


def laplace_loss(x1, x2, logb, t1, t2, weight=None):
    """Loss based on Laplace Distribution.

    Loss for a single two-dimensional vector (x1, x2) with radial
    spread b and true (t1, t2) vector.
    """

    # left derivative of sqrt at zero is not defined, so prefer torch.norm():
    # https://github.com/pytorch/pytorch/issues/2421
    # norm = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)


    # print('shape logb', logb.shape)

    
    norm = (torch.stack((x1, x2)) - torch.stack((t1, t2))).norm(dim=0)

    # constrain range of logb
    logb = 3.0 * torch.tanh(logb / 3.0)

    losses = 0.694 + logb + norm * torch.exp(-logb)
    if weight is not None:
        losses = losses * weight
    return torch.sum(losses)            # TODO why sum? shouldn't it be mean? because each image can have different number of nans !
    # return torch.mean(losses)


def l1_loss(x1, x2, _, t1, t2, weight=None):
    """L1 loss.

    Loss for a single two-dimensional vector (x1, x2)
    true (t1, t2) vector.
    """
    losses = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
    # losses = (x1 - t1)**2 + (x2 - t2)**2
    if weight is not None:
        losses = losses * weight
    return torch.sum(losses)
    # return torch.mean(losses)


def logl1_loss(logx, t, **kwargs):
    """Swap in replacement for functional.l1_loss."""
    return torch.nn.functional.l1_loss(
        logx, torch.log(t), **kwargs)


def margin_loss(x1, x2, t1, t2, max_r1, max_r2, max_r3, max_r4):
    x = torch.stack((x1, x2))
    t = torch.stack((t1, t2))

    # print('shape x', x.shape)
    # print('shape t', t.shape)
    # print('shape max_r1', max_r1.shape)

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


class SmoothL1Loss(object):
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
        # d = (x1 - t1)**2 + (x2 - t2)**2
        smooth_regime = d < r

        smooth_loss = 0.5 / r[smooth_regime] * d[smooth_regime] ** 2
        linear_loss = d[smooth_regime == 0] - (0.5 * r[smooth_regime == 0])
        losses = torch.cat((smooth_loss, linear_loss))

        if weight is not None:
            losses = losses * weight

        self.scale = None
        return torch.sum(losses)
        # return torch.mean(losses)

class MultiHeadLoss(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas, loss_debug=0):
        super(MultiHeadLoss, self).__init__()

        if not lambdas:
            lambdas = [1.0 for l in losses for _ in l.field_names]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas

        self.field_names = [n for l in self.losses for n in l.field_names]

        ### for freezing one head
        self.loss_debug = [1.0 for l in losses for _ in l.field_names]
        if loss_debug == 1:
            self.loss_debug[3:5] = [0., 0.] 
        elif loss_debug == 2:
            self.loss_debug[0:3] = [0., 0., 0.]
        
        print('multihead loss: %s, %s', self.field_names, self.lambdas)
        LOG.info('multihead loss: %s, %s', self.field_names, self.lambdas)

    def forward(self, head_fields, head_targets):  # pylint: disable=arguments-differ
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        assert self.task_sparsity_weight == 0.0  # TODO implement
        flat_head_losses = [ll
                            for l, f, t in zip(self.losses, head_fields, head_targets)
                            for ll in l(f, t)]

        ### for freezing one head
        loss_values_ = [ld * l
                    for ld, l in zip(self.loss_debug, flat_head_losses)
                    if l is not None]
        
        assert len(self.lambdas) == len(flat_head_losses)
        loss_values = [lam * l
                       for lam, l in zip(self.lambdas, loss_values_)
                       if l is not None]

        total_loss = sum(loss_values) if loss_values else None
        # total_loss = sum(flat_head_losses) if flat_head_losses else None

        return total_loss, flat_head_losses


class MultiHeadLossAutoTune(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas, *, sparse_task_parameters=None, loss_debug=0):
        """Auto-tuning multi-head less.

        Uses idea from "Multi-Task Learning Using Uncertainty to Weigh Losses
        for Scene Geometry and Semantics" by Kendall, Gal and Cipolla.

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

        self.log_sigmas = torch.nn.Parameter(
            torch.zeros((len(lambdas),), dtype=torch.float64),
            requires_grad=True,
        )

        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss with autotune: %s', self.field_names)
        assert len(self.field_names) == len(self.lambdas)
        assert len(self.field_names) == len(self.log_sigmas)


        ### for freezing one head
        self.loss_debug = [1.0 for l in losses for _ in l.field_names]
        if loss_debug == 1:
            self.loss_debug[3:5] = [0., 0.] 
        elif loss_debug == 2:
            self.loss_debug[0:3] = [0., 0., 0.]


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

        ### for freezing one head
        loss_values_ = [ld * l
                    for ld, l in zip(self.loss_debug, flat_head_losses)
                    if l is not None]

        loss_values = [lam * l / (2.0 * (log_sigma.exp() ** 2))
                       for lam, log_sigma, l in zip(self.lambdas, self.log_sigmas, loss_values_)
                       if l is not None]
        auto_reg = [lam * log_sigma
                    for lam, log_sigma, l in zip(self.lambdas, self.log_sigmas, loss_values_)
                    if l is not None]
        
        
        
        total_loss = sum(loss_values) + sum(auto_reg) if loss_values else None

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
    margin = False

    def __init__(self, head_net: heads.CompositeField, regression_loss):
        super(CompositeLoss, self).__init__()
        self.n_vectors = head_net.meta.n_vectors
        self.n_scales = head_net.meta.n_scales

        LOG.debug('%s: n_vectors = %d, n_scales = %d, margin = %s',
                  head_net.meta.name, self.n_vectors, self.n_scales, self.margin)

        self.regression_loss = regression_loss or laplace_loss
        self.field_names = (
            ['{}.c'.format(head_net.meta.name)] +
            ['{}.vec{}'.format(head_net.meta.name, i + 1) for i in range(self.n_vectors)] +
            ['{}.scales{}'.format(head_net.meta.name, i + 1) for i in range(self.n_scales)]
        )
        if self.margin:
            self.field_names += ['{}.margin{}'.format(head_net.meta.name, i + 1)
                                 for i in range(self.n_vectors)]

        self.bce_blackout = None

    def _confidence_loss(self, x_confidence, target_confidence):
        bce_masks = torch.isnan(target_confidence).bitwise_not_()
        if not torch.any(bce_masks):
            return None

        # TODO assumes one confidence
        # print('x_confidence')
        # print(x_confidence.shape)
        x_confidence = x_confidence[:, :, 0]

        batch_size = x_confidence.shape[0]
        LOG.debug('batch size = %d', batch_size)

        if self.bce_blackout:
            x_confidence = x_confidence[:, self.bce_blackout]
            bce_masks = bce_masks[:, self.bce_blackout]
            target_confidence = target_confidence[:, self.bce_blackout]

        LOG.debug('BCE: x = %s, target = %s, mask = %s',
                  x_confidence.shape, target_confidence.shape, bce_masks.shape)
        # print(bce_masks.sum())
        # print(target_confidence.sum())
        bce_target = torch.masked_select(target_confidence, bce_masks)
        # print(bce_target.sum())
        # raise
        bce_weight = 1.0
        x_confidence = torch.masked_select(x_confidence, bce_masks)
        if self.background_weight != 1.0:
            bce_weight = torch.ones_like(bce_target, requires_grad=False)
            bce_weight[bce_target == 0] *= self.background_weight
        elif self.focal_gamma != 0.0:
            bce_weight = torch.empty_like(bce_target, requires_grad=False)
            bce_weight[bce_target == 1] = x_confidence[bce_target == 1]
            bce_weight[bce_target == 0] = -x_confidence[bce_target == 0]
            bce_weight = (1.0 + torch.exp(bce_weight)).pow(-self.focal_gamma)
        # print(x_confidence.shape)
        ce_loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            x_confidence,
            bce_target,
            # weight=bce_weight,
            reduction='none',
        ) * bce_weight).sum() / (1000.0 * batch_size) #.mean()

        return ce_loss

    def _localization_loss(self, x_regs, x_logbs, target_regs):
        batch_size = target_regs[0].shape[0]
        # print('localization loss')
        # print(x_regs.shape)
        # print(x_logbs.shape)
        # print(len(target_regs))
        # print(target_regs[0].shape)

        reg_losses = []

        for i, target_reg in enumerate(target_regs):
            # print(target_reg.shape)
            reg_masks = torch.isnan(target_reg[:, :, 0]).bitwise_not_()
            if not torch.any(reg_masks):
                reg_losses.append(None)
                continue
            
            # print('local loss')
            # print(x_regs.shape)
            # print(x_logbs.shape)
            # print(target_reg.shape)

            reg_losses.append(self.regression_loss(
                torch.masked_select(x_regs[:, :, i, 0], reg_masks),
                torch.masked_select(x_regs[:, :, i, 1], reg_masks),
                torch.masked_select(x_logbs[:, :, i], reg_masks),
                torch.masked_select(target_reg[:, :, 0], reg_masks),
                torch.masked_select(target_reg[:, :, 1], reg_masks),
                weight=0.1,
            )/(100.0 * batch_size))

        # print(len(reg_losses))
        # raise

        return reg_losses

    @staticmethod
    def _scale_losses(x_scales, target_scales):
        # batch_size = target_scales[0].shape[0]
        batch_size = x_scales.shape[0]
        assert x_scales.shape[2] == len(target_scales)
        # print('scale_losess')
        # print(x_scales.shape)
        # print(len(target_scales))
        # print(target_scales[0].shape)
        scale_losses = []
        for i, target_scale in enumerate(target_scales):
            # print(target_reg.shape)
            reg_masks = torch.isnan(target_scale).bitwise_not_()
            if not torch.any(reg_masks):
                scale_losses.append(None)
                continue

            scale_losses.append(
                logl1_loss(
                torch.masked_select(x_scales[:, :, i], torch.isnan(target_scale).bitwise_not_()),
                torch.masked_select(target_scale, torch.isnan(target_scale).bitwise_not_()),
                reduction='sum', #mean            # TODO shouldn't it be mean???
            ) / (100.0 * batch_size))

        return scale_losses

        # batch_size = x_scales.shape[0]
        # return [
        #     logl1_loss(
        #         torch.masked_select(x_scales[:, :, i], torch.isnan(target_scale).bitwise_not_()),
        #         torch.masked_select(target_scale, torch.isnan(target_scale).bitwise_not_()),
        #         reduction='mean',            # TODO shouldn't it be mean???
        #     ) #/ (100.0 * batch_size)
        #     for i, target_scale in enumerate(target_scales)
        # ]

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
        # print('hell yes')
        # print(t)
        # print(len(t))
        # t, mask = t

        x = [xx.double() for xx in x]
        t = [tt.double() for tt in t]
        # print('pif instances')
        # print(len(t))
        # print(t[0].shape)
        # print(mask.shape)
        # print(t.shape)

        x_confidence, x_regs, x_logbs, x_scales = x
        # print(x_confidence.shape)
        # print(x_regs.shape)
        # print(x_logbs.shape)
        # print(x_scales.shape)


        assert len(t) == 1 + self.n_vectors + self.n_scales
        running_t = iter(t)
        target_confidence = next(running_t)
        target_regs = [next(running_t) for _ in range(self.n_vectors)]
        target_scales = [next(running_t) for _ in range(self.n_scales)]

        # import os
        # import pickle
        # if not os.path.isfile('targets.pickle'):
        #     with open('targets.pickle','wb') as f:
        #         pickle.dump((target_confidence, target_regs, target_scales),f)

        # print('targets')
        # print(target_confidence.shape)
        # print(target_regs[0].shape)
        # print(target_scales[0].shape)
        

        ce_loss = self._confidence_loss(x_confidence, target_confidence)
        reg_losses = self._localization_loss(x_regs, x_logbs, target_regs)
        scale_losses = self._scale_losses(x_scales, target_scales)
        margin_losses = self._margin_losses(x_regs, target_regs,
                                            target_confidence=target_confidence)
        # print('++++++++++++++++++++++++PIF OUTPUT')
        # print([ce_loss] + reg_losses + scale_losses + margin_losses)
        # print(ce_loss)
        # print(reg_losses)
        # print(scale_losses)
        # print(margin_losses)
        return [ce_loss] + reg_losses + scale_losses + margin_losses



class PanopticLoss(torch.nn.Module):
    def __init__(self, args):
        super(PanopticLoss, self).__init__()
        self.field_names = ['pan.semantic', 'pan.offset']
        self.semantic_loss = build_loss_from_cfg(args, 'semantic')
        # self.semantic_loss_weight = self.field_names[0]
        self.offset_loss = build_loss_from_cfg(args, 'offset')
        # self.offset_loss_weight = self.field_names[0]

    def forward(self, results, targets):
        
        # batch_size = results['semantic'].size(0)
        # loss = 0
        if 'semantic_weights' in targets.keys():
            semantic_loss = self.semantic_loss(
                results['semantic'], targets['semantic'], semantic_weights=targets['semantic_weights']
            )
        else:
            semantic_loss = self.semantic_loss(
                results['semantic'], targets['semantic'])

        # self.loss_meter_dict['Semantic loss'].update(semantic_loss.detach().cpu().item(), batch_size)
        # loss += semantic_loss
        if self.offset_loss is not None:
            # Pixel-wise loss weight
            offset_loss_weights = targets['offset_weights'][:, None, :, :].expand_as(results['offset'])
            # print(targets['offset'].shape)
            # print(results['offset'].shape)
            offset_loss = self.offset_loss(results['offset'], targets['offset']) * offset_loss_weights
            # print(offset_loss.shape)
            
            # safe division
            # TODO figure out the purpose of these lines (maybe add /2 to take the actual avg.)
            if offset_loss_weights.sum() > 0:
                offset_loss = offset_loss.sum() / offset_loss_weights.sum()     # offset_loss_weights (number of pixels of containing things)
            else:
                offset_loss = offset_loss.sum() * 0
            # self.loss_meter_dict['Offset loss'].update(offset_loss.detach().cpu().item(), batch_size)
            # loss += offset_loss
            # print(offset_loss)
            # raise

        # print('in losses (semantic loss)', semantic_loss)
        
        return [semantic_loss] + [offset_loss * 0.01]

class PanopticLossDummy(torch.nn.Module):
    def __init__(self, args):
        super(PanopticLossDummy, self).__init__()
        self.field_names = ['pan.semantic', 'pan.offset']
        self.semantic_loss = build_loss_from_cfg(args, 'semantic')
        # self.semantic_loss_weight = self.field_names[0]
        self.offset_loss = build_loss_from_cfg(args, 'offset')
        # self.offset_loss_weight = self.field_names[0]
        self.printed = False

    def forward(self, results, targets):
        if not self.printed:
            # results offset: torch.Size([4, 2, 385, 385])
            # targets offset: torch.Size([4, 2, 385, 385])
            # targets offset: torch.Size([4, 385, 385])
            # results semantic: torch.Size([4, 2, 385, 385])
            # targets semantic: torch.Size([4, 385, 385])
            # targets semantic: torch.Size([4, 385, 385])
            print('results offset:',results['offset'].shape)
            print('targets offset:',targets['offset'].shape)
            print('targets offset:',targets['offset_weights'].shape)
            print('results semantic:',results['semantic'].shape)
            print('targets semantic:',targets['semantic'].shape)
            print('targets semantic:',targets['semantic_weights'].shape)
            self.printed = True
        return [results['semantic'].sum()*0., results['offset'].sum()*0.]

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
    group.add_argument('--focal-gamma', default=CompositeLoss.focal_gamma, type=float,
                       help='when > 0.0, use focal loss with the given gamma')
    group.add_argument('--margin-loss', default=False, action='store_true',
                       help='[experimental]')
    group.add_argument('--auto-tune-mtl', default=False, action='store_true',
                       help='use Kendall\'s prescription for adjusting the multitask weight')
    assert MultiHeadLoss.task_sparsity_weight == MultiHeadLossAutoTune.task_sparsity_weight
    group.add_argument('--task-sparsity-weight',
                       default=MultiHeadLoss.task_sparsity_weight, type=float,
                       help='[experimental]')

    group.add_argument('--seman-loss-name', default='hard_pixel_mining',
                       choices=['cross_entropy', 'ohem', 'hard_pixel_mining'],
                       help='type of panoptic loss')
    group.add_argument('--seman-loss-ignore', default=-1, type=int,
                       help='label to ignore')
    group.add_argument('--seman-loss-threshold', default=0.7, type=float,
                        help='threshold for softmax score (of gt class), only predictions with softmax score below this threshold will be kept.')
    group.add_argument('--seman-loss-min-kept', default=100000, type=int,
                        help='minimum number of pixels to be kept, it is used to adjust the threshold value to avoid number of examples being too small.')
    group.add_argument('--seman-loss-top-k-percent', default=0.2, type=float,
                        help='the value lies in [0.0, 1.0]. When its value < 1.0, only compute the loss for the top k percent pixels (e.g., the top 20% pixels). This is useful for hard pixel mining.')
    

    group.add_argument('--offset-loss-name', default='mse',
                       choices=['l1', 'mse'],
                       help='type of panoptic loss')
    group.add_argument('--offset-loss-reduction', default='none',
                       choices=['none', 'mean', 'sum'],
                       help='L1Loss Reduction')

    group.add_argument('--loss-debug', default=0, type=int,
                        help='1 for freezing pan and 2 for freezing cif heads')

def configure(args):
    # apply for CompositeLoss
    CompositeLoss.background_weight = args.background_weight
    CompositeLoss.focal_gamma = args.focal_gamma
    CompositeLoss.margin = args.margin_loss

    # MultiHeadLoss
    MultiHeadLoss.task_sparsity_weight = args.task_sparsity_weight
    MultiHeadLossAutoTune.task_sparsity_weight = args.task_sparsity_weight

    # SmoothL1
    SmoothL1Loss.r_smooth = args.r_smooth



def factory_from_args(args, head_nets):
    return factory(
        head_nets,
        args.lambdas,
        reg_loss_name=args.regression_loss,
        device=args.device,
        auto_tune_mtl=args.auto_tune_mtl,
        config=args,
        loss_debug=args.loss_debug
    )


# pylint: disable=too-many-branches
def factory(head_nets, lambdas, *,
            reg_loss_name=None, device=None,
            auto_tune_mtl=False,
            config=None,
            loss_debug=0):
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

    # losses = [CompositeLoss(head_net, reg_loss) for head_net in head_nets]
    ### AMA


    losses = [CompositeLoss(head_nets[0], reg_loss)]

    if len(head_nets) > 1 and isinstance(head_nets[1], heads.AssociationMeta):
        losses.append(CompositeLoss(head_nets[1], reg_loss))

    if len(head_nets) > 1 and isinstance(head_nets[1], heads.InstanceSegHead):
        losses.append(SegmantationLoss(head_nets[1],device))

    if len(head_nets) > 1 and isinstance(head_nets[1], heads.PanopticDeeplabHead):
        losses.append(PanopticLoss(config))
    
    # print('leeeeeeeeeeeeeeeeeeeeeeeeeeeen', len(head_nets))
    # for i in range(len(head_nets)):
    #     print(type(head_nets[i]))
    if len(head_nets) > 2 and isinstance(head_nets[2], heads.CompositeFieldFused):
        print('loss________________________________-ball')
        losses.append(CompositeLoss(head_nets[2], reg_loss))
        # losses.append(PanopticLossDummy(config))
    if auto_tune_mtl:
        loss = MultiHeadLossAutoTune(losses, lambdas,
                                     sparse_task_parameters=sparse_task_parameters, loss_debug=loss_debug)
    else:
        loss = MultiHeadLoss(losses, lambdas, loss_debug=loss_debug)

    if device is not None:
        loss = loss.to(device)

    return loss

# def facrtory_loss_head_single(head_meta, out_features):
#     if head_meta.name == 'pan':
#         return heads.PanopticDeeplabHead(head_meta, out_features)
#     # elif head_meta.name == 'ball':
#     #     heads.CompositeFieldFused(head_meta, out_features)
#     elif head_meta.name in ['cif', 'cifcent', 'ball']:
#         return heads.CompositeFieldFused(head_meta, out_features)

### panoptic deeplab loss build
def build_loss_from_cfg(config, loss='semantic'):
    """Builds loss function with specific configuration.
    Args:
        config: the configuration.

    Returns:
        A nn.Module loss.
    """
    if loss == 'semantic':
        # return RegularCE(ignore_label=-1)
        if config.seman_loss_name == 'cross_entropy':
            # return CrossEntropyLoss(ignore_index=config.IGNORE, reduction='mean')
            return RegularCE(ignore_label=config.seman_loss_ignore)
        elif config.seman_loss_name == 'ohem':
            return OhemCE(ignore_label=config.seman_loss_ignore, threshold=config.seman_loss_threshold, min_kept=config.seman_loss_min_kept)
        elif config.seman_loss_name == 'hard_pixel_mining':
            print('Hard pixel mining chosen!')
            return DeepLabCE(ignore_label=config.seman_loss_ignore, top_k_percent_pixels=config.seman_loss_top_k_percent)
        elif config.seman_loss_name == 'mse':
            return MSELoss(reduction=config.seman_loss_reduction)
        elif config.seman_loss_name == 'l1':
            return L1Loss(reduction=config.seman_loss_reduction)
        else:
            raise ValueError('Unknown loss type: {}'.format(config.seman_loss_name))

    elif loss == 'offset':
        # return MSELoss(reduction='none')
        if config.offset_loss_name == 'cross_entropy':
            # return CrossEntropyLoss(ignore_index=config.IGNORE, reduction='mean')
            return RegularCE(ignore_label=config.offset_loss_ignore)
        elif config.offset_loss_name == 'ohem':
            return OhemCE(ignore_label=config.offset_loss_ignore, threshold=config.offset_loss_threshold, min_kept=config.offset_loss_min_kept)
        elif config.offset_loss_name == 'hard_pixel_mining':
            return DeepLabCE(ignore_label=config.offset_loss_ignore, top_k_percent_pixels=config.offset_loss_top_k_percent)
        elif config.offset_loss_name == 'mse':
            return MSELoss(reduction=config.offset_loss_reduction)
        elif config.offset_loss_name == 'l1':
            return L1Loss(reduction=config.offset_loss_reduction)
        else:
            raise ValueErr
