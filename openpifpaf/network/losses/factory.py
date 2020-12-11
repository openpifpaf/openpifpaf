import logging

from ... import headmeta
from . import components
from .composite import CompositeLoss
from .multi_head import MultiHeadLoss, MultiHeadLossAutoTuneKendall, MultiHeadLossAutoTuneVariance

LOG = logging.getLogger(__name__)

#: headmeta class to Loss class
LOSSES = {
    headmeta.Cif: CompositeLoss,
    headmeta.Caf: CompositeLoss,
    headmeta.CifDet: CompositeLoss,
}
LOSS_COMPONENTS = {
    components.Bce,
    components.SmoothL1,
}


def cli(parser):
    group = parser.add_argument_group('losses')
    group.add_argument('--lambdas', default=None, type=float, nargs='+',
                       help='prefactor for head losses')
    group.add_argument('--regression-loss', default='laplace',
                       choices=['smoothl1', 'smootherl1', 'l1', 'laplace'],
                       help='type of regression loss')
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

    for l in set(LOSSES.values()):
        l.cli(parser)
    for lc in LOSS_COMPONENTS:
        lc.cli(parser)


def configure(args):
    # MultiHeadLoss
    MultiHeadLoss.task_sparsity_weight = args.task_sparsity_weight
    MultiHeadLossAutoTuneKendall.task_sparsity_weight = args.task_sparsity_weight
    MultiHeadLossAutoTuneVariance.task_sparsity_weight = args.task_sparsity_weight

    for l in set(LOSSES.values()):
        l.configure(args)
    for lc in LOSS_COMPONENTS:
        lc.configure(args)


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
        reg_loss = components.SmoothL1()
    elif reg_loss_name == 'l1':
        reg_loss = components.l1_loss
    elif reg_loss_name == 'laplace':
        reg_loss = components.laplace_loss
    elif reg_loss_name is None:
        reg_loss = components.laplace_loss
    else:
        raise Exception('unknown regression loss type {}'.format(reg_loss_name))

    sparse_task_parameters = None
    if MultiHeadLoss.task_sparsity_weight:
        sparse_task_parameters = []
        for head_net in head_nets:
            if getattr(head_net, 'sparse_task_parameters', None) is not None:
                sparse_task_parameters += head_net.sparse_task_parameters
            elif hasattr(head_net, 'conv'):
                sparse_task_parameters.append(head_net.conv.weight)
            else:
                raise Exception('unknown l1 parameters for given head: {} ({})'
                                ''.format(head_net.meta.name, type(head_net)))

    losses = [LOSSES[head_net.meta.__class__](head_net, reg_loss)
              for head_net in head_nets]
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
