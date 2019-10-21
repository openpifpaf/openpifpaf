import logging
import torch

LOG = logging.getLogger(__name__)


def cli(parser):
    group = parser.add_argument_group('optimizer')
    group.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum, beta1 in Adam')
    group.add_argument('--beta2', type=float, default=0.999,
                       help='beta2 for Adam/AMSGrad')
    group.add_argument('--adam-eps', type=float, default=1e-6,
                       help='eps value for Adam/AMSGrad')
    group.add_argument('--no-nesterov', dest='nesterov', default=True, action='store_false',
                       help='do not use Nesterov momentum for SGD update')
    group.add_argument('--weight-decay', type=float, default=0.0,
                       help='SGD/Adam/AMSGrad weight decay')
    group.add_argument('--adam', action='store_true',
                       help='use Adam optimizer')
    group.add_argument('--amsgrad', action='store_true',
                       help='use Adam optimizer with AMSGrad option')

    group_s = parser.add_argument_group('learning rate scheduler')
    group_s.add_argument('--lr', type=float, default=1e-3,
                         help='learning rate')
    group_s.add_argument('--lr-decay', default=[], nargs='+', type=int,
                         help='epochs at which to decay the learning rate')
    group_s.add_argument('--lr-burn-in-epochs', default=2, type=int,
                         help='number of epochs at the beginning with lower learning rate')
    group_s.add_argument('--lr-burn-in-factor', default=0.001, type=float,
                         help='learning pre-factor during burn-in')
    group_s.add_argument('--lr-gamma', default=0.1, type=float,
                         help='learning rate decay factor')


class LearningRateLambda(object):
    def __init__(self, burn_in, decay_schedule, *,
                 gamma=0.1,
                 burn_in_factor=0.01):
        self.burn_in = burn_in
        self.decay_schedule = decay_schedule
        self.gamma = gamma
        self.burn_in_factor = burn_in_factor

    def __call__(self, step_i):
        if step_i < self.burn_in:
            return self.burn_in_factor**(1.0 - step_i / self.burn_in)

        lambda_ = 1.0
        for d in self.decay_schedule:
            if step_i >= d:
                lambda_ *= self.gamma

        return lambda_


def factory_optimizer(args, parameters):
    if args.amsgrad:
        args.adam = True

    if args.adam:
        LOG.info('Adam optimizer')
        optimizer = torch.optim.Adam(
            (p for p in parameters if p.requires_grad),
            lr=args.lr, betas=(args.momentum, args.beta2),
            weight_decay=args.weight_decay, eps=args.adam_eps, amsgrad=args.amsgrad)
    else:
        LOG.info('SGD optimizer')
        optimizer = torch.optim.SGD(
            (p for p in parameters if p.requires_grad),
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
            nesterov=args.nesterov)

    return optimizer


def factory_lrscheduler(args, optimizer, training_batches_per_epoch):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [LearningRateLambda(args.lr_burn_in_epochs * training_batches_per_epoch,
                            [s * training_batches_per_epoch for s in args.lr_decay],
                            gamma=args.lr_gamma,
                            burn_in_factor=args.lr_burn_in_factor)],
    )
