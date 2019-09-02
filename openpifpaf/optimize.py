import logging
import torch


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
    group_s.add_argument('--lr-burn-in-epochs', default=3, type=int,
                         help='number of epochs at the beginning with lower learning rate')
    group_s.add_argument('--lr-burn-in-factor', default=0.01, type=float,
                         help='learning pre-factor during burn-in')
    group_s.add_argument('--lr-gamma', default=0.1, type=float,
                         help='learning rate decay factor')


class LearningRateLambda(object):
    def __init__(self, decay_schedule, *,
                 gamma=0.1,
                 burn_in_epochs=3,
                 burn_in_factor=0.01,
                 steps_per_epoch=1):
        self.decay_schedule = decay_schedule
        self.gamma = gamma
        self.burn_in_epochs = burn_in_epochs
        self.burn_in_factor = burn_in_factor
        self.steps_per_epoch = steps_per_epoch

    def __call__(self, step_i):
        epoch = step_i / self.steps_per_epoch

        if epoch < self.burn_in_epochs:
            return self.burn_in_factor**(1.0 - epoch / self.burn_in_epochs)

        lambda_ = 1.0
        for d in self.decay_schedule:
            if epoch >= d:
                lambda_ *= self.gamma

        return lambda_


def factory(args, parameters):
    if args.amsgrad:
        args.adam = True

    if args.adam:
        logging.info('Adam optimizer')
        optimizer = torch.optim.Adam(
            (p for p in parameters if p.requires_grad),
            lr=args.lr, betas=(args.momentum, args.beta2),
            weight_decay=args.weight_decay, eps=args.adam_eps, amsgrad=args.amsgrad)
    else:
        logging.info('SGD optimizer')
        optimizer = torch.optim.SGD(
            (p for p in parameters if p.requires_grad),
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
            nesterov=args.nesterov)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, [LearningRateLambda(args.lr_decay,
                                       gamma=args.lr_gamma,
                                       burn_in_epochs=args.lr_burn_in_epochs,
                                       burn_in_factor=args.lr_burn_in_factor)])
    return optimizer, scheduler
