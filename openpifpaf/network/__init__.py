"""Backbone networks, head networks and tools for training."""

from .factory import cli, configure, factory, factory_from_args, local_checkpoint_path
from .trainer import Trainer
from . import losses

# from .panoptic_deeplab import RegularCE, OhemCE, DeepLabCE

# L1Loss = nn.L1Loss
# MSELoss = nn.MSELoss
# CrossEntropyLoss = nn.CrossEntropyLoss
