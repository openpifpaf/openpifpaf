from .module import DataModule
from .multiloader import MultiLoader
from .multimodule import MultiDataModule

DATAMODULES = {}


def factory(dataset):
    if '-' in dataset:
        datamodules = [factory(ds) for ds in dataset.split('-')]
        return MultiDataModule(datamodules)

    if dataset not in DATAMODULES:
        raise Exception('dataset {} unknown'.format(dataset))
    return DATAMODULES[dataset]()


def cli(parser):
    DataModule.cli_module(parser)

    group = parser.add_argument_group('generic data module parameters')
    group.add_argument('--dataset')
    group.add_argument('--dataset-weights', default=None, nargs='+', type=float,
                       help='n-1 weights for the datasets')

    for dm in DATAMODULES.values():
        dm.cli(parser)


def configure(args):
    DataModule.configure_module(args)
    MultiLoader.weights = args.dataset_weights

    for dm in DATAMODULES.values():
        dm.configure(args)
