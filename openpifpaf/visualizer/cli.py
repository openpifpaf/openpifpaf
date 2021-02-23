from .base import Base


def cli(parser):
    group = parser.add_argument_group('visualizer')
    group.add_argument('--debug-indices', default=[], nargs='+',
                       help=('Indices of fields to create debug plots for '
                             'of the form headname:fieldindex, e.g. cif:5. '
                             'Optionally, specify the visualization type, '
                             'e.g. cif:5:hr for the high resolution plot only. '
                             'Use comma separation to specify multiple '
                             'head names, field indices or visualization '
                             'types, e.g. cif:5,6:confidence,hr to visualize '
                             'CIF fields 5 and 6 but only show confidence and '
                             'hr.'))


def configure(args):
    # configure visualizer
    Base.set_all_indices(args.debug_indices)
