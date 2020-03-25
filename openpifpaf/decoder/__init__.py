"""Collections of decoders: fields to annotations."""

from .annotation import Annotation, AnnotationWithoutSkeleton
from .factory import cli, configure, factory_decode, factory_from_args
from .occupancy import Occupancy
from .pif import Pif
from .pifpaf import PifPaf
from .cifcaf_frontier import CifCafFrontier
from .pafs_dijkstra import PafsDijkstra
from .processor import Processor
from .visualizer import Visualizer
