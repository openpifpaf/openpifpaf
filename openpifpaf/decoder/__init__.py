"""Collections of decoders: fields to annotations."""

from .annotation import Annotation, AnnotationWithoutSkeleton
from .factory import cli, factory_decode, factory_from_args
from .pif import Pif
from .pifpaf import PifPaf
from .pifpaf_dijkstra import PifPafDijkstra
from .processor import Processor
from .visualizer import Visualizer
