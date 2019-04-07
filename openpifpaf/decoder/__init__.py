"""Collections of decoders: fields to annotations."""

from .annotation import Annotation, AnnotationWithoutSkeleton
from .decoder import Decoder
from .factory import cli, factory_decode, factory_from_args
from .pif import Pif
from .pifpaf import PifPaf
from .pifspafs import PifsPafs
from .processor import Processor
from .visualizer import Visualizer
