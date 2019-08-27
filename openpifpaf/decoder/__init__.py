"""Collections of decoders: fields to annotations."""

from .annotation import Annotation, AnnotationWithoutSkeleton
from .decoder import Decoder
from .factory import cli, factory_decode, factory_from_args
from .paf_stack import PafStack
from .pif import Pif
from .pifpaf import PifPaf
from .pifpaf2 import PifPaf2
from .processor import Processor
from .visualizer import Visualizer
