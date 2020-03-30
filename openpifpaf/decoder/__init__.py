"""Collections of decoders: fields to annotations."""

from .annotation import Annotation, AnnotationWithoutSkeleton
from .factory import cli, configure, factory_decode, factory_from_args
from .field_config import FieldConfig
from .occupancy import Occupancy
from .processor import Processor
