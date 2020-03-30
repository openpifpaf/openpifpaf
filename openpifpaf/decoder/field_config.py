import dataclasses
from typing import List


@dataclasses.dataclass
class FieldConfig:
    cif_indices: List[int] = dataclasses.field(default_factory=lambda: [0])
    caf_indices: List[int] = dataclasses.field(default_factory=lambda: [1])
    cif_strides: List[int] = dataclasses.field(default_factory=lambda: [8])
    caf_strides: List[int] = dataclasses.field(default_factory=lambda: [8])
    cif_min_scales: List[float] = dataclasses.field(default_factory=lambda: [0.0])
    caf_min_distances: List[float] = dataclasses.field(default_factory=lambda: [0.0])
    caf_max_distances: List[float] = dataclasses.field(default_factory=lambda: [None])
    seed_mask: List[int] = None
    confidence_scales: List[float] = None
    cif_visualizers: list = None
    caf_visualizers: list = None

    def verify(self):
        assert len(self.cif_strides) == len(self.cif_indices)
        assert len(self.cif_strides) == len(self.cif_min_scales)

        assert len(self.caf_strides) == len(self.caf_indices)
        assert len(self.caf_strides) == len(self.caf_min_distances)
        assert len(self.caf_strides) == len(self.caf_max_distances)
