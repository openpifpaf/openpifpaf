"""Head meta objects contain meta information about head networks.

This includes the name, the name of the individual fields, the composition, etc.
"""

from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass
class Base:
    name: str

    @property
    def n_fields(self):
        raise NotImplementedError


@dataclass
class Intensity(Base):
    keypoints: List[str]
    sigmas: List[float]
    pose: Any
    draw_skeleton: List[Tuple[int, int]] = None

    n_confidences: int = 1
    n_vectors: int = 1
    n_scales: int = 1

    @property
    def n_fields(self):
        return len(self.keypoints)


@dataclass
class Association(Base):
    keypoints: List[str]
    sigmas: List[float]
    pose: Any
    skeleton: List[Tuple[int, int]]
    sparse_skeleton: List[Tuple[int, int]] = None
    dense_to_sparse_radius: float = 2.0
    only_in_field_of_view: bool = False

    n_confidences: int = 1
    n_vectors: int = 2
    n_scales: int = 2

    @property
    def n_fields(self):
        return len(self.skeleton)


@dataclass
class Detection(Base):
    categories: List[str]

    n_confidences: int = 1
    n_vectors: int = 2
    n_scales: int = 0

    @property
    def n_fields(self):
        return len(self.categories)
