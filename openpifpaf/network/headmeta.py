"""Head meta objects contain meta information about head networks.

This includes the name, the name of the individual fields, the composition, etc.

vector_offsets: has to be length n_vectors and identifies which vectors
get their location offset added during inference.
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

    vector_offsets = [True]

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
    only_in_field_of_view: bool = False

    n_confidences: int = 1
    n_vectors: int = 2
    n_scales: int = 2

    vector_offsets = [True, True]

    @property
    def n_fields(self):
        return len(self.skeleton)

    @staticmethod
    def concatenate(metas):
        # TODO: by keypoint name, update skeleton indices if meta.keypoints
        # is not the same for all metas.
        concatenated = Association(
            name='_'.join(m.name for m in metas),
            keypoints=metas[0].keypoints,
            sigmas=metas[0].sigmas,
            pose=metas[0].pose,
            skeleton=[s for meta in metas for s in meta.skeleton],
            sparse_skeleton=metas[0].sparse_skeleton,
            only_in_field_of_view=metas[0].only_in_field_of_view,
        )
        return concatenated


@dataclass
class Detection(Base):
    categories: List[str]

    n_confidences: int = 1
    n_vectors: int = 2
    n_scales: int = 0

    vector_offsets = [True, False]

    @property
    def n_fields(self):
        return len(self.categories)
