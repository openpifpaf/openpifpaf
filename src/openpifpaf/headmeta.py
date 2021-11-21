"""Head meta objects contain meta information about head networks.

This includes the name, the name of the individual fields, the composition, etc.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, List, Tuple

import numpy as np


@dataclass
class Base:
    name: str
    dataset: str

    head_index: int = field(default=None, init=False)
    base_stride: int = field(default=None, init=False)
    upsample_stride: int = field(default=1, init=False)

    @property
    def stride(self) -> int:
        if self.base_stride is None:
            return None
        return self.base_stride // self.upsample_stride

    @property
    def n_fields(self) -> int:
        raise NotImplementedError


@dataclass
class Cif(Base):
    """Head meta data for a Composite Intensity Field (CIF)."""

    keypoints: List[str]
    sigmas: List[float]
    pose: Any = None
    draw_skeleton: List[Tuple[int, int]] = None
    score_weights: List[float] = None

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 1
    n_scales: ClassVar[int] = 1

    vector_offsets = [True]
    decoder_min_scale = 0.0
    decoder_seed_mask: List[int] = None

    training_weights: List[float] = None

    @property
    def n_fields(self):
        return len(self.keypoints)


@dataclass
class Caf(Base):
    """Head meta data for a Composite Association Field (CAF)."""

    keypoints: List[str]
    sigmas: List[float]
    skeleton: List[Tuple[int, int]]
    pose: Any = None
    sparse_skeleton: List[Tuple[int, int]] = None
    dense_to_sparse_radius: float = 2.0
    only_in_field_of_view: bool = False

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 2

    vector_offsets = [True, True]
    decoder_min_distance = 0.0
    decoder_max_distance = float('inf')
    decoder_confidence_scales: List[float] = None

    training_weights: List[float] = None

    @property
    def n_fields(self):
        return len(self.skeleton)

    @staticmethod
    def concatenate(metas):
        # TODO: by keypoint name, update skeleton indices if meta.keypoints
        # is not the same for all metas.
        concatenated = Caf(
            name='_'.join(m.name for m in metas),
            dataset=metas[0].dataset,
            keypoints=metas[0].keypoints,
            sigmas=metas[0].sigmas,
            pose=metas[0].pose,
            skeleton=[s for meta in metas for s in meta.skeleton],
            sparse_skeleton=metas[0].sparse_skeleton,
            only_in_field_of_view=metas[0].only_in_field_of_view,
            decoder_confidence_scales=[
                s
                for meta in metas
                for s in (meta.decoder_confidence_scales
                          if meta.decoder_confidence_scales
                          else [1.0 for _ in meta.skeleton])
            ]
        )
        concatenated.head_index = metas[0].head_index
        concatenated.base_stride = metas[0].base_stride
        concatenated.upsample_stride = metas[0].upsample_stride
        return concatenated


@dataclass
class CifDet(Base):
    """Head meta data for a Composite Intensity Field (CIF) for Detection."""

    categories: List[str]

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 0

    vector_offsets = [True, False]
    decoder_min_scale = 0.0

    training_weights: List[float] = None

    @property
    def n_fields(self):
        return len(self.categories)


@dataclass
class TSingleImageCif(Cif):
    """Single-Image CIF head in tracking models."""


@dataclass
class TSingleImageCaf(Caf):
    """Single-Image CAF head in tracking models."""


@dataclass
class Tcaf(Base):
    """Tracking Composite Association Field."""

    keypoints_single_frame: List[str]
    sigmas_single_frame: List[float]
    pose_single_frame: Any
    draw_skeleton_single_frame: List[Tuple[int, int]] = None
    keypoints: List[str] = None
    sigmas: List[float] = None
    pose: Any = None
    draw_skeleton: List[Tuple[int, int]] = None
    only_in_field_of_view: bool = False

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 2

    training_weights: List[float] = None

    vector_offsets = [True, True]

    def __post_init__(self):
        if self.keypoints is None:
            self.keypoints = np.concatenate((
                self.keypoints_single_frame,
                self.keypoints_single_frame,
            ), axis=0)
        if self.sigmas is None:
            self.sigmas = np.concatenate((
                self.sigmas_single_frame,
                self.sigmas_single_frame,
            ), axis=0)
        if self.pose is None:
            self.pose = np.concatenate((
                self.pose_single_frame,
                self.pose_single_frame,
            ), axis=0)
        if self.draw_skeleton is None:
            self.draw_skeleton = np.concatenate((
                self.draw_skeleton_single_frame,
                self.draw_skeleton_single_frame,
            ), axis=0)

    @property
    def skeleton(self):
        return [(i + 1, i + 1 + len(self.keypoints_single_frame))
                for i, _ in enumerate(self.keypoints_single_frame)]

    @property
    def n_fields(self):
        return len(self.keypoints_single_frame)
