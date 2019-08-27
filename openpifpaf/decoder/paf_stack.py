"""Decoder wrapper that stacks PAF fields."""

import logging

import numpy as np

from .decoder import Decoder


LOG = logging.getLogger(__name__)


class PafStack(Decoder):
    def __init__(self, paf_indices, wrapped_decoder):
        self.paf_indices = paf_indices
        self.wrapped_decoder = wrapped_decoder

    def __call__(self, fields, initial_annotations=None):
        other_fields = [f for i, f in enumerate(fields) if i not in self.paf_indices]
        paf_fields = [f for i, f in enumerate(fields) if i in self.paf_indices]

        fields = other_fields + [[
            np.concatenate(fs, axis=0)
            for fs in zip(*paf_fields)
        ]]
        return self.wrapped_decoder(fields, initial_annotations)
