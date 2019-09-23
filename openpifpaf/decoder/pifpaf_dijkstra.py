"""Decoder for pif-paf fields."""

import logging
import time

import numpy as np

from . import generator
from .decoder import Decoder
from .paf_scored import PafScored
from .pif_hr import PifHr
from .pif_seeds import PifSeeds
from .utils import normalize_pif, normalize_paf

LOG = logging.getLogger(__name__)


class PifPafDijkstra(Decoder):
    default_force_complete = True
    default_connection_method = 'max'
    default_fixed_b = None
    default_pif_fixed_scale = None
    default_paf_th = 0.1

    def __init__(self, stride, *,
                 seed_threshold=0.2,
                 head_indices=None,
                 skeleton=None,
                 confidence_scales=None,
                 debug_visualizer=None):
        self.head_indices = head_indices
        self.skeleton = skeleton

        self.stride = stride
        self.seed_threshold = seed_threshold
        self.debug_visualizer = debug_visualizer
        self.force_complete = self.default_force_complete
        self.connection_method = self.default_connection_method
        self.fixed_b = self.default_fixed_b
        self.pif_fixed_scale = self.default_pif_fixed_scale

        self.pif_nn = 16
        self.paf_nn = 1 if self.connection_method == 'max' else 35
        self.paf_th = self.default_paf_th

        self.confidence_scales = confidence_scales

    # @classmethod
    # def cli(cls, parser):
    #     group = parser.add_argument_group('PifPaf decoder')
    #     group.add_argument('--fixed-b', default=None, type=float,
    #                        help='overwrite b with fixed value, e.g. 0.5')
    #     group.add_argument('--pif-fixed-scale', default=None, type=float,
    #                        help='overwrite pif scale with a fixed value')
    #     group.add_argument('--paf-th', default=cls.default_paf_th, type=float,
    #                        help='paf threshold')
    #     group.add_argument('--connection-method',
    #                        default=cls.default_connection_method,
    #                        choices=('median', 'max', 'blend'),
    #                        help='connection method to use, max is faster')

    @classmethod
    def apply_args(cls, args):
        cls.default_fixed_b = args.fixed_b
        cls.default_pif_fixed_scale = args.pif_fixed_scale
        cls.default_paf_th = args.paf_th
        cls.default_connection_method = args.connection_method

        # arg defined in factory
        cls.default_force_complete = args.force_complete_pose

    def __call__(self, fields, initial_annotations=None):
        start = time.perf_counter()

        pif, paf = fields[self.head_indices[0]], fields[self.head_indices[1]]
        if self.confidence_scales:
            cs = np.array(self.confidence_scales).reshape((-1, 1, 1))
            # print(paf[0].shape, cs.shape)
            # print('applying cs', cs)
            paf[0] = np.copy(paf[0])
            paf[0] *= cs
        if self.debug_visualizer:
            self.debug_visualizer.pif_raw(pif, self.stride)
            self.debug_visualizer.paf_raw(paf, self.stride, reg_components=3)
        paf = normalize_paf(*paf, fixed_b=self.fixed_b)
        pif = normalize_pif(*pif, fixed_scale=self.pif_fixed_scale)
        pifhr = PifHr(self.pif_nn).fill(pif, self.stride)
        seeds = PifSeeds(pifhr.targets, self.seed_threshold,
                         debug_visualizer=self.debug_visualizer).fill(pif, self.stride).get()
        paf_scored = PafScored(
            np.minimum(1.0, pifhr.targets),
            self.skeleton,
            score_th=self.paf_th,
        ).fill(paf, self.stride)

        gen = generator.Dijkstra(
            pifhr, paf_scored, seeds,
            seed_threshold=self.seed_threshold,
            connection_method=self.connection_method,
            paf_nn=self.paf_nn,
            paf_th=self.paf_th,
            skeleton=self.skeleton,
            debug_visualizer=self.debug_visualizer,
        )

        annotations = gen.annotations(initial_annotations=initial_annotations)
        if self.force_complete:
            gen.paf_scored = PafScored(
                np.minimum(1.0, pifhr.targets),
                self.skeleton,
                score_th=0.0001,
            ).fill(paf, self.stride)
            annotations = gen.complete_annotations(annotations)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations
