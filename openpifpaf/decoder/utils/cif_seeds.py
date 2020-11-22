import logging
import time

import torch

# pylint: disable=import-error
from ...functional import scalar_values
from .cif_hr import CifHr
from ... import headmeta, visualizer

LOG = logging.getLogger(__name__)


class CifSeeds:
    threshold = 0.5
    score_scale = 1.0
    debug_visualizer = visualizer.Seeds()
    ablation_nms = False
    ablation_no_rescore = False

    def __init__(self, cifhr: CifHr):
        self.cifhr = cifhr
        self.seeds = []

    def fill(self, all_fields, metas):
        for meta in metas:
            self.fill_single(all_fields, meta)
        return self

    def fill_single(self, all_fields, meta: headmeta.Cif):
        start = time.perf_counter()

        sv = 0.0

        cif = all_fields[meta.head_index]
        if self.ablation_nms:
            cif = self.nms(cif)
        for field_i, p in enumerate(cif):
            if meta.decoder_seed_mask is not None and not meta.decoder_seed_mask[field_i]:
                continue
            p = p[:, p[0] > self.threshold]
            if meta.decoder_min_scale:
                p = p[:, p[4] > meta.decoder_min_scale / meta.stride]
            c, x, y, _, s = p

            start_sv = time.perf_counter()
            if self.ablation_no_rescore:
                v = c
            else:
                v = scalar_values(self.cifhr[field_i],
                                  x * meta.stride, y * meta.stride,
                                  default=0.0)
                v = 0.9 * v + 0.1 * c
            sv += time.perf_counter() - start_sv

            if self.score_scale != 1.0:
                v = v * self.score_scale
            m = v > self.threshold
            x, y, v, s = x[m] * meta.stride, y[m] * meta.stride, v[m], s[m] * meta.stride

            for vv, xx, yy, ss in zip(v, x, y, s):
                self.seeds.append((vv, field_i, xx, yy, ss))

        LOG.debug('seeds %d, %.3fs (C++ %.3fs)', len(self.seeds), time.perf_counter() - start, sv)
        return self

    def get(self):
        self.debug_visualizer.predicted(self.seeds)
        return sorted(self.seeds, reverse=True)

    @staticmethod
    def nms(cif):
        classes_x = cif[:, 0:1]
        classes_x = torch.from_numpy(classes_x)

        classes_x_max = torch.nn.functional.max_pool2d(
            classes_x,
            kernel_size=3, padding=1, stride=1,
        )
        cif[:, 0:1][classes_x < classes_x_max] = 0.0

        return cif


class CifDetSeeds(CifSeeds):
    def fill_single(self, all_fields, meta: headmeta.CifDet):
        start = time.perf_counter()

        cif = all_fields[meta.head_index]
        for field_i, p in enumerate(cif):
            p = p[:, p[0] > self.threshold]
            if meta.decoder_min_scale:
                p = p[:, p[4] > meta.decoder_min_scale / meta.stride]
                p = p[:, p[5] > meta.decoder_min_scale / meta.stride]
            c, x, y, w, h, _, __ = p

            if self.ablation_no_rescore:
                v = c
            else:
                v = scalar_values(self.cifhr[field_i],
                                  x * meta.stride, y * meta.stride,
                                  default=0.0)
                v = 0.9 * v + 0.1 * c

            if self.score_scale != 1.0:
                v = v * self.score_scale
            m = v > self.threshold

            x = x[m] * meta.stride
            y = y[m] * meta.stride
            v = v[m]
            w = w[m] * meta.stride
            h = h[m] * meta.stride

            for vv, xx, yy, ww, hh in zip(v, x, y, w, h):
                self.seeds.append((vv, field_i, xx, yy, ww, hh))

        LOG.debug('seeds %d, %.3fs', len(self.seeds), time.perf_counter() - start)
        return self
