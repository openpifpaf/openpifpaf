"""The Processor runs the model to obtain fields and passes them to a decoder."""

import cProfile
import io
import logging
import pstats
import time

import numpy as np
import torch

from .utils import scalar_square_add_single


class Processor(object):
    def __init__(self, model, decode, *,
                 keypoint_threshold=0.0, instance_threshold=0.0,
                 debug_visualizer=None,
                 profile=None,
                 device=None):
        self.log = logging.getLogger(self.__class__.__name__)

        if profile is True:
            profile = cProfile.Profile()

        self.model = model
        self.decode = decode
        self.keypoint_threshold = keypoint_threshold
        self.instance_threshold = instance_threshold
        self.debug_visualizer = debug_visualizer
        self.profile = profile
        self.device = device

    def set_cpu_image(self, cpu_image, processed_image):
        if self.debug_visualizer is not None:
            self.debug_visualizer.set_image(cpu_image, processed_image)

    def fields(self, image_batch):
        start = time.time()
        if self.device is not None:
            image_batch = image_batch.to(self.device, non_blocking=True)

        with torch.no_grad():
            heads = self.model(image_batch)

            # to numpy
            fields = [[field.cpu().numpy() for field in head] for head in heads]

            # index by batch entry
            fields = [
                [[field[i] for field in head] for head in fields]
                for i in range(image_batch.shape[0])
            ]

        print('nn processing time', time.time() - start)
        return fields

    @staticmethod
    def soft_nms(annotations):
        if not annotations:
            return annotations

        occupied = np.zeros((
            17,
            int(max(np.max(ann.data[:, 1]) for ann in annotations) + 1),
            int(max(np.max(ann.data[:, 0]) for ann in annotations) + 1),
        ))

        annotations = sorted(annotations, key=lambda a: -a.score())
        for ann in annotations:
            joint_scales = (ann.joint_scales
                            if ann.joint_scales is not None
                            else np.ones((ann.data.shape[0]),) * 4.0)
            for xyv, occ, joint_s in zip(ann.data, occupied, joint_scales):
                v = xyv[2]
                if v == 0.0:
                    continue

                ij = np.round(xyv[:2]).astype(np.int)
                i = np.clip(ij[0], 0, occ.shape[1] - 1)
                j = np.clip(ij[1], 0, occ.shape[0] - 1)
                if occ[j, i]:
                    xyv[2] = 0.0
                else:
                    scalar_square_add_single(occ, xyv[0], xyv[1], joint_s, 1)

        annotations = [ann for ann in annotations if np.any(ann.data[:, 2] > 0.0)]
        annotations = sorted(annotations, key=lambda a: -a.score())
        return annotations

    def keypoint_sets(self, fields):
        annotations = self.annotations(fields)
        return self.keypoint_sets_from_annotations(annotations)

    @staticmethod
    def keypoint_sets_from_annotations(annotations):
        keypoint_sets = [ann.data for ann in annotations]
        scores = [ann.score() for ann in annotations]
        if not keypoint_sets:
            return np.zeros((0, 17, 3)), np.zeros((0,))
        keypoint_sets = np.array(keypoint_sets)
        scores = np.array(scores)

        return keypoint_sets, scores

    def annotations(self, fields):
        start = time.time()
        if self.profile is not None:
            self.profile.enable()

        annotations = self.decode(fields)

        # scale to input size
        output_stride = self.model.io_scales()[-1]
        for ann in annotations:
            ann.data[:, 0:2] *= output_stride
            if ann.joint_scales is not None:
                ann.joint_scales *= output_stride

        # nms
        annotations = self.soft_nms(annotations)

        # treshold
        for ann in annotations:
            kps = ann.data
            kps[kps[:, 2] < self.keypoint_threshold] = 0.0
        annotations = [ann for ann in annotations
                       if ann.score() >= self.instance_threshold]
        annotations = sorted(annotations, key=lambda a: -a.score())

        if self.profile is not None:
            self.profile.disable()
            iostream = io.StringIO()
            ps = pstats.Stats(self.profile, stream=iostream)
            ps = ps.sort_stats('tottime')
            ps.print_stats()
            ps.dump_stats('decoder.prof')
            print(iostream.getvalue())

        self.log.info('%d annotations: %s', len(annotations),
                      [np.sum(ann.data[:, 2] > 0.1) for ann in annotations])
        self.log.debug('total processing time: %.3fs', time.time() - start)
        return annotations

    def keypoint_sets_two_scales(self, fields, fields_half_scale):
        start = time.time()
        annotations = self.decode(fields)
        annotations_half_scale = self.decode(fields_half_scale)

        # scale to input size
        output_stride = self.model.io_scales()[-1]
        for ann in annotations:
            ann.data[:, 0:2] *= output_stride
            if ann.joint_scales is not None:
                ann.joint_scales *= output_stride
        for ann in annotations_half_scale:
            ann.data[:, 0:2] *= 2.0 * output_stride
            if ann.joint_scales is not None:
                ann.joint_scales *= 2.0 * output_stride
        annotations += annotations_half_scale

        # nms
        annotations = self.soft_nms(annotations)
        if not annotations:
            return np.zeros((1, 17, 3)), np.zeros((1,))

        # threshold results
        keypoint_sets, scores = [], []
        for ann in annotations:
            score = ann.score()
            if score < self.instance_threshold:
                continue
            kps = ann.data
            kps[kps[:, 2] < self.keypoint_threshold] = 0.0

            keypoint_sets.append(kps)
            scores.append(score)
        keypoint_sets = np.array(keypoint_sets)
        scores = np.array(scores)

        print('keypoint sets', keypoint_sets.shape[0],
              [np.sum(kp[:, 2] > 0.1) for kp in keypoint_sets])
        print('total processing time', time.time() - start)
        return keypoint_sets, scores
