from abc import ABCMeta, abstractmethod
import copy


class Preprocess(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, image, anns, meta):
        """Implementation of preprocess operation."""

    @staticmethod
    def keypoint_sets_inverse(keypoint_sets, meta):
        keypoint_sets = keypoint_sets.copy()

        keypoint_sets[:, :, 0] += meta['offset'][0]
        keypoint_sets[:, :, 1] += meta['offset'][1]

        keypoint_sets[:, :, 0] = keypoint_sets[:, :, 0] / meta['scale'][0]
        keypoint_sets[:, :, 1] = keypoint_sets[:, :, 1] / meta['scale'][1]

        if meta['hflip']:
            w = meta['width_height'][0]
            keypoint_sets[:, :, 0] = -keypoint_sets[:, :, 0] + (w - 1)
            for keypoints in keypoint_sets:
                if meta.get('horizontal_swap'):
                    keypoints[:] = meta['horizontal_swap'](keypoints)

        return keypoint_sets

    @staticmethod
    def annotations_inverse(annotations, meta):
        annotations = copy.deepcopy(annotations)

        for ann in annotations:
            ann.data[:, 0] += meta['offset'][0]
            ann.data[:, 1] += meta['offset'][1]

            ann.data[:, 0] = ann.data[:, 0] / meta['scale'][0]
            ann.data[:, 1] = ann.data[:, 1] / meta['scale'][1]

            if meta['hflip']:
                w = meta['width_height'][0]
                ann.data[:, 0] = -ann.data[:, 0] + (w - 1)
                if meta.get('horizontal_swap'):
                    ann.data[:] = meta['horizontal_swap'](ann.data)

        for ann in annotations:
            for _, __, c1, c2 in ann.decoding_order:
                c1[:2] += meta['offset']
                c2[:2] += meta['offset']

                c1[:2] /= meta['scale']
                c2[:2] /= meta['scale']

        return annotations
