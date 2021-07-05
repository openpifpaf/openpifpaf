import copy
import logging

import numpy as np

import openpifpaf

LOG = logging.getLogger(__name__)


class NormalizePosetrack(openpifpaf.transforms.Preprocess):
    def __init__(self, ignore_missing_bbox=False, fix_annotations=True):
        self.ignore_missing_bbox = ignore_missing_bbox
        self.fix_annotations = fix_annotations

    @staticmethod
    def add_crowd_annotations(anns, image_info):
        ignore_regions = []
        if 'ignore_regions_x' in image_info:
            ignore_regions = list(
                zip(image_info['ignore_regions_x'],
                    image_info['ignore_regions_y'])
            )

        # add ignore regions to annotations
        anns += [
            {
                'bbox': [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                'keypoints': [(x, y, 0.0) for x, y in zip(xs, ys)],
                'iscrowd': True,
                'track_id': -1,
            }
            for xs, ys in ignore_regions if xs and ys
        ]

        return anns

    def normalize_annotations(self, anns, valid_area, image_id):
        # convert as much data as possible to numpy arrays to avoid every float
        # being turned into its own torch.Tensor()
        for ann in anns:
            ann['image_id'] = image_id
            ann['keypoints'] = np.asarray(ann['keypoints'], dtype=np.float32).reshape(-1, 3)

            # Posetrack does not distinguish between visible keypoints and
            # invisible keypoints. Treat all annotated keypoints as visible.
            ann['keypoints'][ann['keypoints'][:, 2] > 0.0, 2] = 2.0

            # Fix keypoints.
            # PoseTrack data contains some bad data.
            if self.fix_annotations:
                ann['keypoints'][ann['keypoints'][:, 0] < valid_area[0], 2] = 0.0
                ann['keypoints'][ann['keypoints'][:, 1] < valid_area[1], 2] = 0.0
                ann['keypoints'][ann['keypoints'][:, 0] > valid_area[0] + valid_area[2], 2] = 0.0
                ann['keypoints'][ann['keypoints'][:, 1] > valid_area[1] + valid_area[3], 2] = 0.0

            if 'bbox' in ann:
                ann['bbox'] = np.asarray(ann['bbox'], dtype=np.float32)
                # v = ann['keypoints'][:, 2] > 0
                # print(ann['bbox'], min(ann['keypoints'][v, 0]), min(ann['keypoints'][v, 1]))
                # if not np.all(ann['keypoints'][v, 0] >= ann['bbox'][0]) or \
                #    not np.all(ann['keypoints'][v, 1] >= ann['bbox'][1]) or \
                #    not np.all(ann['keypoints'][v, 0] <= ann['bbox'][0] + ann['bbox'][2]) or \
                #    not np.all(ann['keypoints'][v, 1] <= ann['bbox'][1] + ann['bbox'][3]):
                #     print(ann)
                #     assert False
            else:
                ann['bbox'] = np.zeros((4,), dtype=np.float32)
                if not self.ignore_missing_bbox:
                    assert all(c == 0.0 for c in ann['keypoints'][:, 2])

            if 'bbox_head' in ann:
                ann['bbox_head'] = np.asarray(ann['bbox_head'], dtype=np.float32)

            if 'iscrowd' not in ann:
                ann['iscrowd'] = False
                assert len(ann['keypoints']) == 17

            if not ann['iscrowd']:
                # no ear annotations
                assert ann['keypoints'][3, 2] == 0.0
                assert ann['keypoints'][4, 2] == 0.0

        return anns

    def __call__(self, image, anns, meta=None):
        meta = copy.deepcopy(meta)

        w, h = image.size
        meta_init = {
            'offset': np.array((0.0, 0.0)),
            'scale': np.array((1.0, 1.0)),
            'rotation': {'angle': 0.0, 'width': None, 'height': None},
            'valid_area': np.array((0.0, 0.0, w - 1, h - 1)),
            'hflip': False,
            'width_height': np.array((w, h)),
        }
        if meta is None:
            meta = meta_init
        else:
            for k, v in meta_init.items():
                if k in meta:
                    continue
                meta[k] = v

        image_info = anns['image']
        anns = copy.deepcopy(anns['annotations'])

        anns = self.add_crowd_annotations(anns, image_info)
        anns = self.normalize_annotations(anns, meta['valid_area'], image_info['frame_id'])
        return image, anns, meta


class NormalizeMOT(openpifpaf.transforms.Preprocess):
    @staticmethod
    def normalize_annotations(anns):
        anns = copy.deepcopy(anns)

        # convert as much data as possible to numpy arrays to avoid every float
        # being turned into its own torch.Tensor()
        for ann in anns:
            ann['keypoints'] = np.asarray(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            ann['bbox'] = np.asarray(ann['bbox'], dtype=np.float32)
            del ann['segmentation']

        return anns

    def __call__(self, image, anns, meta=None):
        anns = self.normalize_annotations(anns)

        if meta is None:
            w, h = image.size
            meta = {
                'offset': np.array((0.0, 0.0)),
                'scale': np.array((1.0, 1.0)),
                'valid_area': np.array((0.0, 0.0, w, h)),
                'hflip': False,
                'width_height': np.array((w, h)),
            }

        return image, anns, meta
