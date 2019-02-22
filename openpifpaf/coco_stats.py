"""Compute height distribution for all and medium sized bounding boxes."""

import argparse

import numpy as np
import torch

from . import datasets, transforms

ANNOTATIONS = 'data-mscoco/annotations/person_keypoints_val2017.json'
IMAGE_DIR = 'data-mscoco/images/val2017/'


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--long-edge', default=321, type=int,
                        help='long edge of input images')
    args = parser.parse_args()

    preprocess = transforms.SquareRescale(args.long_edge)
    data = datasets.CocoKeypoints(
        root=IMAGE_DIR,
        annFile=ANNOTATIONS,
        preprocess=preprocess,
    )
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=1, num_workers=2)

    bbox_heights = []
    bbox_heights_medium = []
    for i, (_, anns) in enumerate(data_loader):
        print('batch {}/{}'.format(i, len(data_loader)))

        for ann in anns:
            mask = ann['iscrowd'] == 0
            bbox_heights.append(ann['bbox'][mask, 3])

            areas = ann['bbox_original'][:, 2] * ann['bbox_original'][:, 3]
            mask_medium = mask & (32**2 < areas) & (areas < 96**2)
            bbox_heights_medium.append(ann['bbox'][mask_medium, 3])

    bbox_heights = np.array([h for batch in bbox_heights for h in batch])
    bbox_heights_medium = np.array([h for batch in bbox_heights_medium for h in batch])
    print(
        'bbox height: all = {} +/- {}, medium = {} +/- {}'.format(
            np.mean(bbox_heights), np.std(bbox_heights),
            np.mean(bbox_heights_medium), np.std(bbox_heights_medium),
        )
    )


if __name__ == '__main__':
    main()
