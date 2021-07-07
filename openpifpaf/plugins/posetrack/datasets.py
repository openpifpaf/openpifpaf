from collections import defaultdict
import json
import logging
import os
import random

import PIL
import pysparkling
import torch

LOG = logging.getLogger(__name__)


class Posetrack2018(torch.utils.data.Dataset):
    """Dataset reader for Posetrack2018."""

    def __init__(self, annotation_files, data_root, *, preprocess,
                 group=None,
                 only_annotated=False,
                 max_per_sequence=None,
                 n_image_stack=1):
        super().__init__()

        if group is None:
            group = (0,)
        self.group = group

        self.preprocess = preprocess
        self.data_root = data_root
        self.only_annotated = only_annotated
        self.max_per_sequence = max_per_sequence
        self.n_image_stack = n_image_stack

        spark_context = pysparkling.Context()
        self.files = (
            spark_context
            .wholeTextFiles(annotation_files)
            .mapValues(json.loads)
            .cache()
        )
        self.annotations = self.files.flatMap(self.group_annotations).collect()

        LOG.info('sequences = %d, annotations = %d',
                 self.files.count(), len(self.annotations))

    def meta_images(self):
        return self.files.mapValues(lambda r: r['images']).collectAsMap()

    def meta_categories(self):
        return self.files.mapValues(lambda r: r['categories']).collectAsMap()

    def group_annotations(self, file_raw_annotation):
        file_name, raw_annotation = file_raw_annotation

        all_images = raw_annotation['images']
        all_annotations = raw_annotation.get('annotations', [])  # not in test set
        all_annotations_by_image_id = defaultdict(list)
        for ann in all_annotations:
            all_annotations_by_image_id[ann['image_id']].append(ann)

        frame_groups = self.group
        if not isinstance(frame_groups[0], (tuple, list)):
            frame_groups = [frame_groups]

        groups = []
        for i in range(0, len(all_images)):
            for this_group in frame_groups:
                if any(i + gi < 0 for gi in this_group):
                    continue
                image_groups = [all_images[i + gi] for gi in this_group]
                groups.append([
                    {
                        'annotation_file': file_name,
                        'image': image,
                        'annotations': all_annotations_by_image_id[image['frame_id']],
                    }
                    for image in image_groups
                ])

        if self.only_annotated:
            groups = [group for group in groups if all(s['annotations'] for s in group)]

        if self.max_per_sequence:
            if len(groups) > self.max_per_sequence:
                LOG.debug('groups per file %d -> %d', len(groups), self.max_per_sequence)
                groups = random.choices(groups, k=self.max_per_sequence)

        return groups

    def __getitem__(self, index):
        group = self.annotations[index]

        images = []
        anns = []
        metas = []

        for group_i, data in enumerate(group):
            local_file_path = os.path.join(self.data_root, data['image']['file_name'])
            with open(local_file_path, 'rb') as f:
                image = PIL.Image.open(f).convert('RGB')
                images.append(image)

            anns.append(data)

            metas.append({
                'dataset_index': index,
                'image_id': data['image']['frame_id'],
                'file_name': data['image']['file_name'],
                'local_file_path': local_file_path,
                'annotation_file': data['annotation_file'],
                'group_i': group_i,
            })

        # preprocess images and targets
        images, anns, metas = self.preprocess(images, anns, metas)
        return images, anns, metas

    def __len__(self):
        return len(self.annotations)


class Posetrack2017(torch.utils.data.Dataset):
    """Dataset reader for Posetrack."""

    def __init__(self, annotation_files, data_root, *, preprocess,
                 group=None,
                 only_annotated=False,
                 max_per_sequence=None,
                 n_image_stack=1):
        super().__init__()

        if group is None:
            group = (0,)
        self.group = group

        self.preprocess = preprocess
        self.data_root = data_root
        self.only_annotated = only_annotated
        self.max_per_sequence = max_per_sequence
        self.n_image_stack = n_image_stack

        spark_context = pysparkling.Context()
        self.files = (
            spark_context
            .wholeTextFiles(annotation_files)
            .mapValues(json.loads)
            .cache()
        )
        self.annotations = self.files.flatMap(self.group_annotations).collect()

        LOG.info('sequences = %d, annotations = %d',
                 self.files.count(), len(self.annotations))

    def meta_images(self):
        return (
            self.files
            .mapValues(lambda r: [i['image'][0]['name'] for i in r['annolist']])
            .collectAsMap()
        )

    def meta_categories(self):
        return (
            self.files
            .mapValues(lambda _: [{
                'supercategory': 'person',
                'id': 1,
                'name': 'person',
            }])
            .collectAsMap()
        )

    def group_annotations(self, file_raw_annotation):
        file_name, raw_annotation = file_raw_annotation

        all_annotations = raw_annotation.get('annolist', [])  # not in test set

        groups = []
        start_frame = 0 if not self.only_annotated else 12
        for i in range(start_frame, len(all_annotations)):
            this_group = self.group
            if isinstance(this_group, (tuple, list)) and isinstance(this_group[0], (tuple, list)):
                this_group = random.choice(this_group)

            group = [
                {
                    'annotation_file': file_name,
                    'image': {
                        'frame_id': (
                            i + gi
                            if i + gi >= 0
                            else i - 4 if i - 4 >= 0 else i
                        ),
                    },
                }
                for gi in this_group
            ]

            # extract the image info to be like 2018
            for g in group:
                annotations = all_annotations[g['image']['frame_id']]
                g['image']['file_name'] = annotations['image'][0]['name']
                # g['annotations'] = annotations
                g['annotations'] = []

            groups.append(group)

        if self.only_annotated:
            groups = [group for group in groups if all(s['annotations'] for s in group)]

        if self.max_per_sequence:
            if len(groups) > self.max_per_sequence:
                LOG.debug('groups per file %d -> %d', len(groups), self.max_per_sequence)
                groups = random.choices(groups, k=self.max_per_sequence)

        return groups

    def __getitem__(self, index):
        group = self.annotations[index]

        images = []
        anns = []
        metas = []

        for group_i, data in enumerate(group):
            with open(os.path.join(self.data_root, data['image']['file_name']), 'rb') as f:
                image = PIL.Image.open(f).convert('RGB')
                images.append(image)

            anns.append(data)

            metas.append({
                'dataset_index': index,
                'image_id': data['image']['frame_id'],
                'file_name': data['image']['file_name'],
                'annotation_file': data['annotation_file'],
                'group_i': group_i,
            })

        # preprocess images and targets
        images, anns, metas = self.preprocess(images, anns, metas)
        return images, anns, metas

    def __len__(self):
        return len(self.annotations)
