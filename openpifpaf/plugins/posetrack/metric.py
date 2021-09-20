from collections import defaultdict
import datetime
import glob
import json
import logging
import os
import subprocess

import numpy as np
import openpifpaf

from .constants import POSETRACK_INDEX2017TO2018

LOG = logging.getLogger(__name__)


class Posetrack(openpifpaf.metric.Base):
    def __init__(self, *, images, categories,
                 ground_truth=None, output_format='2018'):
        super().__init__()

        self.images_by_file = images
        self.categories_by_file = categories
        self.ground_truth_directory = ground_truth
        self.output_format = output_format

        self.predictions_by_file = defaultdict(list)
        self.output_dir_suffix = '{}-{}'.format(
            output_format,
            datetime.datetime.now().strftime('%y%m%d-%H%M%S'),
        )

        self._written_mot_stats_file = None
        self._written_ap_stats_file = None

    def stats(self):
        if self._written_ap_stats_file is None \
           or self._written_mot_stats_file is None:
            return {
                'stats': [],
                'text_labels': [],
            }

        with open(self._written_mot_stats_file, 'r') as f_mot:
            mot_stats = json.load(f_mot)
        with open(self._written_ap_stats_file, 'r') as f_ap:
            ap_stats = json.load(f_ap)

        mot_index_by_name = {n: int(i) for i, n in mot_stats['names'].items()}
        ap_index_by_name = {n: int(i) for i, n in ap_stats['names'].items()}

        return {
            'stats': [
                mot_stats['mota'][mot_index_by_name['total']],
                0.5 * (ap_stats['ap'][ap_index_by_name['right_wrist']]
                       + ap_stats['ap'][ap_index_by_name['left_wrist']]),
                0.5 * (ap_stats['ap'][ap_index_by_name['right_ankle']]
                       + ap_stats['ap'][ap_index_by_name['left_ankle']]),
                ap_stats['ap'][ap_index_by_name['total']],
            ],
            'text_labels': [
                'MOTA', 'AP_wrists', 'AP_ankles', 'AP',
            ],
        }

    def accumulate(self, predictions, image_meta, *, ground_truth=None):  # pylint: disable=unused-argument
        annotation_file = image_meta['annotation_file']

        # make sure an entry for this annotation file is created
        # even when a file does not have any predictions
        if annotation_file not in self.predictions_by_file:
            self.predictions_by_file[annotation_file] = []

        for ann in predictions:
            keypoints = np.copy(ann.data)

            # remove points outside image
            w, h = image_meta['width_height']
            keypoints[keypoints[:, 0] < 0.0, 2] = 0.0
            keypoints[keypoints[:, 1] < 0.0, 2] = 0.0
            keypoints[keypoints[:, 0] > w - 1, 2] = 0.0
            keypoints[keypoints[:, 1] > h - 1, 2] = 0.0

            # cleanup
            keypoints[:, 2] = np.clip(keypoints[:, 2], 0.0, 1.0)
            keypoints[keypoints[:, 2] == 0.0, :2] = 0.0

            bbox = [float(v) for v in ann.bbox()]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            bbox = [round(v, 2) for v in bbox]

            self.predictions_by_file[annotation_file].append({
                'bbox': bbox,
                'keypoints': [round(k, 2) for k in keypoints.reshape(-1).tolist()],
                'track_id': ann.id_,
                'image_id': image_meta['image_id'],
                'image_file': image_meta['file_name'],
                'category_id': 1,
                'scores': [round(s, 2) for s in keypoints[:, 2].tolist()],
                'score': max(0.01, round(ann.score, 2)),
            })

    def _write2018(self, output_dir, annotation_file, *, additional_data=None):
        sequence_name = os.path.basename(annotation_file)
        out_name = '{}/{}'.format(output_dir, sequence_name)
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
        LOG.info('writing %s', out_name)

        data = {
            'images': self.images_by_file[annotation_file],
            'annotations': self.predictions_by_file[annotation_file],
            'categories': self.categories_by_file[annotation_file],
        }
        if additional_data:
            data = dict(**data, **additional_data)
        with open(out_name, 'w') as f:
            json.dump(data, f)
        LOG.info('wrote %s', out_name)

    def _write2017(self, output_dir, annotation_file, *, additional_data=None):
        sequence_name = os.path.basename(annotation_file)
        out_name = '{}/{}'.format(output_dir, sequence_name)
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
        LOG.info('writing %s', out_name)

        data = {
            # 'args': sys.argv,
            'annolist': self._format_2017(
                self.images_by_file[annotation_file],
                self.predictions_by_file[annotation_file],
            ),
            # 'categories': categories[self.annotation_file],
            # 'total_time': round(total_time, 3),
            # 'nn_time': round(self.nn_time, 3),
            # 'decoder_time': round(self.decoder_time, 3),
            # 'n_images': self.n_images,
        }
        if additional_data:
            data = dict(**data, **additional_data)
        with open(out_name, 'w') as f:
            json.dump(data, f)
        LOG.info('wrote %s', out_name)

    # def _annotation_name_2017(self):
    #     lookup = {}
    #     for fn in os.listdir(self.gt_annotations_2017_folder):
    #         if not fn.endswith('.json'):
    #             continue
    #         sequence_index, _, __ = fn.partition('_')
    #         sequence_index = int(sequence_index)
    #         lookup[sequence_index] = fn

    #     current_sequence_index, _, __ = os.path.basename(self.annotation_file).partition('_')
    #     current_sequence_index = int(current_sequence_index)

    #     return lookup.get(current_sequence_index)

    # def _image_names_2017(self, annotation_file):
    #     with open(annotation_file, 'r') as f:
    #         data_2017 = json.load(f)
    #     return [i['image'][0]['name'] for i in data_2017['annolist']]

    @staticmethod
    def _format_2017(image_names, predictions):
        r"""Create a datastructure that transforms to Posetrack2017 json.

        2017 target:
{
   "annolist": [
       {
           "image": [
               {
                  "name": "images\/bonn_5sec\/000342_mpii\/00000001.jpg"
               }
           ],
           "annorect": [
               {
                   "x1": [625],
                   "y1": [94],
                   "x2": [681],
                   "y2": [178],
                   "score": [0.9],
                   "track_id": [0],
                   "annopoints": [
                       {
                           "point": [
                               {
                                   "id": [0],
                                   "x": [394],
                                   "y": [173],
                                   "score": [0.7],
                               },
                               { ... }
                           ]
                       }
                   ]
                },
                { ... }
           ],
       },
       { ... }
   ]
}

        2018 source in self.data:
    [
        {
            "bbox_head": [
                625,
                94,
                56,
                84
            ],
            "keypoints": [
                659,
                152,
                ...
            ],
            "track_id": 0,
            "image_id": 10003420000,
            "bbox": [
                331.64999999999998,
                105.21967275,
                427.70000000000005,
                154.62123949999997
            ],
            "scores": [],
            "category_id": 1,
            "id": 1000342000000
        },
        ...
    ]

        images contains:
        {
            "has_no_densepose": true,
            "is_labeled": false,
            "file_name": "images/val/000342_mpii_test/000093.jpg",
            "nframes": 100,
            "frame_id": 10003420093,
            "vid_id": "000342",
            "id": 10003420093
        },
        """

        annorect_by_image = defaultdict(list)
        for ann in predictions:
            kps2018 = np.array(ann['keypoints']).reshape(-1, 3)
            kps2017 = kps2018[POSETRACK_INDEX2017TO2018]

            annorect = {
                'x1': [ann['bbox'][0]],
                'y1': [ann['bbox'][1]],
                'x2': [ann['bbox'][2]],
                'y2': [ann['bbox'][3]],
                'score': [ann['score']],
                'track_id': [ann['track_id']],
                'annopoints': [
                    {
                        'point': [
                            {
                                'id': [kpi],
                                'x': [xyv[0]],
                                'y': [xyv[1]],
                                'score': [xyv[2]],
                            }
                            for kpi, xyv in enumerate(kps2017)
                            if xyv[2] > 0.0
                        ]
                    }
                ]
            }
            annorect_by_image[ann['image_file']].append(annorect)

        # image_ids = sorted(annorect_by_image.keys())
        # print(len(image_ids), len(image_names))
        # assert len(image_ids) == len(image_names)
        return [
            {
                'image': {
                    'name': image_name,
                },
                'annorect': annorect_by_image.get(image_name, []),
            }
            for image_name in image_names
        ]

    def write_predictions(self, filename, *, additional_data=None):
        output_dir = '{}-{}'.format(filename, self.output_dir_suffix)
        for annotation_file in self.predictions_by_file.keys():
            if self.output_format == '2018':
                self._write2018(output_dir, annotation_file, additional_data=additional_data)
            elif self.output_format == '2017':
                self._write2017(output_dir, annotation_file, additional_data=additional_data)
            else:
                raise NotImplementedError

        if self.ground_truth_directory is None:
            return

        # use poseval to evaluate right away
        gt_dir = os.path.dirname(self.ground_truth_directory)
        if not gt_dir.endswith('/'):
            gt_dir = gt_dir + '/'

        pred_dir = output_dir
        if not pred_dir.endswith('/'):
            pred_dir = pred_dir + '/'

        out_dir = output_dir
        if out_dir.endswith('/'):
            out_dir = out_dir[:-1]
        out_dir = out_dir + '-poseval/'

        cmd = [
            'python', '-m', 'poseval.evaluate',
            '--groundTruth', gt_dir,
            '--predictions', pred_dir,
            '--outputDir', out_dir,
            '--evalPoseTracking',
            '--evalPoseEstimation',
            '--saveEvalPerSequence',
        ]
        LOG.info('eval command: %s', ' '.join(cmd))
        subprocess.run(cmd, check=True)
        self.print_by_sequence(out_dir)

        self._written_mot_stats_file = os.path.join(out_dir, 'total_MOT_metrics.json')
        self._written_ap_stats_file = os.path.join(out_dir, 'total_AP_metrics.json')

    @staticmethod
    def print_by_sequence(out_dir):
        mot_files = glob.glob(os.path.join(out_dir, '*_MOT_metrics.json'))

        mota = {}
        for file_name in mot_files:
            with open(file_name, 'r') as f:
                d = json.load(f)
            identifier = os.path.basename(file_name).replace('_MOT_metrics.json', '')
            mota[identifier] = d.get('mota', [-1.0])[-1]

        print('sequence, mota')
        for sequence, m in sorted(mota.items(), key=lambda x: x[1]):
            print(sequence, m)
