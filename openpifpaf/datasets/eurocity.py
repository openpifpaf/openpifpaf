import os
import copy
import logging
import json
import glob
import numpy as np
import torch.utils.data
import torchvision
import json
from PIL import Image
from .. import transforms, utils

LOG = logging.getLogger(__name__)

class EuroCity(torch.utils.data.Dataset):

    train_image_dir = "./data/ECP/{}/img/train"
    val_image_dir = "./data/ECP/{}/img/val"
    train_annotations = "./data/ECP/{}/labels/train"
    val_annotations = "./data/ECP/{}/labels/val"

    test_path = {'val': "./data/ECP/{}/img/val", 'test-challenge': "./data/ECP/{}/img/test"}
    categories = ['pedestrian', 'rider', 'scooter', 'motorbike', 'bicycle', 'buggy', 'wheelchair', 'tricycle']
    def __init__(self, root, annFile, *, time=('day', 'night'), target_transforms=None,
                 n_images=None, preprocess=None, all_images=False, rider_vehicles=False):
        self.annFile = annFile
        self.root = root
        if not isinstance(time, tuple):
            imgFolder = root.format(time)
            self.gt_files = glob.glob(imgFolder + '/*/*' + '.png')
        else:
            self.gt_files = []
            for indv_time in time:
                imgFolder = root.format(indv_time)
                self.gt_files.extend(glob.glob(imgFolder + '/*/*' + '.png'))

        if n_images:
            self.gt_files = self.gt_files[:n_images]
        self.gt_files.sort()
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.target_transforms = target_transforms

        self.cat_ids = ["pedestrian", "rider"]
        if rider_vehicles:
            self.cat_ids.extend(['scooter', 'motorbike', 'bicycle','buggy', 'wheelchair', 'tricycle'])
        print("Number of classes: {}".format(len(self.cat_ids)))

    def _get_gt_frame(self, gt_dict):
        if gt_dict['identity'] == 'frame':
            pass
        elif '@converter' in gt_dict:
            gt_dict = gt_dict['children'][0]['children'][0]
        elif gt_dict['identity'] == 'seqlist':
            gt_dict = gt_dict['children']['children']

        # check if json layout is corrupt
        assert gt_dict['identity'] == 'frame'
        return gt_dict

    def _prepare_ecp_gt(self, gt):
        def translate_ecp_pose_to_image_coordinates(angle):
            angle = angle + 90.0

            # map to interval [0, 360)
            angle = angle % 360

            if angle > 180:
                # map to interval (-180, 180]
                angle -= 360.0

            return np.deg2rad(angle)

        orient = None
        if gt['identity'] == 'rider':
            if len(gt['children']) > 0:  # vehicle is annotated
                for cgt in gt['children']:
                    if cgt['identity'] in ['bicycle', 'buggy', 'motorbike', 'scooter', 'tricycle',
                                           'wheelchair']:
                        orient = cgt.get('Orient', None) or cgt.get('orient', None)
        else:
            orient = gt.get('Orient', None) or gt.get('orient', None)

        if orient:
            gt['orient'] = translate_ecp_pose_to_image_coordinates(orient)
            gt.pop('Orient', None)

    def __getitem__(self, index):
        img_path = self.gt_files[index]

        gt_file = img_path.replace('img', 'labels').replace('.png', '.json')
        with open(os.path.join(img_path), 'rb') as f:
            image = Image.open(f).convert('RGB')

        initial_size = image.size
        meta_init = {
            'dataset_index': index,
            'image_id': index,
            'file_dir': img_path,
            'file_name': os.path.splitext(os.path.basename(img_path))[0],
            'mode':img_path.split("/")[-3],
            'time': img_path.split("/")[-5]
        }
        anns = []
        if self.annFile:
            gt_fn = os.path.basename(gt_file)

            with open(gt_file, 'rb') as f:
                gt = json.load(f)

            gt_frame = self._get_gt_frame(gt)
            for gt in gt_frame['children']:
                self._prepare_ecp_gt(gt)
                x = int(gt['x0'])
                y = int(gt['y0'])
                w = int(gt['x1']) - x
                h = int(gt['y1']) - y
                if gt['identity'] in self.cat_ids:
                    anns.append({
                        'image_id': index,
                        'category_id': gt['identity'],
                        'bbox': [x, y, w, h],
                        "area": w*h,
                        "keypoints":[x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2],
                        "iscrowd": 0,
                        "segmentation":[],
                    })
                else:
                    anns.append({
                        'image_id': index,
                        'category_id': -1,
                        'bbox': [x, y, w, h],
                        "area": w*h,
                        "keypoints":[x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2],
                        "iscrowd": 1,
                        "segmentation":[],
                    })
            # preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns, None)
        meta.update(meta_init)

        # transform image

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)
        LOG.debug(meta)
        # transform targets
        if self.target_transforms is not None:
            anns = [t(image, anns, meta) for t in self.target_transforms]

        return image, anns, meta
    def __len__(self):
        return len(self.gt_files)

    def write_evaluations(self, eval_class, path, total_time):
        for filename in eval_class.dict_folder.keys():
            dict_singleFrame = {}
            dict_singleFrame["identity"] = "frame"
            dict_singleFrame["children"] = []
            for instance in eval_class.dict_folder[filename]:
                instance = instance.split(',')
                time = instance[7]
                mode = instance[6]
                if float(instance[4])<=0:
                    continue
                dict_singleFrame["children"].append({"score": float(instance[4]),
                "x0": float(instance[0]),
                "x1": float(instance[0]) + float(instance[2]),
                "y0": float(instance[1]),
                "y1": float(instance[1]) + float(instance[3]),
                "orient": -10.0,
                "identity": self.cat_ids[int(instance[5])]})
            utils.mkdir_if_missing(os.path.join(path, time, mode))
            with open(os.path.join(path, time, mode, filename+".json"), "w") as file:
                json.dump(dict_singleFrame, file)
        n_images = len(eval_class.image_ids)

        print('n images = {}'.format(n_images))
        print('decoder time = {:.1f}s ({:.0f}ms / image)'
              ''.format(eval_class.decoder_time, 1000 * eval_class.decoder_time / n_images))
        print('total time = {:.1f}s ({:.0f}ms / image)'
              ''.format(total_time, 1000 * total_time / n_images))
