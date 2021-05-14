"""
Prepare the dataset to be converted from VOC 2 COCO
This code bridges with the repository https://github.com/yukkyo/voc2coco
that requires annotation lists and train val split and annotation directories

The original datasets include images and annotations divided in 2 parts.
- part1 consists of images from VOC2012 and custom annotations
- part 2 consists of custom images and annotations
"""

import os
import glob
import argparse
import time
import json
from collections import defaultdict
from shutil import copyfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


from openpifpaf.plugins.animalpose.constants import \
    _CATEGORIES, ANIMAL_KEYPOINTS, ALTERNATIVE_NAMES, ANIMAL_SKELETON


def dataset_mappings():
    """Map the two names to 0 n-1"""
    map_n = defaultdict(lambda: 100)  # map to 100 the keypoints not used
    for i, j in zip(ANIMAL_KEYPOINTS, range(len(ANIMAL_KEYPOINTS))):
        map_n[i] = j
    for i, j in zip(ALTERNATIVE_NAMES, range(len(ALTERNATIVE_NAMES))):
        map_n[i] = j
    return map_n


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir_data', default='data-animalpose',
                        help='dataset directory')
    parser.add_argument('--dir_out', default='data-animalpose',
                        help='where to save xml annotations and output json ')
    parser.add_argument('--sample', action='store_true',
                        help='Whether to only process the first 50 images')
    args = parser.parse_args()
    return args


class VocToCoco:

    dir_splits = os.path.join('openpifpaf', 'plugins', 'animalpose', 'scripts')
    json_file = {}
    map_names = dataset_mappings()
    n_kps = len(ANIMAL_KEYPOINTS)
    cnt_kps = [0] * n_kps

    def __init__(self, dir_dataset, dir_out, args):
        """
        :param dir_dataset: Original dataset directory
        :param dir_out: Processed dataset directory
        """

        # Set directories
        self.dir_dataset = dir_dataset
        self.dir_images_1 = os.path.join(
            dir_dataset, 'TrainVal', 'VOCdevkit', 'VOC2011', 'JPEGImages')
        self.dir_images_2 = os.path.join(dir_dataset, 'animalpose_image_part2')
        self.dir_annotations_1 = os.path.join(dir_dataset, 'PASCAL2011_animal_annotation')
        self.dir_annotations_2 = os.path.join(dir_dataset, 'animalpose_anno2')
        assert os.path.isdir(self.dir_images_1), self.dir_images_1 + " not found"
        assert os.path.isdir(self.dir_images_2), self.dir_images_2 + " not found"
        assert os.path.isdir(self.dir_annotations_1), self.dir_annotations_1 + " not found"
        assert os.path.isdir(self.dir_annotations_2), self.dir_annotations_2 + " not found"
        self.dir_out_im = os.path.join(dir_out, 'images')
        self.dir_out_ann = os.path.join(dir_out, 'annotations')
        os.makedirs(self.dir_out_im, exist_ok=True)
        os.makedirs(self.dir_out_ann, exist_ok=True)
        assert not os.listdir(self.dir_out_im), "Empty image directory to avoid duplicates"
        assert not os.listdir(self.dir_out_ann), "Empty annotation directory to avoid duplicates"
        os.makedirs(os.path.join(self.dir_out_im, 'train'))
        os.makedirs(os.path.join(self.dir_out_im, 'val'))
        self.sample = args.sample

    def process(self):
        splits = self._split_train_val()
        all_xml_paths = []
        for phase in ('train', 'val'):
            metadata = splits[phase]
            if self.sample:
                metadata = metadata[:50]
            cnt_images = 0
            cnt_instances = 0
            self.cnt_kps = [0] * len(ANIMAL_KEYPOINTS)
            self.initiate_json()  # Initiate json file at each phase

            for (im_path, im_id, xml_paths) in metadata:
                self._process_image(im_path, im_id)
                cnt_images += 1
                for xml_path in xml_paths:
                    self._process_annotation(xml_path, im_id)
                    cnt_instances += 1
                    all_xml_paths.append(xml_path)

                # Split the image in a new folder
                dst = os.path.join(self.dir_out_im, phase, os.path.basename(im_path))
                copyfile(im_path, dst)

                # Count
                if (cnt_images % 1000) == 0:
                    print(f'Parsed {cnt_images} images' + ' and copied to new directory')

            # Save
            name = 'animal_keypoints_' + str(self.n_kps) + '_'
            if self.sample:
                name = name + 'sample_'

            path_json = os.path.join(self.dir_out_ann, name + phase + '.json')
            with open(path_json, 'w') as outfile:
                json.dump(self.json_file, outfile)
            print(f'Phase:{phase}')
            print(f'Average number of keypoints labelled:'
                  f' {sum(self.cnt_kps) / cnt_instances:.1f} / {self.n_kps}')
            print(f'Saved {cnt_instances} instances over {cnt_images} images ')
            print(f'JSON PATH:  {path_json}')

    def _process_image(self, im_path, im_id):
        """Update image field in json file"""
        file_name = os.path.split(im_path)[1]
        im = Image.open(im_path)
        width, height = im.size

        self.json_file["images"].append({
            'coco_url': "unknown",
            'file_name': file_name,
            'id': im_id,
            'license': 1,
            'date_captured': "unknown",
            'width': width,
            'height': height})

    def _process_annotation(self, xml_path, im_id):
        """Process single instance"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        box_obj = root.findall('visible_bounds')
        assert len(box_obj) <= 1, "multiple elements in a single annotation file not supported"

        x_min = round(float((box_obj[0].attrib['xmin']))) - 1
        width = round(float(box_obj[0].attrib['width']))
        height = round(float(box_obj[0].attrib['height']))
        try:
            y_min = round(float(box_obj[0].attrib['ymin'])) - 1
        except KeyError:
            y_min = round(float(box_obj[0].attrib['xmax'])) - 1
        box = [x_min, y_min, width, height]

        kp_obj = root.findall('keypoints')
        assert len(kp_obj) <= 1, "multiple elements in a single annotation file not supported"
        kps_list = kp_obj[0].findall('keypoint')

        kps, num = self._process_keypoint(kps_list)

        self.json_file["annotations"].append({
            'image_id': im_id,
            'category_id': 1,
            'iscrowd': 0,
            'id': im_id,
            'area': box[2] * box[3],
            'bbox': box,
            'num_keypoints': num,
            'keypoints': kps,
            'segmentation': []})

    def _process_keypoint(self, kps_list):
        """Extract single keypoint from XML"""
        cnt = 0
        kps_out = np.zeros((self.n_kps, 3))
        for kp in kps_list:
            n = self.map_names[kp.attrib['name']]
            if n < 100 and kp.attrib['visible'] == '1':
                kps_out[n, 0] = float(kp.attrib['x'])
                kps_out[n, 1] = float(kp.attrib['y'])
                kps_out[n, 2] = 2
                cnt += 1
                self.cnt_kps[n] += 1
        kps_out = list(kps_out.reshape((-1,)))
        return kps_out, cnt

    def _split_train_val(self):
        """
        Split train/val from txt files : create im_meta:
        im_path
        im_id
        xml_paths of the image
        """
        with open(os.path.join(self.dir_splits, 'train.txt'), 'r') as f:
            lists = dict(train=f.read().splitlines())
        with open(os.path.join(self.dir_splits, 'val.txt'), 'r') as f:
            lists['val'] = f.read().splitlines()
        splits = {'train': [], 'val': []}
        set1 = set(lists['train'])
        set2 = set(lists['val'])
        assert not set1.intersection(set2), "intersection not empty"
        for phase in splits:
            for name in lists[phase]:
                basename = os.path.splitext(name)[0]
                if name[:2] == '20':  # Pascal type annotations
                    date, id_str = basename.split(sep='_')
                    im_id = int(str(int(date)) + str(int(id_str)))
                    ann_folder = self.dir_annotations_1
                    im_path = os.path.join(self.dir_images_1, name)
                else:
                    idx_cat, cat = map_categories(basename[0:2])
                    im_id = int(str(999) + str(idx_cat) + basename[2:])
                    ann_folder = self.dir_annotations_2
                    im_path = os.path.join(self.dir_images_2, cat, name)

                xml_paths = find_annotations(im_path, ann_folder)
                splits[phase].append((im_path, im_id, xml_paths))

            print(f'Read {len(splits[phase])} {phase} images')
        return splits

    def initiate_json(self):
        """
        Initiate Json for training and val phase
        """
        self.json_file["info"] = dict(url="https://github.com/vita-epfl/openpifpaf",
                                      date_created=time.strftime(
                                          "%a, %d %b %Y %H:%M:%S +0000", time.localtime()),
                                      description="Animalpose dataset with MS-COCO format")

        self.json_file["categories"] = [dict(name='animal',
                                             id=1,
                                             skeleton=ANIMAL_SKELETON,
                                             supercategory='animal',
                                             keypoints=[])]
        self.json_file["images"] = []
        self.json_file["annotations"] = []


def find_annotations(im_path, ann_folder):
    xml_paths = []
    for cat in _CATEGORIES:
        root = os.path.join(ann_folder, cat, os.path.splitext(os.path.basename(im_path))[0])
        paths = glob.glob(root + '[_,.]*xml')  # Avoid duplicates of the form cow13 cow130
        if not paths:
            continue
        xml_paths.extend(paths)
    assert xml_paths, "No annotations, expected at least one"
    return xml_paths


def map_categories(cat_name):
    """It works with partial names, like do for dogs"""
    for idx, cat in enumerate(_CATEGORIES):
        if cat_name in cat:
            return idx + 1, cat   # categories starting from one
    print(cat_name)
    raise ValueError


def main():
    args = cli()
    voc_coco = VocToCoco(args.dir_data, args.dir_out, args)
    voc_coco.process()


if __name__ == "__main__":
    main()
