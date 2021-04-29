"""
Convert txt files of ApolloCar3D into json file with COCO format
"""

import glob
import os
import time
from shutil import copyfile
import json
import argparse

import numpy as np
from PIL import Image

# Packages for data processing, crowd annotations and histograms
try:
    import matplotlib.pyplot as plt  # pylint: disable=import-error
except ModuleNotFoundError as err:
    if err.name != 'matplotlib':
        raise err
    plt = None
try:
    import cv2  # pylint: disable=import-error
except ModuleNotFoundError as err:
    if err.name != 'cv2':
        raise err
    cv2 = None  # pylint: disable=invalid-name

from .constants import CAR_KEYPOINTS_24, CAR_SKELETON_24,\
    CAR_KEYPOINTS_66, CAR_SKELETON_66, KPS_MAPPING
from .transforms import skeleton_mapping


def cli():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir_data', default='data-apollocar3d/train',
                        help='dataset directory')
    parser.add_argument('--dir_out', default='data-apollocar3d',
                        help='where to save annotations and files')
    parser.add_argument('--sample', action='store_true',
                        help='Whether to only process the first 50 images')
    parser.add_argument('--single_sample', action='store_true',
                        help='Whether to only process the first image')
    parser.add_argument('--split_images', action='store_true',
                        help='Whether to copy images into train val split folder')
    parser.add_argument('--histogram', action='store_true',
                        help='Whether to show keypoints histogram')
    args = parser.parse_args()
    return args


class ApolloToCoco:

    # Prepare json format
    map_sk = skeleton_mapping(KPS_MAPPING)

    sample = False
    single_sample = False
    split_images = False
    histogram = False

    def __init__(self, dir_dataset, dir_out):
        """
        :param dir_dataset: Original dataset directory
        :param dir_out: Processed dataset directory
        """

        assert os.path.isdir(dir_dataset), 'dataset directory not found'
        self.dir_dataset = dir_dataset
        self.dir_mask = os.path.join(dir_dataset, 'ignore_mask')
        assert os.path.isdir(self.dir_mask), 'crowd annotations not found: ' + self.dir_mask

        self.dir_out_im = os.path.join(dir_out, 'images')
        self.dir_out_ann = os.path.join(dir_out, 'annotations')
        os.makedirs(self.dir_out_im, exist_ok=True)
        os.makedirs(self.dir_out_ann, exist_ok=True)

        self.json_file_24 = {}
        self.json_file_66 = {}

        # Load train val split
        path_train = os.path.join(self.dir_dataset, 'split', 'train-list.txt')
        path_val = os.path.join(self.dir_dataset, 'split', 'validation-list.txt', )
        self.splits = {}
        for name, path in zip(('train', 'val'), (path_train, path_val)):
            with open(path, "r") as ff:
                lines = ff.readlines()
            self.splits[name] = [os.path.join(self.dir_dataset, 'images', line.strip())
                                 for line in lines]
            assert self.splits[name], "specified path is empty"

    def process(self):
        """Parse and process the txt dataset into a single json file compatible with coco format"""

        for phase, im_paths in self.splits.items():  # Train and Val
            cnt_images = 0
            cnt_instances = 0
            cnt_kps = [0] * 66
            self.initiate_json()  # Initiate json file at each phase

            # save only 50 images
            if self.sample:
                im_paths = im_paths[:50]
            if self.split_images:
                path_dir = (os.path.join(self.dir_out_im, phase))
                os.makedirs(path_dir, exist_ok=True)
                assert not os.listdir(path_dir), "Directory to save images is not empty. " \
                    "Remove flag --split_images ?"
            elif self.single_sample:
                im_paths = self.splits['train'][:1]
                print(f'Single sample for train/val:{im_paths}')

            for im_path in im_paths:
                im_size, im_name, im_id = self._process_image(im_path)
                cnt_images += 1

                # Process its annotations
                txt_paths = glob.glob(os.path.join(self.dir_dataset, 'keypoints', im_name,
                                                   im_name + '*.txt'))
                for txt_path in txt_paths:
                    data = np.loadtxt(txt_path, delimiter='\t', ndmin=2)
                    cnt_kps = self._process_annotation(data, txt_path, im_size, im_id, cnt_kps)
                    cnt_instances += 1

                # Split the image in a new folder
                if self.split_images:
                    dst = os.path.join(self.dir_out_im, phase, os.path.split(im_path)[-1])
                    copyfile(im_path, dst)

                # Add crowd annotations
                mask_path = os.path.join(self.dir_mask, im_name + '.jpg')
                self._process_mask(mask_path, im_id)

                # Count
                if (cnt_images % 1000) == 0:
                    text = ' and copied to new directory' if self.split_images else ''
                    print(f'Parsed {cnt_images} images' + text)

            self.save_json_files(phase)
            print(f'\nPhase:{phase}')
            print(f'Average number of keypoints labelled: {sum(cnt_kps) / cnt_instances:.1f} / 66')
            print(f'JSON files directory:  {self.dir_out_ann}')
            print(f'Saved {cnt_instances} instances over {cnt_images} images ')
            if self.histogram:
                histogram(cnt_kps)

    def save_json_files(self, phase):
        for j_file, n_kps in [(self.json_file_24, 24), (self.json_file_66, 66)]:
            name = 'apollo_keypoints_' + str(n_kps) + '_'
            if self.sample:
                name = name + 'sample_'
            elif self.single_sample:
                name = name + 'single_sample_'

            path_json = os.path.join(self.dir_out_ann, name + phase + '.json')
            with open(path_json, 'w') as outfile:
                json.dump(j_file, outfile)

    def _process_image(self, im_path):
        """Update image field in json file"""
        file_name = os.path.basename(im_path)
        im_name = os.path.splitext(file_name)[0]
        im_id = int(im_name.split(sep='_')[1])  # Numeric code in the image
        im = Image.open(im_path)
        width, height = im.size
        dict_ann = {
            'coco_url': "unknown",
            'file_name': file_name,
            'id': im_id,
            'license': 1,
            'date_captured': "unknown",
            'width': width,
            'height': height}
        self.json_file_24["images"].append(dict_ann)
        self.json_file_66["images"].append(dict_ann)
        return (width, height), im_name, im_id

    def _process_mask(self, mask_path, im_id):
        """Mask crowd annotations"""
        if cv2 is None:
            raise Exception('OpenCV')

        assert os.path.isfile(mask_path), mask_path
        im_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        blur = cv2.GaussianBlur(im_gray, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)
        contours, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for idx, mask in enumerate(contours):
            box = cv2.boundingRect(mask)
            mask_id = int(str(im_id) + '00' + str(idx))  # different id from crowds
            dict_mask = {
                'image_id': im_id,
                'category_id': 1,
                'iscrowd': 1,
                'id': mask_id,
                'area': box[2] * box[3],
                'bbox': box,
                'num_keypoints': 0,
                'keypoints': [],
                'segmentation': []}
            self.json_file_24["annotations"].append(dict_mask)
            self.json_file_66["annotations"].append(dict_mask)

    def _process_annotation(self, all_kps, txt_path, im_size, im_id, cnt_kps):
        """Process single instance"""

        # Enlarge box
        box_tight = [np.min(all_kps[:, 1]), np.min(all_kps[:, 2]),
                     np.max(all_kps[:, 1]), np.max(all_kps[:, 2])]
        w, h = box_tight[2] - box_tight[0], box_tight[3] - box_tight[1]
        x_o = max(box_tight[0] - 0.1 * w, 0)
        y_o = max(box_tight[1] - 0.1 * h, 0)
        x_i = min(box_tight[0] + 1.1 * w, im_size[0])
        y_i = min(box_tight[1] + 1.1 * h, im_size[1])
        box = [int(x_o), int(y_o), int(x_i - x_o), int(y_i - y_o)]  # (x, y, w, h)

        txt_id = os.path.splitext(txt_path.split(sep='_')[-1])[0]
        car_id = int(str(im_id) + str(int(txt_id)))  # include the specific annotation id
        kps, num = self._transform_keypoints_24(all_kps)
        self.json_file_24["annotations"].append({
            'image_id': im_id,
            'category_id': 1,
            'iscrowd': 0,
            'id': car_id,
            'area': box[2] * box[3],
            'bbox': box,
            'num_keypoints': num,
            'keypoints': kps,
            'segmentation': []})

        kps, num = self._transform_keypoints_66(all_kps)
        self.json_file_66["annotations"].append({
            'image_id': im_id,
            'category_id': 1,
            'iscrowd': 0,
            'id': car_id,
            'area': box[2] * box[3],
            'bbox': box,
            'num_keypoints': num,
            'keypoints': kps,
            'segmentation': []})
        # Stats
        for num in all_kps[:, 0]:
            cnt_kps[int(num)] += 1
        return cnt_kps

    def _transform_keypoints_24(self, kps):
        """
        24 keypoint version
        Map, filter keypoints and add visibility
        :array of [[#, x, y], ...]
        :return  [x, y, visibility, x, y, visibility, .. ]
        """
        kps_out = np.zeros((len(CAR_KEYPOINTS_24), 3))
        cnt = 0
        for kp in kps:
            n = self.map_sk[int(kp[0])]
            if n < 100:  # Filter extra keypoints
                kps_out[n, 0] = kp[1]
                kps_out[n, 1] = kp[2]
                kps_out[n, 2] = 2
                cnt += 1
        kps_out = list(kps_out.reshape((-1,)))
        return kps_out, cnt

    @staticmethod
    def _transform_keypoints_66(kps):
        """
        66 keypoint version
        Add visibility
        :array of [[#, x, y], ...]
        :return  [x, y, visibility, x, y, visibility, .. ]
        """
        kps_out = np.zeros((len(CAR_KEYPOINTS_66), 3))
        cnt = 0
        for kp in kps:
            n = int(kp[0])
            kps_out[n, 0] = kp[1]
            kps_out[n, 1] = kp[2]
            kps_out[n, 2] = 2
            cnt += 1
        kps_out = list(kps_out.reshape((-1,)))
        return kps_out, cnt

    def initiate_json(self):
        """
        Initiate Json for training and val phase for the 24 kp and the 66 kp version
        """
        for j_file, n_kp in [(self.json_file_24, 24), (self.json_file_66, 66)]:
            j_file["info"] = dict(url="https://github.com/openpifpaf/openpifpaf",
                                  date_created=time.strftime("%a, %d %b %Y %H:%M:%S +0000",
                                                             time.localtime()),
                                  description=("Conversion of ApolloCar3D dataset into MS-COCO"
                                               " format with {n_kp} keypoints"))

            skel = CAR_SKELETON_24 if n_kp == 24 else CAR_SKELETON_66
            car_kps = CAR_KEYPOINTS_24 if n_kp == 24 else CAR_KEYPOINTS_66
            j_file["categories"] = [dict(name='car',
                                         id=1,
                                         skeleton=skel,
                                         supercategory='car',
                                         keypoints=car_kps)]
            j_file["images"] = []
            j_file["annotations"] = []


def histogram(cnt_kps):
    bins = np.arange(len(cnt_kps))
    data = np.array(cnt_kps)
    plt.figure()
    plt.title("Distribution of the keypoints")
    plt.bar(bins, data)
    plt.xticks(np.arange(len(cnt_kps), step=5))
    plt.show()


def main():
    args = cli()

    # configure
    ApolloToCoco.sample = args.sample
    ApolloToCoco.single_sample = args.single_sample
    ApolloToCoco.split_images = args.split_images
    ApolloToCoco.histogram = args.histogram

    apollo_coco = ApolloToCoco(args.dir_data, args.dir_out)
    apollo_coco.process()


if __name__ == "__main__":
    main()
