import argparse
import logging
import time

import numpy as np
import torch

try:
    import cv2  # pylint: disable=import-error
except ImportError:
    cv2 = None

import PIL
try:
    import PIL.ImageGrab
except ImportError:
    pass

try:
    import mss
except ImportError:
    mss = None

LOG = logging.getLogger(__name__)


# pylint: disable=abstract-method
class Stream(torch.utils.data.IterableDataset):
    horizontal_flip = None
    rotate = None
    crop = None
    scale = 1.0
    start_frame = None
    start_msec = None
    max_frames = None

    def __init__(self, source, *,
                 preprocess=None,
                 with_raw_image=True):
        super().__init__()

        self.source = source
        self.preprocess = preprocess
        self.with_raw_image = with_raw_image

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""
        group = parser.add_argument_group('Stream')
        group.add_argument('--horizontal-flip', default=False, action='store_true',
                           help='mirror input image')
        group.add_argument('--scale', default=1.0, type=float,
                           help='input image scale factor')
        group.add_argument('--start-frame', type=int, default=None,
                           help='start frame')
        group.add_argument('--start-msec', type=float, default=None,
                           help='start millisecond')
        group.add_argument('--crop', type=int, nargs=4, default=None,
                           help='left top right bottom')
        group.add_argument('--rotate', default=None, choices=('left', 'right', '180'),
                           help='rotate')
        group.add_argument('--max-frames', type=int, default=None,
                           help='max frames')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""
        cls.horizontal_flip = args.horizontal_flip
        cls.scale = args.scale
        cls.start_frame = args.start_frame
        cls.start_msec = args.start_msec
        cls.crop = args.crop
        cls.rotate = args.rotate
        cls.max_frames = args.max_frames

    # pylint: disable=unsubscriptable-object
    def preprocessing(self, image):
        if self.scale != 1.0:
            image = cv2.resize(image, None, fx=self.scale, fy=self.scale)
            LOG.debug('resized image size: %s', image.shape)
        if self.horizontal_flip:
            image = image[:, ::-1]
        if self.crop:
            if self.crop[0]:
                image = image[:, self.crop[0]:]
            if self.crop[1]:
                image = image[self.crop[1]:, :]
            if self.crop[2]:
                image = image[:, :-self.crop[2]]
            if self.crop[3]:
                image = image[:-self.crop[3], :]
        if self.rotate == 'left':
            image = np.swapaxes(image, 0, 1)
            image = np.flip(image, axis=0)
        elif self.rotate == 'right':
            image = np.swapaxes(image, 0, 1)
            image = np.flip(image, axis=1)
        elif self.rotate == '180':
            image = np.flip(image, axis=0)
            image = np.flip(image, axis=1)

        image_pil = PIL.Image.fromarray(np.ascontiguousarray(image))
        meta = {
            'hflip': False,
            'offset': np.array([0.0, 0.0]),
            'scale': np.array([1.0, 1.0]),
            'valid_area': np.array([0.0, 0.0, image_pil.size[0], image_pil.size[1]]),
        }
        processed_image, anns, meta = self.preprocess(image_pil, [], meta)
        return image, processed_image, anns, meta

    # pylint: disable=too-many-branches
    def __iter__(self):
        if self.source == 'screen':
            capture = 'screen'
            if mss is None:
                print('!!!!!!!!!!! install mss (pip install mss) for faster screen grabs')
        else:
            capture = cv2.VideoCapture(self.source)
            if self.start_frame:
                capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            if self.start_msec:
                capture.set(cv2.CAP_PROP_POS_MSEC, self.start_msec)

        frame_start = 0 if not self.start_frame else self.start_frame
        frame_i = frame_start
        while True:
            frame_i += 1
            if self.max_frames and frame_i - frame_start > self.max_frames:
                LOG.info('reached max frames %d', self.max_frames)
                break

            if capture == 'screen':
                if mss is None:
                    image = np.asarray(PIL.ImageGrab.grab().convert('RGB'))
                else:
                    with mss.mss() as sct:
                        monitor = sct.monitors[1]
                        image = np.asarray(sct.grab(monitor))[:, :, 2::-1]
            else:
                _, image = capture.read()
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if image is None:
                LOG.info('no more images captured')
                break

            start_preprocess = time.perf_counter()
            image, processed_image, anns, meta = self.preprocessing(image)
            meta['frame_i'] = frame_i
            meta['preprocessing_s'] = time.perf_counter() - start_preprocess

            if self.with_raw_image:
                yield image, processed_image, anns, meta
            else:
                yield processed_image, anns, meta
