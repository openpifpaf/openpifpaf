"""Video demo application.

Use --scale=0.2 to reduce the input image size to 20%.
Use --json-output for headless processing.

Example commands:
    python3 -m pifpaf.video --source=0  # default webcam
    python3 -m pifpaf.video --source=1  # another webcam

    # streaming source
    python3 -m pifpaf.video --source=http://127.0.0.1:8080/video

    # file system source (any valid OpenCV source)
    python3 -m pifpaf.video --source=docs/coco/000000081988.jpg

Trouble shooting:
* MacOSX: try to prefix the command with "MPLBACKEND=MACOSX".
"""


import argparse
import json
import logging
import os
import time

import numpy as np
import PIL
try:
    import PIL.ImageGrab
except ImportError:
    pass
import torch

import cv2  # pylint: disable=import-error
from . import decoder, logger, network, plugin, show, transforms, visualizer, __version__

try:
    import mss
except ImportError:
    mss = None

LOG = logging.getLogger(__name__)


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def cli():  # pylint: disable=too-many-statements,too-many-branches
    plugin.register()

    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.video',
        usage='%(prog)s [options]',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    network.Factory.cli(parser)
    decoder.cli(parser)
    logger.cli(parser)
    show.cli(parser)
    visualizer.cli(parser)

    parser.add_argument('--source', default='0',
                        help=('OpenCV source url. Integer for webcams. '
                              'Or ipwebcam urls (rtsp/rtmp). '
                              'Use "screen" for screen grabs.'))
    parser.add_argument('--video-output', default=None, nargs='?', const=True,
                        help='video output file')
    parser.add_argument('--json-output', default=None, nargs='?', const=True,
                        help='json output file')
    parser.add_argument('--horizontal-flip', default=False, action='store_true')
    parser.add_argument('--long-edge', default=None, type=int,
                        help='long edge of input images')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--scale', default=1.0, type=float,
                        help='input image scale factor')
    parser.add_argument('--start-frame', type=int, default=None)
    parser.add_argument('--start-msec', type=float, default=None)
    parser.add_argument('--skip-frames', type=int, default=1)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--crop', type=int, nargs=4, default=None, help='left top right bottom')
    parser.add_argument('--rotate', default=None, choices=('left', 'right', '180'))
    parser.add_argument('--separate-debug-ax', default=False, action='store_true')
    args = parser.parse_args()

    logger.configure(args, LOG)  # logger first

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.info('neural network device: %s (CUDA available: %s, count: %d)',
             args.device, torch.cuda.is_available(), torch.cuda.device_count())

    decoder.configure(args)
    network.Factory.configure(args)
    show.configure(args)
    visualizer.configure(args)

    # check whether source should be an int
    if len(args.source) == 1:
        args.source = int(args.source)

    # standard filenames
    if args.video_output is True:
        args.video_output = '{}.openpifpaf.mp4'.format(args.source)
        if os.path.exists(args.video_output):
            os.remove(args.video_output)
    assert args.video_output is None or not os.path.exists(args.video_output)
    if args.json_output is True:
        args.json_output = '{}.openpifpaf.json'.format(args.source)
        if os.path.exists(args.json_output):
            os.remove(args.json_output)
    assert args.json_output is None or not os.path.exists(args.json_output)

    return args


def processor_factory(args):
    model, _ = network.Factory().factory()
    model = model.to(args.device)

    head_metas = [hn.meta for hn in model.head_nets]
    processor = decoder.factory(head_metas)

    return processor, model


# pylint: disable=too-many-branches,too-many-statements
def main():
    args = cli()
    processor, model = processor_factory(args)

    # assemble preprocessing transforms
    rescale_t = None
    if args.long_edge is not None:
        rescale_t = transforms.RescaleAbsolute(args.long_edge, fast=True)
    preprocess = transforms.Compose([
        transforms.NormalizeAnnotations(),
        rescale_t,
        transforms.CenterPadTight(16),
        transforms.EVAL_TRANSFORM,
    ])

    # create keypoint painter
    annotation_painter = show.AnnotationPainter()

    if args.source == 'screen':
        capture = 'screen'
        if mss is None:
            print('!!!!!!!!!!! install mss (pip install mss) for faster screen grabs')
    else:
        capture = cv2.VideoCapture(args.source)
        if args.start_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        if args.start_msec:
            capture.set(cv2.CAP_PROP_POS_MSEC, args.start_msec)

    animation = show.AnimationFrame(
        video_output=args.video_output,
        second_visual=args.separate_debug_ax,
    )
    last_loop = time.time()
    for frame_i, (ax, ax_second) in enumerate(animation.iter()):
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

        if frame_i % args.skip_frames != 0:
            animation.skip_frame()
            continue

        start = time.time()
        if args.scale != 1.0:
            image = cv2.resize(image, None, fx=args.scale, fy=args.scale)
            LOG.debug('resized image size: %s', image.shape)
        if args.horizontal_flip:
            image = image[:, ::-1]
        if args.crop:
            if args.crop[0]:
                image = image[:, args.crop[0]:]
            if args.crop[1]:
                image = image[args.crop[1]:, :]
            if args.crop[2]:
                image = image[:, :-args.crop[2]]
            if args.crop[3]:
                image = image[:-args.crop[3], :]
        if args.rotate == 'left':
            image = np.swapaxes(image, 0, 1)
            image = np.flip(image, axis=0)
        elif args.rotate == 'right':
            image = np.swapaxes(image, 0, 1)
            image = np.flip(image, axis=1)
        elif args.rotate == '180':
            image = np.flip(image, axis=0)
            image = np.flip(image, axis=1)

        if ax is None:
            ax, ax_second = animation.frame_init(image)

        image_pil = PIL.Image.fromarray(image)
        meta = {
            'hflip': False,
            'offset': np.array([0.0, 0.0]),
            'scale': np.array([1.0, 1.0]),
            'valid_area': np.array([0.0, 0.0, image_pil.size[0], image_pil.size[1]]),
        }
        processed_image, _, meta = preprocess(image_pil, [], meta)
        visualizer.Base.image(image, meta=meta)
        visualizer.Base.processed_image(processed_image)
        visualizer.Base.common_ax = ax_second if args.separate_debug_ax else ax
        preprocessing_time = time.time() - start

        preds = processor.batch(model, torch.unsqueeze(processed_image, 0), device=args.device)[0]

        start_post = time.perf_counter()
        preds = [ann.inverse_transform(meta) for ann in preds]

        if args.json_output:
            with open(args.json_output, 'a+') as f:
                json.dump({
                    'frame': frame_i,
                    'predictions': [ann.json_data() for ann in preds]
                }, f, separators=(',', ':'))
                f.write('\n')
        if (not args.json_output or args.video_output) \
           and (args.separate_debug_ax or not (args.debug or args.debug_indices)):
            ax.imshow(image)
            annotation_painter.annotations(ax, preds)
        postprocessing_time = time.perf_counter() - start_post
        LOG.info('frame %d, loop time = %.3fs (pre = %.3fs, post = %.3fs), FPS = %.3f',
                 frame_i,
                 time.time() - last_loop,
                 preprocessing_time,
                 postprocessing_time,
                 1.0 / (time.time() - last_loop))
        last_loop = time.time()

        if args.max_frames and frame_i >= args.max_frames:
            break


if __name__ == '__main__':
    main()
