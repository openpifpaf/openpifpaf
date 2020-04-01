"""Video demo application.

Example commands:
    python3 -m pifpaf.video  # usbcam or webcam 0
    python3 -m pifpaf.video --source=1  # usbcam or webcam 1

    # streaming source
    python3 -m pifpaf.video --source=http://128.179.139.21:8080/video

    # file system source (any valid OpenCV source)
    python3 -m pifpaf.video --source=docs/coco/000000081988.jpg

Trouble shooting:
* MacOSX: try to prefix the command with "MPLBACKEND=MACOSX".
"""


import argparse
import json
import logging
import time

import PIL
import torch

import cv2  # pylint: disable=import-error
from .network import nets
from . import decoder, show, transforms, visualizer

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

LOG = logging.getLogger(__name__)


class Animation(object):
    def __init__(self, processor, keypoint_painter):
        self.processor = processor
        self.keypoint_painter = keypoint_painter

        if plt is None:
            LOG.error('matplotlib is not installed')

    @staticmethod
    def clean_axis(ax):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.cla()
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    def __call__(self, first_image, fig_width=4.0, **kwargs):
        if plt is None:
            while True:
                image, all_fields = yield
            return

        if 'figsize' not in kwargs:
            kwargs['figsize'] = (fig_width, fig_width * first_image.shape[0] / first_image.shape[1])

        fig = plt.figure(**kwargs)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        ax.set_xlim(0, first_image.shape[1])
        ax.set_ylim(first_image.shape[0], 0)
        text = 'OpenPifPaf'
        ax.text(1, 1, text,
                fontsize=10, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5, linewidth=0))
        fig.add_axes(ax)
        ax.imshow(first_image)
        fig.show()

        while True:
            image, all_fields = yield
            annotations = self.processor.annotations(all_fields)

            draw_start = time.time()
            self.clean_axis(ax)
            ax.imshow(image)
            self.keypoint_painter.annotations(ax, annotations)
            fig.canvas.draw()
            LOG.debug('draw %.3fs', time.time() - draw_start)
            plt.pause(0.01)

        plt.close(fig)


class JsonOutput(object):
    def __init__(self, processor, output):
        self.processor = processor
        self.output = output

    def __call__(self, first_image, **kwargs):
        while True:
            _, all_fields = yield
            pred = self.processor.annotations(all_fields)

            with open(self.output, 'a+') as f:
                json.dump([ann.json_data() for ann in pred], f)
                f.write('\n')


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    nets.cli(parser)
    decoder.cli(parser, force_complete_pose=False, instance_threshold=0.1, seed_threshold=0.5)
    show.cli(parser)
    parser.add_argument('--no-colored-connections',
                        dest='colored_connections', default=True, action='store_false',
                        help='do not use colored connections to draw poses')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--source', default='0',
                        help='OpenCV source url. Integer for webcams. Or ipwebcam streams.')
    parser.add_argument('--scale', default=0.1, type=float,
                        help='input image scale factor')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--json-output', help='store annotations in a json file')
    group = parser.add_argument_group('logging')
    group.add_argument('-q', '--quiet', default=False, action='store_true',
                       help='only show warning messages or above')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    args = parser.parse_args()

    # configure logging
    log_level = logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig()
    logging.getLogger('openpifpaf').setLevel(log_level)
    LOG.setLevel(log_level)

    show.configure(args)

    # check whether source should be an int
    if len(args.source) == 1:
        args.source = int(args.source)

    # add args.device
    args.device = torch.device('cpu')
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')

    return args


def main():
    args = cli()

    # create keypoint painter
    if not args.json_output:
        if args.colored_connections:
            keypoint_painter = show.KeypointPainter(color_connections=True, linewidth=6)
        else:
            keypoint_painter = show.KeypointPainter()

    # load model
    model, _ = nets.factory_from_args(args)
    model = model.to(args.device)
    processor = decoder.factory_from_args(args, model)

    last_loop = time.time()
    capture = cv2.VideoCapture(args.source)

    animation = None
    frame_i = 0
    while True:
        frame_i += 1
        _, image = capture.read()
        if image is None:
            LOG.info('no more images captured')
            break

        if frame_i < args.start_frame:
            continue

        if args.scale != 1.0:
            image = cv2.resize(image, None, fx=args.scale, fy=args.scale)
            LOG.debug('resized image size: %s', image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if animation is None:
            if args.json_output:
                animation = JsonOutput(processor, args.json_output)(image)
                animation.send(None)
            else:
                animation = Animation(processor, keypoint_painter)(image)
                animation.send(None)

        start = time.time()
        image_pil = PIL.Image.fromarray(image)
        processed_image_cpu, _, __ = transforms.EVAL_TRANSFORM(image_pil, [], None)
        visualizer.BaseVisualizer.image(image_pil)
        visualizer.BaseVisualizer.processed_image(processed_image_cpu)
        processed_image = processed_image_cpu.contiguous().to(args.device, non_blocking=True)
        LOG.debug('preprocessing time %.3fs', time.time() - start)

        fields = processor.fields(torch.unsqueeze(processed_image, 0))[0]
        animation.send((image, fields))

        LOG.info('frame %d, loop time = %.3fs, FPS = %.3f',
                 frame_i,
                 time.time() - last_loop,
                 1.0 / (time.time() - last_loop))
        last_loop = time.time()

        if args.max_frames and frame_i >= args.start_frame + args.max_frames:
            break


if __name__ == '__main__':
    main()
