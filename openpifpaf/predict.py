"""Predict poses for given images."""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from .network import nets
from . import decoder, show, transforms


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    nets.cli(parser)
    decoder.cli(parser, instance_threshold=0.05)
    parser.add_argument('images', nargs='*',
                        help='input images')
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('-o', '--output-directory',
                        help=('Output directory. When using this option, make '
                              'sure input images have distinct file names.'))
    parser.add_argument('--show', default=False, action='store_true',
                        help='show image of output overlay')
    parser.add_argument('--no-image', dest='image', default=True, action='store_false',
                        help='do not create images')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--figure-width', default=10.0, type=float,
                        help='figure width')
    parser.add_argument('--dpi-factor', default=1.0, type=float,
                        help='increase dpi of output image by this factor')
    args = parser.parse_args()

    # glob
    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")

    # add args.device
    args.device = torch.device('cpu')
    pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        pin_memory = True

    # load model
    model, _ = nets.factory(args)
    model = model.to(args.device)
    processors = decoder.factory(args, model)

    for image_i, image_path in enumerate(args.images):
        if args.output_directory is None:
            output_path = image_path
        else:
            file_name = os.path.basename(image_path)
            output_path = os.path.join(args.output_directory, file_name)
        print('image', image_i, image_path, output_path)
        image = show.load_image(image_path)

        start = time.time()
        processed_image_cpu = transforms.image_transform(image.copy())
        processed_image = processed_image_cpu.contiguous().to(args.device, non_blocking=True)
        print('preprocessing time', time.time() - start)

        processors[0].set_cpu_image(processed_image_cpu)
        all_fields = processors[0].fields(processed_image.float())
        for processor in processors:
            keypoint_sets, scores = processor.keypoint_sets(all_fields)

            with open(output_path + '.pifpaf.json', 'w') as f:
                json.dump([
                    {'keypoints': np.around(kps, 1).reshape(-1).tolist(),
                     'bbox': [np.min(kps[:, 0]), np.min(kps[:, 1]),
                              np.max(kps[:, 0]), np.max(kps[:, 1])]}
                    for kps in keypoint_sets
                ], f)

            if args.image:
                with show.image_canvas(image,
                                       output_path + '.keypoints.png',
                                       show=args.show,
                                       fig_width=args.figure_width,
                                       dpi_factor=args.dpi_factor) as ax:
                    show.white_screen(ax, alpha=0.5)
                    show.keypoints(ax, keypoint_sets, show_box=False)
                with show.image_canvas(image,
                                       output_path + '.skeleton.png',
                                       show=args.show,
                                       fig_width=args.figure_width,
                                       dpi_factor=args.dpi_factor) as ax:
                    show.keypoints(ax, keypoint_sets,
                                   scores=scores, show_box=False,
                                   markersize=1,
                                   color_connections=True, linewidth=6)


if __name__ == '__main__':
    main()
