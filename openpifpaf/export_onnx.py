"""Export a checkpoint as an ONNX model.

Applies onnx utilities to improve the exported model and
also tries to simplify the model with onnx-simplifier.

https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md
https://github.com/daquexian/onnx-simplifier
"""

import argparse
import logging
import shutil

import torch

import openpifpaf

try:
    import onnx
except ImportError:
    onnx = None

try:
    import onnxsim
except ImportError:
    onnxsim = None

LOG = logging.getLogger(__name__)


def image_size_warning(basenet_stride, input_w, input_h):
    if input_w % basenet_stride != 1:
        LOG.warning(
            'input width (%d) should be a multiple of basenet '
            'stride (%d) + 1: closest are %d and %d',
            input_w, basenet_stride,
            (input_w - 1) // basenet_stride * basenet_stride + 1,
            ((input_w - 1) // basenet_stride + 1) * basenet_stride + 1,
        )

    if input_h % basenet_stride != 1:
        LOG.warning(
            'input height (%d) should be a multiple of basenet '
            'stride (%d) + 1: closest are %d and %d',
            input_h, basenet_stride,
            (input_h - 1) // basenet_stride * basenet_stride + 1,
            ((input_h - 1) // basenet_stride + 1) * basenet_stride + 1,
        )


def apply(model, outfile, verbose=True, input_w=129, input_h=97):
    image_size_warning(model.base_net.stride, input_w, input_h)

    # configure
    openpifpaf.network.heads.CompositeField3.inplace_ops = False
    openpifpaf.network.heads.CompositeField4.inplace_ops = False

    dummy_input = torch.randn(1, 3, input_h, input_w)

    torch.onnx.export(
        model, dummy_input, outfile, verbose=verbose,
        input_names=['input_batch'], output_names=['cif', 'caf'],
        # keep_initializers_as_inputs=True,
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes={
            'input_batch': {0: 'dynamic'},
            'cif': {0: 'dynamic'},
            'caf': {0: 'dynamic'},
        },
    )


def check(modelfile):
    model = onnx.load(modelfile)
    onnx.checker.check_model(model)


def simplify(infile, outfile=None, input_w=129, input_h=97):
    if outfile is None:
        assert infile.endswith('.onnx')
        outfile = infile
        infile = infile.replace('.onnx', '.unsimplified.onnx')
        shutil.copyfile(outfile, infile)

    simplified_model, check_ok = onnxsim.simplify(
        infile,
        input_shapes={'input_batch': [1, 3, input_h, input_w]},
        check_n=3,
        perform_optimization=False,
    )
    assert check_ok
    onnx.save(simplified_model, outfile)


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def main():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.export_onnx',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=openpifpaf.__version__))

    openpifpaf.network.Factory.cli(parser)

    parser.add_argument('--outfile', default='openpifpaf-resnet50.onnx')
    parser.add_argument('--simplify', dest='simplify', default=False, action='store_true')
    parser.add_argument('--check', dest='check', default=False, action='store_true')
    parser.add_argument('--input-width', type=int, default=129)
    parser.add_argument('--input-height', type=int, default=97)
    args = parser.parse_args()

    openpifpaf.network.Factory.configure(args)

    model, _ = openpifpaf.network.Factory().factory()

    apply(model, args.outfile, input_w=args.input_width, input_h=args.input_height)
    if args.simplify:
        simplify(args.outfile, input_w=args.input_width, input_h=args.input_height)
    if args.check:
        check(args.outfile)


if __name__ == '__main__':
    main()
