"""Export a checkpoint as an ONNX model.

Applies onnx utilities to improve the exported model and
also tries to simplify the model with onnx-simplifier.

https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md
https://github.com/daquexian/onnx-simplifier
"""

import argparse
import shutil

import torch

import openpifpaf

try:
    import onnx
    import onnx.utils
except ImportError:
    onnx = None

try:
    import onnxsim
except ImportError:
    onnxsim = None


class GetPif(torch.nn.Module):
    def forward(self, heads):  # pylint: disable=arguments-differ
        return heads[0]


class GetPifC(torch.nn.Module):
    def forward(self, heads):  # pylint: disable=arguments-differ
        return heads[0][0]


def apply(model, outfile, verbose=True):
    # dummy_input = torch.randn(1, 3, 193, 257)
    dummy_input = torch.randn(1, 3, 97, 129)
    # model = torch.nn.Sequential(model, GetPifC())

    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.
    input_names = ['input_batch']
    # output_names = [
    #     'pif_c',
    #     'pif_r',
    #     'pif_b',
    #     'pif_s',
    #     'paf_c',
    #     'paf_r1',
    #     'paf_r2',
    #     'paf_b1',
    #     'paf_b2',
    # ]
    output_names = ['cif', 'caf']

    torch.onnx.export(
        model, dummy_input, outfile, verbose=verbose,
        input_names=input_names, output_names=output_names,
        # opset_version=10,
        # do_constant_folding=True,
        # dynamic_axes={  # TODO: gives warnings
        #     'input_batch': {0: 'batch', 2: 'height', 3: 'width'},
        #     'pif_c': {0: 'batch', 2: 'fheight', 3: 'fwidth'},
        #     'pif_r': {0: 'batch', 3: 'fheight', 4: 'fwidth'},
        #     'pif_b': {0: 'batch', 2: 'fheight', 3: 'fwidth'},
        #     'pif_s': {0: 'batch', 2: 'fheight', 3: 'fwidth'},

        #     'paf_c': {0: 'batch', 2: 'fheight', 3: 'fwidth'},
        #     'paf_r1': {0: 'batch', 3: 'fheight', 4: 'fwidth'},
        #     'paf_b1': {0: 'batch', 2: 'fheight', 3: 'fwidth'},
        #     'paf_r2': {0: 'batch', 3: 'fheight', 4: 'fwidth'},
        #     'paf_b2': {0: 'batch', 2: 'fheight', 3: 'fwidth'},
        # },
    )


def optimize(infile, outfile=None):
    if outfile is None:
        assert infile.endswith('.onnx')
        outfile = infile
        infile = infile.replace('.onnx', '.unoptimized.onnx')
        shutil.copyfile(outfile, infile)

    model = onnx.load(infile)
    optimized_model = onnx.optimizer.optimize(model)
    onnx.save(optimized_model, outfile)


def check(modelfile):
    model = onnx.load(modelfile)
    onnx.checker.check_model(model)


def polish(infile, outfile=None):
    if outfile is None:
        assert infile.endswith('.onnx')
        outfile = infile
        infile = infile.replace('.onnx', '.unpolished.onnx')
        shutil.copyfile(outfile, infile)

    model = onnx.load(infile)
    polished_model = onnx.utils.polish_model(model)
    onnx.save(polished_model, outfile)


def simplify(infile, outfile=None):
    if outfile is None:
        assert infile.endswith('.onnx')
        outfile = infile
        infile = infile.replace('.onnx', '.unsimplified.onnx')
        shutil.copyfile(outfile, infile)

    simplified_model = onnxsim.simplify(infile, check_n=0, perform_optimization=False)
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

    parser.add_argument('--checkpoint', default='resnet50')
    parser.add_argument('--outfile', default='openpifpaf-resnet50.onnx')
    parser.add_argument('--simplify', dest='simplify', default=False, action='store_true')
    parser.add_argument('--polish', dest='polish', default=False, action='store_true',
                        help='runs checker, optimizer and shape inference')
    parser.add_argument('--optimize', dest='optimize', default=False, action='store_true')
    parser.add_argument('--check', dest='check', default=False, action='store_true')
    args = parser.parse_args()

    model, _ = openpifpaf.network.factory(checkpoint=args.checkpoint)
    apply(model, args.outfile)
    if args.simplify:
        simplify(args.outfile)
    if args.optimize:
        optimize(args.outfile)
    if args.polish:
        polish(args.outfile)
    if args.check:
        check(args.outfile)


if __name__ == '__main__':
    main()
