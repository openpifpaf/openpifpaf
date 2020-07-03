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


def apply(model, outfile, input_w=129, input_h=97, verbose=True):
    # dummy_input = torch.randn(1, 3, 193, 257)
    dummy_input = torch.randn(1, 3, input_h, input_w)
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
        keep_initializers_as_inputs=True,
        # opset_version=10,
        do_constant_folding=True,
        export_params=True,
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


def apply_components(model, outfile, input_w=129, input_h=97, verbose=True):
    output_files = []
    torch.onnx.export(
        model.base_net,
        torch.randn(1, 3, input_h, input_w),
        outfile + '.base.onnx', verbose=verbose,
        input_names=['input_batch'], output_names=['features'],
        keep_initializers_as_inputs=True,
        # opset_version=10,
        do_constant_folding=True,
        export_params=True,
    )
    output_files.append(outfile + '.base.onnx')

    for head_i, head in enumerate(model.head_nets):
        torch.onnx.export(
            head,
            torch.randn(
                1,
                model.base_net.out_features,
                (input_h - 1) // 16 + 1,
                (input_w - 1) // 16 + 1,
            ),
            outfile + '.head{}.onnx'.format(head_i), verbose=verbose,
            input_names=['features'], output_names=['head{}'.format(head_i)],
            keep_initializers_as_inputs=True,
            # opset_version=10,
            do_constant_folding=True,
            export_params=True,
        )
        output_files.append(outfile + '.head{}.onnx'.format(head_i))

    return output_files


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

    simplified_model, check_ok = onnxsim.simplify(infile, check_n=3, perform_optimization=False)
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

    parser.add_argument('--checkpoint', default='resnet50')
    parser.add_argument('--outfile', default='openpifpaf-resnet50.onnx')
    parser.add_argument('--simplify', dest='simplify', default=False, action='store_true')
    parser.add_argument('--polish', dest='polish', default=False, action='store_true',
                        help='runs checker, optimizer and shape inference')
    parser.add_argument('--optimize', dest='optimize', default=False, action='store_true')
    parser.add_argument('--check', dest='check', default=False, action='store_true')
    parser.add_argument('--input-width', type=int, default=129)
    parser.add_argument('--input-height', type=int, default=97)
    parser.add_argument('--model-components', default=False, action='store_true')
    args = parser.parse_args()

    model, _ = openpifpaf.network.factory(checkpoint=args.checkpoint)

    if args.model_components:
        out_files = apply_components(
            model, args.outfile,
            input_w=args.input_width, input_h=args.input_height)
    else:
        apply(model, args.outfile, input_w=args.input_width, input_h=args.input_height)
        out_files = [args.outfile]

    for out_name in out_files:
        if args.simplify:
            simplify(out_name)
        if args.optimize:
            optimize(out_name)
        if args.polish:
            polish(out_name)
        if args.check:
            check(out_name)


if __name__ == '__main__':
    main()
