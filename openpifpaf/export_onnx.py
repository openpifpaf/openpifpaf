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


# monkey patch
class MonkeyPatches:
    def __init__(self):
        self.original_compositehead_patched_forward = \
            openpifpaf.network.heads.CompositeField.forward

    def apply(self):
        openpifpaf.network.heads.CompositeField.forward = \
            self.compositehead_patched_forward

    def revert(self):
        openpifpaf.network.heads.CompositeField.forward = \
            self.original_compositehead_patched_forward

    @staticmethod
    def compositehead_patched_forward(self_, x):
        x = self_.dropout(x)

        # classification
        classes_x = [class_conv(x) for class_conv in self_.class_convs]
        if not self_.training:
            classes_x = [torch.sigmoid(class_x) for class_x in classes_x]

        # regressions
        regs_x = [reg_conv(x) * self_.dilation for reg_conv in self_.reg_convs]
        regs_x_spread = [reg_spread(x) for reg_spread in self_.reg_spreads]
        # regs_x_spread = [torch.nn.functional.leaky_relu(x + 3.0) - 3.0 for x in regs_x_spread]
        # problem for ONNX is the "- 3.0"
        regs_x_spread = [torch.clamp(x, min=-3.0) for x in regs_x_spread]

        # scale
        scales_x = [scale_conv(x) for scale_conv in self_.scale_convs]
        scales_x = [torch.nn.functional.relu(scale_x) for scale_x in scales_x]

        for _ in range(self_._quad):  # pylint: disable=protected-access
            classes_x = [self_.dequad_op(class_x)[:, :, :-1, :-1]
                         for class_x in classes_x]
            regs_x = [self_.dequad_op(reg_x)[:, :, :-1, :-1]
                      for reg_x in regs_x]
            regs_x_spread = [self_.dequad_op(reg_x_spread)[:, :, :-1, :-1]
                             for reg_x_spread in regs_x_spread]
            scales_x = [self_.dequad_op(scale_x)[:, :, :-1, :-1]
                        for scale_x in scales_x]

        # reshape regressions:
        # Accessing the shape of .data instead of the shape of reg_x saves
        # nodes in the ONNX graph.
        regs_x = [
            reg_x.reshape(reg_x.data.shape[0],
                          reg_x.data.shape[1] // 2,
                          2,
                          reg_x.data.shape[2],
                          reg_x.data.shape[3])
            for reg_x in regs_x
        ]

        return classes_x + regs_x + regs_x_spread + scales_x


class GetPif(torch.nn.Module):
    def forward(self, heads):  # pylint: disable=arguments-differ
        return heads[0]


class GetPifC(torch.nn.Module):
    def forward(self, heads):  # pylint: disable=arguments-differ
        return heads[0][0]


def apply(checkpoint, outfile, verbose=True):
    monkey_patches = MonkeyPatches()
    monkey_patches.apply()

    # dummy_input = torch.randn(1, 3, 193, 257)
    dummy_input = torch.randn(1, 3, 97, 129)
    model, _ = openpifpaf.network.nets.factory(checkpoint=checkpoint)
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
    output_names = [
        'pif_c',
        'pif_r',
        'pif_b',
        'pif_s',
        'paf_c',
        'paf_r1',
        'paf_r2',
        'paf_b1',
        'paf_b2',
    ]

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

    monkey_patches.revert()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='resnet50')
    parser.add_argument('--outfile', default='openpifpaf-resnet50.onnx')
    parser.add_argument('--simplify', dest='simplify', default=False, action='store_true')
    parser.add_argument('--no-polish', dest='polish', default=True, action='store_false',
                        help='runs checker, optimizer and shape inference')
    parser.add_argument('--optimize', dest='optimize', default=False, action='store_true')
    parser.add_argument('--no-check', dest='check', default=True, action='store_false')
    args = parser.parse_args()

    apply(args.checkpoint, args.outfile)
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
