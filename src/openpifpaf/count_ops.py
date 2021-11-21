"""Export a checkpoint as an ONNX model.

Applies onnx utilities to improve the exported model and
also tries to simplify the model with onnx-simplifier.

https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md
https://github.com/daquexian/onnx-simplifier
"""

import argparse

import torch

import openpifpaf

try:
    import thop
except ImportError as e:
    raise Exception('need to install thop (pip install thop) for this script') from e


def count(model):
    dummy_input = torch.randn(1, 3, 641, 641)
    return thop.profile(model, inputs=(dummy_input, ))


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def main():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.count_ops',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=openpifpaf.__version__))

    openpifpaf.network.Factory.cli(parser)
    args = parser.parse_args()
    openpifpaf.network.Factory.configure(args)

    model, _ = openpifpaf.network.Factory().factory()

    gmacs, params = count(model)
    print('GMACs = {0:.2f}, million params = {1:.2f}'.format(gmacs / 1e9, params / 1e6))


if __name__ == '__main__':
    main()
