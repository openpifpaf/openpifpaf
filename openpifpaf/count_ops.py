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
except ImportError:
    raise Exception('need to install thop (pip install thop) for this script')


def count(checkpoint):
    dummy_input = torch.randn(1, 3, 641, 641)
    model, _ = openpifpaf.network.nets.factory(checkpoint=checkpoint)
    return thop.profile(model, inputs=(dummy_input, ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='resnet50')
    args = parser.parse_args()

    gmacs, params = count(args.checkpoint)
    print('GMACs = {0:.2f}, million params = {1:.2f}'.format(gmacs / 1e9, params / 1e6))


if __name__ == '__main__':
    main()
