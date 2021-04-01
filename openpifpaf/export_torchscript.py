"""Export a checkpoint as a TorchScript model."""

import argparse
import logging

import torch

import openpifpaf

from .export_onnx import image_size_warning

LOG = logging.getLogger(__name__)


class DummyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.ops.load_library('openpifpafcpp.so')
        self.cifhr = torch.zeros((17, 300, 400))

    def forward(self, cif_head, caf_head):
        self.cifhr[:] = 0.0
        torch.ops.my_ops.cif_hr_accumulate_op(self.cifhr, cif_head, 8, 0.1, 16, 0.0, 1.0)
        return self.cifhr


class EncoderDecoder(torch.nn.Module):
    def __init__(self, traced_encoder, decoder):
        super().__init__()
        self.traced_encoder = traced_encoder
        self.decoder = decoder

    def forward(self, x):
        cif_head_batch, caf_head_batch = self.traced_encoder(x)
        o = [self.decoder(cif_head, caf_head)
             for cif_head, caf_head in zip(cif_head_batch, caf_head_batch)]
        return o


def apply(model, outfile, *, input_w=129, input_h=97, minimum_deployment_target='iOS14'):
    image_size_warning(model.base_net.stride, input_w, input_h)

    # configure: inplace-ops are not supported
    openpifpaf.network.heads.CompositeField3.inplace_ops = False

    dummy_input = torch.randn(1, 3, input_h, input_w)
    with torch.no_grad():
        traced_encoder = torch.jit.trace(model, dummy_input)
    decoder = DummyDecoder()

    encoder_decoder = EncoderDecoder(traced_encoder, decoder)
    encoder_decoder = torch.jit.script(encoder_decoder)

    print(encoder_decoder.graph)
    torch.jit.save(encoder_decoder, outfile)


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def main():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.export_torchscript',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=openpifpaf.__version__))

    openpifpaf.network.Factory.cli(parser)

    parser.add_argument('--outfile', default='openpifpaf-shufflenetv2k16.torchscript.pt')
    parser.add_argument('--input-width', type=int, default=129)
    parser.add_argument('--input-height', type=int, default=97)
    args = parser.parse_args()

    openpifpaf.network.Factory.configure(args)

    model, _ = openpifpaf.network.Factory().factory()

    assert args.outfile.endswith('.torchscript.pt')
    apply(model, args.outfile, input_w=args.input_width, input_h=args.input_height)


if __name__ == '__main__':
    main()
