"""Export a checkpoint as a TorchScript model."""

import argparse
import logging

import torch

import openpifpaf

from .export_onnx import image_size_warning

LOG = logging.getLogger(__name__)


class DecoderModule(torch.nn.Module):
    def __init__(self, cif_meta, caf_meta):
        super().__init__()

        self.n_keypoints = len(cif_meta.keypoints)
        self.skeleton = torch.LongTensor(caf_meta.skeleton) - 1

    def forward(self, cif_field, caf_field):
        initial_annotations_t = torch.empty((0, self.n_keypoints, 4))
        initial_ids_t = torch.empty((0,), dtype=torch.int64)

        return torch.ops.openpifpaf_decoder.cifcaf_op(
            self.n_keypoints, self.skeleton,
            cif_field, 8,
            caf_field, 8,
            initial_annotations_t, initial_ids_t,
        )


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


def apply(model, outfile, *, input_w=129, input_h=97):
    image_size_warning(model.base_net.stride, input_w, input_h)

    # configure: inplace-ops are not supported
    openpifpaf.network.heads.CompositeField3.inplace_ops = False
    openpifpaf.network.heads.CompositeField4.inplace_ops = False

    dummy_input = torch.randn(1, 3, input_h, input_w)
    with torch.no_grad():
        traced_encoder = torch.jit.trace(model, dummy_input)
    decoder = DecoderModule(model.head_metas[0], model.head_metas[1])

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
