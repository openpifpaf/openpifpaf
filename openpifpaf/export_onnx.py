import argparse
import torch

import openpifpaf


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
        if self_.apply_class_sigmoid:
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

        n_fields = 17 if len(regs_x) == 1 else 19
        # print(self_.shortname, n_fields)
        regs_x = [
            reg_x.reshape(1,
                          n_fields,
                          2,
                          25,
                          33)
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

    dummy_input = torch.randn(1, 3, 193, 257)
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
        'paf_b1',
        'paf_r2',
        'paf_b2',
    ]

    torch.onnx.export(model, dummy_input, outfile, verbose=verbose,
                      input_names=input_names, output_names=output_names)

    monkey_patches.revert()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='resnet50')
    parser.add_argument('--outfile', default='openpifpaf-resnet50.onnx')
    args = parser.parse_args()
    apply(args.checkpoint, args.outfile)


if __name__ == '__main__':
    main()
