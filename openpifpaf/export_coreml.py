"""Export a checkpoint as a CoreML model."""

import argparse

import torch

import openpifpaf

try:
    import coremltools
except ImportError:
    coremltools = None


def apply(model, outfile, input_w=129, input_h=97):
    assert coremltools is not None

    dummy_input = torch.randn(1, 3, input_h, input_w)
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)

    output_names = ['cif', 'caf']

    coreml_model = coremltools.convert(
        traced_model,
        inputs=[coremltools.ImageType(name='image', shape=dummy_input.shape)],
        # classifier_config = ct.ClassifierConfig(class_labels) # provide only if step 4 was performed
    )

    # Meta
    coreml_model.input_description['image'] = 'Input image to be classified'
    # coreml_model.output_description['cif'] = 'Composite Intensity Field'
    # coreml_model.output_description['caf'] = 'Composite Association Field'
    coreml_model.author = 'Original Paper: Kreiss, Bertoni, Alahi, "Composite Fields for Human Pose Estimation"'
    coreml_model.license = 'Please see https://github.com/vita-epfl/openpifpaf'
    coreml_model.short_description = 'Composite Fields for Human Pose Estimation'
    coreml_model.version = openpifpaf.__version__

    coreml_model.save(outfile)

    # # test predict
    # test_predict = coreml_model.predict({'input_1': dummy_input.numpy()})
    # print('!!!!!!!!', test_predict)


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def main():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.export_coreml',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=openpifpaf.__version__))

    parser.add_argument('--checkpoint', default='resnet50')
    parser.add_argument('--outfile', default='openpifpaf-resnet50.mlmodel')
    parser.add_argument('--input-width', type=int, default=129)
    parser.add_argument('--input-height', type=int, default=97)
    args = parser.parse_args()

    model, _ = openpifpaf.network.factory(checkpoint=args.checkpoint)

    assert args.outfile.endswith('.mlmodel')
    apply(model, args.outfile, input_w=args.input_width, input_h=args.input_height)


if __name__ == '__main__':
    main()
