"""Export a checkpoint as a CoreML model."""

import argparse
import logging

import torch

import openpifpaf

from .export_onnx import image_size_warning

try:
    import coremltools
except ImportError:
    coremltools = None

LOG = logging.getLogger(__name__)


def apply(model, outfile, *, input_w=129, input_h=97, minimum_deployment_target='iOS14'):
    assert coremltools is not None
    image_size_warning(model.base_net.stride, input_w, input_h)

    # configure: inplace-ops are not supported
    openpifpaf.network.heads.CompositeField3.inplace_ops = False

    dummy_input = torch.randn(1, 3, input_h, input_w)
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)

    coreml_model = coremltools.convert(
        traced_model,
        inputs=[coremltools.ImageType(name='image', shape=dummy_input.shape,
                                      bias=[-1.0, -1.0, -1.0], scale=1.0 / 127.0)],
        # classifier_config = ct.ClassifierConfig(class_labels)
        minimum_deployment_target=getattr(coremltools.target, minimum_deployment_target),
    )

    # pylint: disable=protected-access
    coremltools.models.utils.rename_feature(
        coreml_model._spec, coreml_model._spec.description.output[0].name, 'cif_head')
    coremltools.models.utils.rename_feature(
        coreml_model._spec, coreml_model._spec.description.output[1].name, 'caf_head')

    # Meta
    coreml_model.input_description['image'] = 'Input image to be classified'
    coreml_model.output_description['cif_head'] = 'Composite Intensity Field'
    coreml_model.output_description['caf_head'] = 'Composite Association Field'
    coreml_model.author = 'Kreiss, Bertoni, Alahi: Composite Fields for Human Pose Estimation'
    coreml_model.license = 'Please see https://github.com/vita-epfl/openpifpaf'
    coreml_model.short_description = 'Composite Fields for Human Pose Estimation'
    coreml_model.version = openpifpaf.__version__

    coreml_model.save(outfile)

    # # test predict
    # test_predict = coreml_model.predict({'input_1': dummy_input.numpy()})
    # print('!!!!!!!!', test_predict)


def print_preprocessing_spec(out_name):
    spec = coremltools.models.utils.load_spec(out_name)
    print(spec.neuralNetwork.preprocessing)  # pylint: disable=no-member


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
    parser.add_argument('--minimum-deployment-target', choices=('iOS13', 'iOS14'), default='iOS14')
    args = parser.parse_args()

    model, _ = openpifpaf.network.factory(checkpoint=args.checkpoint)

    assert args.outfile.endswith('.mlmodel')
    apply(model, args.outfile,
          input_w=args.input_width, input_h=args.input_height,
          minimum_deployment_target=args.minimum_deployment_target)
    print_preprocessing_spec(args.outfile)


if __name__ == '__main__':
    main()
