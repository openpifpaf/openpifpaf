import time
# Third-party library imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
#import onnx
# Local module imports from OpenPifPaf
import openpifpaf
from openpifpaf import decoder, logger, network, show, visualizer, __version__
from openpifpaf.predictor import Predictor
import onnxruntime as ort
import gc
import argparse
import PIL
print(torch.__version__)
#print(onnx.__version__)
def open_image(image_path):
        pil_im = PIL.Image.open(image_path).convert('RGB')
        return pil_im
def image_size_warning(basenet_stride, input_w, input_h):
    if input_w % basenet_stride != 1:
        print(
            'Warning: input width should be a multiple of basenet stride + 1: closest are %d and %d' %
            ((input_w - 1) // basenet_stride * basenet_stride + 1,
             ((input_w - 1) // basenet_stride + 1) * basenet_stride + 1)
        )
    if input_h % basenet_stride != 1:
        print(
            'Warning: input height should be a multiple of basenet stride + 1: closest are %d and %d' %
            ((input_h - 1) // basenet_stride * basenet_stride + 1,
             ((input_h - 1) // basenet_stride + 1) * basenet_stride + 1)
        )
def apply(model, outfile, input_w, input_h, channels, output_names=None):
    if output_names is None:
        output_names = ['cif', 'caf']
    openpifpaf.network.heads.CompositeField3.inplace_ops = False
    openpifpaf.network.heads.CompositeField4.inplace_ops = False
    image_size_warning(model.base_net.stride, input_w, input_h)
    model = model.to('cpu')
    dummy_input = torch.randn(1, channels, input_h, input_w)
#    with torch.inference_mode():
        # initialize
    model(dummy_input)
    gc.collect()
    print('reached the conversion part')
    torch.onnx.export(
            model, dummy_input, outfile,
            input_names=['input_batch'], output_names=output_names,
            opset_version=11, do_constant_folding=True,
            dynamic_axes={
                'input_batch': {0: 'dynamic'},
                'cif': {0: 'dynamic'},
                'caf': {0: 'dynamic'}
            }
        )
def test_onnx_model(onnx_model_path, image_path, input_width, input_height, use_gpu=True):
    providers = ['CUDAExecutionProvider'] if use_gpu and torch.cuda.is_available() else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_model_path, providers=providers)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (input_width, input_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.float32(image).transpose(2, 0, 1) / 255.0
    image = np.expand_dims(image, axis=0)
    input_name = ort_session.get_inputs()[0].name
    output_names = [output.name for output in ort_session.get_outputs()]
    start_time = time.time()
    pred_onnx = ort_session.run(output_names, {input_name: image})
    end_time = time.time()
    print("ONNX Runtime Inference Complete")
    print(f"Inference Time: {end_time - start_time} seconds")
    return pred_onnx
def compare_model_performance(checkpoint_path, onnx_model_path, image_path, input_width, input_height, export_module=True, use_gpu=True):
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    image = cv2.imread(image_path)
    image = cv2.resize(image, (input_width, input_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.float32(image).transpose(2, 0, 1) / 255.0
    image_tensor = torch.tensor(image).unsqueeze(0).to(device)
    # Load PyTorch model
    datamodule = openpifpaf.datasets.factory('nba')
    pytorch_model, _ = openpifpaf.network.Factory(checkpoint=checkpoint_path).factory(head_metas=datamodule.head_metas)
    pytorch_model = pytorch_model.to(device).eval()
    print(pytorch_model)
    print(image_tensor.shape)
    with torch.no_grad():
        for i in range(0):
                start_time = time.time()
                pred_pytorch = pytorch_model(image_tensor)
                end_time = time.time()
                print(f"PyTorch Inference Time (Run {i+1}): {end_time - start_time} seconds")
    # Prepare model for ONNX export
    if export_module:
        apply(pytorch_model, outfile=onnx_model_path, input_w=input_width, input_h=input_height, channels=3)
    # ONNX prediction loop
    providers = ['CUDAExecutionProvider'] if use_gpu and torch.cuda.is_available() else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_model_path, providers=providers)
    input_name = ort_session.get_inputs()[0].name
    output_names = [output.name for output in ort_session.get_outputs()]
    for i in range(15):
        start_time = time.time()
        pred_onnx = ort_session.run(output_names, {input_name: image_tensor.cpu().numpy()})
        end_time = time.time()
        print(f"ONNX Inference Time (Run {i+1}): {end_time - start_time} seconds")
        assert pred_pytorch[0].shape == pred_onnx[0].shape
        assert pred_pytorch[1].shape == pred_onnx[1].shape
        np.testing.assert_allclose(pred_pytorch[0].cpu().numpy(), pred_onnx[0], rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(pred_pytorch[1].cpu().numpy(), pred_onnx[1], rtol=1e-03, atol=1e-05)
def predict_single(image_path, checkpoint_path, onnx=False, input_width=3825, input_height=1617, export_path=None, prune=None):
    import argparse
    image = cv2.imread(image_path)
    print(image)

#    image = cv2.resize(image, (input_width, input_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = PIL.Image.fromarray(image)
    print(image.shape)
    parser = argparse.ArgumentParser(description='Infer openpifpaf joints for game')
    decoder.cli(parser)
    logger.cli(parser)
    network.Factory.cli(parser)
    Predictor.cli(parser)
    show.cli(parser)
    visualizer.cli(parser)
    args = parser.parse_args()
    args.device = torch.device('cuda')
    args.line_width = 1
    decoder.configure(args)
    network.Factory.configure(args)
    Predictor.configure(args)
    show.configure(args)
    visualizer.configure(args)
    predictor = Predictor(checkpoint= checkpoint_path, visualize_image=True, 
                          visualize_processed_image=True, onnx_weights='/Users/aleksandrsimonyan/Desktop/openpifpaf_onnx_converted_3793_1585.onnx')# onnx=r'D:\markerless-mocap\weights\openpifpaf_onnx_converted_3793_1585.onnx')# onnx=r'C:\Users\asimonyan\Desktop\open_pif_paf\experiments\shuflnet2kv30\epo.epoch22.onnx')#onnx=r'C:\Users\asimonyan\Desktop\open_pif_paf\experiments\shuflnet2kv30\epo.epoch22.onnx')
    img = open_image(image_path)
    w,h = pil_image.size
    print(w,h)
    img = open_image(image_path)
    w,h = pil_image.size
    pil_image = img.resize((input_width, input_height))
    predictions, gt_anns, image_meta =predictor.pil_image(pil_image)
    annotation_painter = openpifpaf.show.AnnotationPainter()
    with openpifpaf.show.image_canvas(pil_image) as ax:
            annotation_painter.annotations(ax, predictions)
            plt.show()
if __name__ == '__main__':

    target_sizes = [3793, 1585]
    parser = argparse.ArgumentParser(description='OpenPifPaf ONNX Conversion and Prediction')
#    parser.add_argument('-i','--input', default=checkpoint_path, type=str, required=False, help='Path to the input checkpoint')
#    parser.add_argument('-o','--output', default=export_path,type=str, required=False, help='Path to the output ONNX file')
#    parser.add_argument('-im', '--image', default=image_path, type=str, required=False, help='Path to the image file')
    args = parser.parse_args()
    predict_single('/Users/aleksandrsimonyan/Desktop/cam_91206.png', '/Users/aleksandrsimonyan/Desktop/cam_91206.png', onnx=True, input_width=target_sizes[0], input_height=target_sizes[1], export_path=None, prune=None)
    # Assuming compare_model_performance and predict_single are your existing functions:\
    ''
    compare_model_performance(
        checkpoint_path=None,
        onnx_model_path=None,
        image_path='/Users/aleksandrsimonyan/Desktop/cam_91206.png',
        input_width=target_sizes[0],
        input_height=target_sizes[1],
        export_module=True
    )
''