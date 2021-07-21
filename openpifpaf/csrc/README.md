## Build libtorch example with cmake

Roughly following the
[PyTorch Tutorial on TorchScript in C++](https://pytorch.org/tutorials/advanced/cpp_export.html).

This demo requires header and shared library files for OpenCV.
On a Mac, use `brew install opencv`.

```sh
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
cmake --build . --config Release
```


## Create TorchScript Module and run Examples

```
python -m openpifpaf.export_torchscript --input-height=427 --input-width=640

# single image
./openpifpaf-image openpifpaf-shufflenetv2k16.torchscript.pt ../../../docs/coco/000000081988.jpg

# stream from OpenCV webcam 0 (includes resizing to 640x427)
./openpifpaf-video openpifpaf-shufflenetv2k16.torchscript.pt 0 640 427
```
