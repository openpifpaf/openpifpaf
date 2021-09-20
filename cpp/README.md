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

On Linux, special care needs to be taken
for binary compatibility between OpenCV and libtorch. Get the libtorch binaries
with cxx11 ABI from [pytorch.org](https://pytorch.org/get-started/locally/),
e.g. [libtorch-cxx11-abi-shared-with-deps-1.9.0+cpu.zip](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcpu.zip). Unzip
the downloaded file which will create a `libtorch` folder in the current directory.

```sh
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=../libtorch/share/cmake/Torch ..
cmake --build . --config Release
```

This build is tested in `tests/test_cmake.py`.


## Create TorchScript Module and run Examples

```
python -m openpifpaf.export_torchscript --input-height=427 --input-width=640

# single image
./openpifpaf-image openpifpaf-shufflenetv2k16.torchscript.pt ../../../docs/coco/000000081988.jpg

# stream from OpenCV webcam 0 (includes resizing to 640x427)
./openpifpaf-video openpifpaf-shufflenetv2k16.torchscript.pt 0 640 427
```
