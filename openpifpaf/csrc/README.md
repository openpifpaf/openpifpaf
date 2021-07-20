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
