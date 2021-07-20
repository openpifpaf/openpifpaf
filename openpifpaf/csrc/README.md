## Build libtorch example with cmake

Roughly following the
[PyTorch Tutorial on TorchScript in C++](https://pytorch.org/tutorials/advanced/cpp_export.html).

```sh
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
cmake --build . --config Release
```
