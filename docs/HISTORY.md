# History

* [main](https://github.com/openpifpaf/openpifpaf/compare/v0.13.0...main)
* [0.13.0](https://github.com/openpifpaf/openpifpaf/compare/v0.12.14...v0.13.0) (2021-09-26)
    * major rewrite of the decoders to C++ backed by libtorch
* [0.12.14](https://github.com/openpifpaf/openpifpaf/compare/v0.12.13...v0.12.14) (2021-09-15)
    * new NuScenes plugin [#467](https://github.com/openpifpaf/openpifpaf/pull/467), [#472](https://github.com/openpifpaf/openpifpaf/pull/472)
    * new transformer backbones [#463](https://github.com/openpifpaf/openpifpaf/pull/463), [#471](https://github.com/openpifpaf/openpifpaf/pull/471), [#481](https://github.com/openpifpaf/openpifpaf/pull/481), [#488](https://github.com/openpifpaf/openpifpaf/pull/488): pretrained `swin_s`, `swin_b`, `swin_t_input_upsample`
    * new Apollo checkpoints [#473](https://github.com/openpifpaf/openpifpaf/pull/473)
    * lazy ops for visualizer [#490](https://github.com/openpifpaf/openpifpaf/pull/490)
* [0.12.13](https://github.com/openpifpaf/openpifpaf/compare/v0.12.12...v0.12.13) (2021-07-22)
    * fix regression: avoid CenterPad when batch=1 [#453](https://github.com/openpifpaf/openpifpaf/pull/453)
    * include draw time in "post" time [#454](https://github.com/openpifpaf/openpifpaf/pull/454)
* [0.12.12](https://github.com/openpifpaf/openpifpaf/compare/v0.12.11...v0.12.12) (2021-07-07)
    * constrain Pillow dependency below 8.3 for torchvision compatibility
    * Wholebody fixes [#451](https://github.com/openpifpaf/openpifpaf/pull/451)
    * det decoder: nms by category and minor fix to nms suppression
    * debug: properly show crowd annotations in encoder debug plots
* [0.12.11](https://github.com/openpifpaf/openpifpaf/compare/v0.12.10...v0.12.11) (2021-06-11)
    * wheels for aarch64
    * new pretrained models: mobilenetv3large and resnet18-cocodet
    * show: default output is now jpeg with image-dpi-factor=2.0
    * ONNX export with dynamic batch dimension
* [0.12.10](https://github.com/openpifpaf/openpifpaf/compare/v0.12.9...v0.12.10) (2021-05-19)
    * new Guide tutorials: Wholebody, Apollo, Animals
    * Predictor API
    * MobileNetv3 backbone
    * BCE background clamp
    * support and CI tests for Python 3.9
    * fix for multiple metrics
* [0.12.9](https://github.com/openpifpaf/openpifpaf/compare/v0.12.8...v0.12.9) (2021-04-21)
    * new tutorials: Custom Dataset and ApolloCar3D
    * bugfix: inverse ground-truth annotation for metric [#399](https://github.com/openpifpaf/openpifpaf/pull/399)
    * scipy is now an optional dependency
    * binaries for Apple Silicon (universal2 and arm64)
    * automatically register plugins
    * pad before RotateUniform
    * area-of-interest based cropping now uses bounding box annotations and not keypoints
    * matplotlib 3.4 compatibility
    * ONNX 1.9 compatibility
* [0.12.8](https://github.com/openpifpaf/openpifpaf/compare/v0.12.7...v0.12.8) (2021-04-11)
    * moved to github.com/openpifpaf/openpifpaf
* [0.12.7](https://github.com/vita-epfl/openpifpaf/compare/v0.12.6...v0.12.7) (2021-04-09)
    * Apollo3D car model with 66 keypoints [#376](https://github.com/vita-epfl/openpifpaf/pull/376)
    * new pretrained models for shufflenetv2k16 and shufflenetv2k30
    * new transforms.RotateUniform and transforms.RandomChoice [#380](https://github.com/vita-epfl/openpifpaf/pull/380)
    * cython decoder.utils.Occupancy [#381](https://github.com/vita-epfl/openpifpaf/pull/381)
    * refactor Stream source for video data [#371](https://github.com/vita-epfl/openpifpaf/pull/371)
    * additional tests and minor changes for jit.script compatibility [#377](https://github.com/vita-epfl/openpifpaf/pull/377)
    * improve visualization of occupancy map for debugging
* [0.12.6](https://github.com/vita-epfl/openpifpaf/compare/v0.12.5...v0.12.6) (2021-03-24)
    * new training option: `--clip-grad-value`
    * faster training startup when training from checkpoint: [#363](https://github.com/vita-epfl/openpifpaf/pull/363)
    * test PyTorch 1.8
    * eval tool: show all available eval stats by default
    * new argument for video tool: `--video-output=virtualcam`
* [0.12.5](https://github.com/vita-epfl/openpifpaf/compare/v0.12.4...v0.12.5) (2021-03-04)
    * speed improvement for `openpifpaf.video` [#362](https://github.com/vita-epfl/openpifpaf/pull/362)
    * new paper on arxiv: https://arxiv.org/abs/2103.02440
* [0.12.4](https://github.com/vita-epfl/openpifpaf/compare/v0.12.0...v0.12.4) (2021-03-02)
    * minor changes (mostly to experimental features like DDP)
    * OpenCV tutorial in Guide
    * new `datasets.NumpyImageList`
* [0.12.0](https://github.com/vita-epfl/openpifpaf/compare/v0.11.9...v0.12.0) (2021-02-??)
    Major rewrite with many backwards incompatible changes.
    The Guide is updated: https://vita-epfl.github.io/openpifpaf/intro.html
    Major changes:
    * plugins: many components are now customizable with independent plugins; see [Guide: Plugins Overview](https://vita-epfl.github.io/openpifpaf/plugins_overview.html)
    * Datamodule: a new class that holds all information related to datasets. Create a new Datamodule to add a new dataset to OpenPifPaf. Cifar10 is now implemented as a plugin and [documented](https://vita-epfl.github.io/openpifpaf/plugins_cifar10.html).
    * Metric: now you can implement your own with a plugin.
    * train on multiple datasets simultaneously
    * export to CoreML for iOS
    * export to onnx with full test with onnxruntime
    * `--show` and `--save-all` to either show or store all debug images as files for remote debugging
    * experimental support for DistributedDataParallel (DDP) training
    * moved to Github Actions
    * continuous deployment
* [0.11.9](https://github.com/vita-epfl/openpifpaf/compare/v0.11.8...v0.11.9) (2020-08-25)
    * new `--val-interval` which replaces the use-cases for previous `--duplicate-data`
    * extended ColorJitter ranges for training: brightness=0.4, saturation=0.4, hue=0.1
    * PyLint 2.6 compatibility
    * matplotlib 3.3 compatibility
    * Cython 0.29.21
* [0.11.8](https://github.com/vita-epfl/openpifpaf/compare/v0.11.7...v0.11.8) (2020-08-19)
    * PyTorch 1.6 compatibility
    * modified loss recommendation to avoid autotune
    * use versioneer to create dynamic version numbers
    * use stretch augmentation [#262](https://github.com/vita-epfl/openpifpaf/pull/262)
    * improved decoding performance [#252](https://github.com/vita-epfl/openpifpaf/pull/252)
    * add v0.8 cpp decoder [#249](https://github.com/vita-epfl/openpifpaf/pull/249)
* [0.11.7](https://github.com/vita-epfl/openpifpaf/compare/v0.11.6...v0.11.7) (2020-07-14)
    * fix predict with multiple GPUs
    * fix ONNX optimization
    * fix dense connection option and add a test for it
    * suppress a warning during training that doesn't apply
* [0.11.6](https://github.com/vita-epfl/openpifpaf/compare/v0.11.4...v0.11.6) (2020-06-15)
    * circular truncation for CifHr
    * extend Guide
* [0.11.4](https://github.com/vita-epfl/openpifpaf/compare/v0.11.3...v0.11.4) (2020-06-02)
    * init Guide
    * minor updates
* [0.11.3](https://github.com/vita-epfl/openpifpaf/compare/v0.11.2...v0.11.3) (2020-06-01)
    * new debug message for "neural network device" (to check cpu versus gpu usage)
    * debug output without plots is default; enable debug plots with new `--debug-images`
* [0.11.2](https://github.com/vita-epfl/openpifpaf/compare/v0.11.0...v0.11.2) (2020-05-29)
    * pretrained resnet50 model
    * fix CUDA support in `openpifpaf.video`
    * add `--version` option to all CLIs
* [0.11.0](https://github.com/vita-epfl/openpifpaf/compare/v0.10.1...v0.11.0) (2020-05-12)
    * major refactor
    * now requires Python>=3.6 for type annotations
    * new ShuffleNetV2 models: `shufflenetv2k16w` and `shufflenetv2k30w`
    * 64bit loss and Focal Loss for confidences
    * fast fused convolutions for `CompositeHeadFused`
    * new handling of crowd annotations in encoder
    * new `--extended-scale` training and eval-coco option
    * decoding with frontier is default
    * more robust blending of connection candidates
    * introduced `openpifpaf.visualizer` and many improvements to visualization
    * [experimental] new `cocodet` dataset interface for detections
* [0.10.1](https://github.com/vita-epfl/openpifpaf/compare/v0.10.0...v0.10.1) (2019-12-09)
    * faster decoder
    * refactored scale generation between loss and encoder
* [0.10.0](https://github.com/vita-epfl/openpifpaf/compare/v0.9.0...v0.10.0) (2019-10-21)
    * major refactor: move all factory-code into factories
    * new experimental decoder
    * improved image rescaling randomization
    * module-level logging
    * index-matching bug fixed by @junedgar [#147](https://github.com/vita-epfl/openpifpaf/pull/147)
    * tests for Windows, PyTorch 1.3 and with pylint 2.4
* [0.9.0](https://github.com/vita-epfl/openpifpaf/compare/v0.8.0...v0.9.0) (2019-07-30)
    * make image transforms part of preprocessing [#100](https://github.com/vita-epfl/openpifpaf/pull/100)
    * field-based two-scale implementation [#101](https://github.com/vita-epfl/openpifpaf/pull/101), also modifies single-scale decoder
    * added a `show.CrowdPainter` to visualize crowd annotations
    * new benchmark script and updated performance numbers in README.md, [#104](https://github.com/vita-epfl/openpifpaf/pull/104)
* [0.8.0](https://github.com/vita-epfl/openpifpaf/compare/v0.7.0...v0.8.0) (2019-07-08)
    * add support for `resnext50`, `shufflenetv2x1` and `shufflenetv2x2`
    * new pretrained models
    * new transforms.RandomApply() and transforms.RotateBy90(); removed old transforms
    * new blur augmentation
    * improved BCE masks [#87](https://github.com/vita-epfl/openpifpaf/pull/87)
* [0.7.0](https://github.com/vita-epfl/openpifpaf/compare/v0.6.3...v0.7.0) (2019-06-06)
    * faster seed generation in decoder
    * training log plot improvements (labels, consistent colors)
    * improved debug visualizer for decoder
* [0.6.3](https://github.com/vita-epfl/openpifpaf/compare/v0.6.2...v0.6.3) (2019-05-28)
    * support parallel decoding for `predict` and `eval_coco` (~4x speed improvement) which is automatically activated for batch sizes larger than 1
* [0.6.2](https://github.com/vita-epfl/openpifpaf/compare/v0.6.1...v0.6.2) (2019-05-23)
    * improved decoder performance [#61](https://github.com/vita-epfl/openpifpaf/pull/61), [#63](https://github.com/vita-epfl/openpifpaf/pull/63), [#64](https://github.com/vita-epfl/openpifpaf/pull/64)
    * remove `apply_class_sigmoid` property from head nets and use the standard `model.train()` and `model.eval()` methods instead
    * bugfix for runs with padding
    * improved log messages
    * log the names of the fields (preparation to improve plots of training losses)
* [0.6.1](https://github.com/vita-epfl/openpifpaf/compare/v0.6.0...v0.6.1) (2019-05-13)
    * improved decoder performance [#51](https://github.com/vita-epfl/openpifpaf/pull/51)
    * experiments with ONNX
    * MultiHeadLoss [#50](https://github.com/vita-epfl/openpifpaf/pull/50) (no external API changes)
    * automatically add hash to trained model files for Model Zoo compatibility [#53](https://github.com/vita-epfl/openpifpaf/pull/53)
    * support nested objects in factories [#56](https://github.com/vita-epfl/openpifpaf/pull/56)
* [0.6.0](https://github.com/vita-epfl/openpifpaf/compare/v0.5.1...v0.6.0) (2019-05-07)
    * Torch 1.1.0 compatibility (older versions work but have modified learning rate schedule due to https://github.com/pytorch/pytorch/pull/7889)
    * more aggressive NMS
    * multi-scale support for `eval_coco`: `--two-scale`, `--three-scale`, `--multi-scale`
    * guaranteed complete poses with `--force-complete-pose`
* [0.5.1](https://github.com/vita-epfl/openpifpaf/compare/v0.5.0...v0.5.1) (2019-05-01)
