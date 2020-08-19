# History

* [master](https://github.com/vita-epfl/openpifpaf/compare/v0.11.8...master)
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
    * new benchmark script and updated performance numbers in [README.md](https://github.com/vita-epfl/openpifpaf/blob/master/README.md), [#104](https://github.com/vita-epfl/openpifpaf/pull/104)
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
