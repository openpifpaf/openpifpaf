# History

* [master](https://github.com/vita-epfl/openpifpaf/compare/v0.10.1...master)
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
