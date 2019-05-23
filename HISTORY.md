# History

* [master](https://github.com/vita-epfl/openpifpaf/compare/v0.6.2...master)
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
