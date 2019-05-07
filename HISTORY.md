# History

* [master](https://github.com/vita-epfl/openpifpaf/compare/v0.6.0...master)
* [0.6.0](https://github.com/vita-epfl/openpifpaf/compare/v0.5.1...v0.6.0) (2019-05-07)
    * Torch 1.1.0 compatibility (older versions work but have modified learning rate schedule due to https://github.com/pytorch/pytorch/pull/7889)
    * more aggressive NMS
    * multi-scale support for `eval_coco`: `--two-scale`, `--three-scale`, `--multi-scale`
    * guaranteed complete poses with `--force-complete-pose`
* [0.5.1](https://github.com/vita-epfl/openpifpaf/compare/v0.5.0...v0.5.1) (2019-05-01)
