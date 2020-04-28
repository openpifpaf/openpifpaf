# openpifpaf

Continuously tested on Linux, MacOS and Windows: [![Build Status](https://travis-ci.org/vita-epfl/openpifpaf.svg?branch=master)](https://travis-ci.org/vita-epfl/openpifpaf)<br />
[CVPR 2019 paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Kreiss_PifPaf_Composite_Fields_for_Human_Pose_Estimation_CVPR_2019_paper.html),
[arxiv.org/abs/1903.06593](https://arxiv.org/abs/1903.06593)

> PifPaf: Composite Fields for Human Pose Estimation
>
> We propose a new bottom-up method for multi-person 2D human pose
> estimation that is particularly well suited for urban mobility such as self-driving cars
> and delivery robots. The new method, PifPaf, uses a Part Intensity Field (PIF) to
> localize body parts and a Part Association Field (PAF) to associate body parts with each other to form
> full human poses.
> Our method outperforms previous methods at low resolution and in crowded,
> cluttered and occluded scenes
> thanks to (i) our new composite field PAF encoding fine-grained information and (ii) the choice of Laplace loss for regressions which incorporates a notion of uncertainty.
> Our architecture is based on a fully
> convolutional, single-shot, box-free design.
> We perform on par with the existing
> state-of-the-art bottom-up method on the standard COCO keypoint task
> and produce state-of-the-art results on a modified COCO keypoint task for
> the transportation domain.


# Demo

![example image with overlaid pose predictions](docs/coco/000000081988.jpg.predictions.png)

Image credit: "[Learning to surf](https://www.flickr.com/photos/fotologic/6038911779/in/photostream/)" by fotologic which is licensed under [CC-BY-2.0].<br />
Created with
`python3 -m openpifpaf.predict docs/coco/000000081988.jpg --show --image-output --json-output`
which also produces [json output](docs/coco/000000081988.jpg.predictions.json).

More demos:
* [openpifpafwebdemo](https://github.com/vita-epfl/openpifpafwebdemo) project (best performance)
* OpenPifPaf running in your browser: https://vita-epfl.github.io/openpifpafwebdemo/ (experimental)
* the `openpifpaf.video` command (requires OpenCV)
* [Google Colab demo](https://colab.research.google.com/drive/1H8T4ZE6wc0A9xJE4oGnhgHpUpAH5HL7W)

<img src="docs/wave3.gif" height=250 alt="example image" />


# Install

Python 3 is required. Python 2 is not supported.
Do not clone this repository
and make sure there is no folder named `openpifpaf` in your current directory.

```sh
pip3 install openpifpaf
```

For a live demo, we recommend to try the
[openpifpafwebdemo](https://github.com/vita-epfl/openpifpafwebdemo) project.
Alternatively, `openpifpaf.video` (requires OpenCV) provides a live demo as well.

For development of the openpifpaf source code itself, you need to clone this repository and then:

```sh
pip3 install numpy cython
pip3 install --editable '.[train,test]'
```

The last command installs the Python package in the current directory
(signified by the dot) with the optional dependencies needed for training and
testing. If you modify `functional.pyx`, run this last command again which
recompiles the static code.


# Interfaces

* `python3 -m openpifpaf.predict --help`: [help screen](docs/cli-help-predict.txt)
* `python3 -m openpifpaf.video --help`: [help screen](docs/cli-help-video.txt)
* `python3 -m openpifpaf.train --help`: [help screen](docs/cli-help-train.txt)
* `python3 -m openpifpaf.eval_coco --help`: [help screen](docs/cli-help-eval_coco.txt)
* `python3 -m openpifpaf.logs --help`: [help screen](docs/cli-help-logs.txt)

Tools to work with models:

* `python3 -m openpifpaf.migrate --help`: [help screen](docs/cli-help-migrate.txt)
* `python3 -m openpifpaf.export_onnx --help`: [help screen](docs/cli-help-export_onnx.txt)


# Pre-trained Models

Performance metrics with version 0.11 on the COCO val set obtained with a GTX1080Ti:

| Backbone               | AP       | APᴹ      | APᴸ      | t_{total} [ms]  | t_{dec} [ms] |
|-----------------------:|:--------:|:--------:|:--------:|:---------------:|:------------:|
| [shufflenetv2k18w]     | __65.0__ | ????     | ????     | ??              | ??           |
| [shufflenetv2k32w]     | __71.5__ | ????     | ????     | ??              | ??           |

For comparison, old v0.10:

| Backbone               | AP       | APᴹ      | APᴸ      | t_{total} [ms]  | t_{dec} [ms] |
|-----------------------:|:--------:|:--------:|:--------:|:---------------:|:------------:|
| [shufflenetv2x2] v0.10 | __60.4__ | 55.5     | 67.8     | 56              | 33           |
| [resnet50] v0.10       | __64.4__ | 61.1     | 69.9     | 76              | 32           |
| [resnet101] v0.10      | __67.8__ | 63.6     | 74.3     | 97              | 28           |

[SHUFFLENETV2X1]: https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/v0.1.0/shufflenetv2x1-pif-paf-edge401-190705-151607-d9a35d7e.pkl
[shufflenetv2x2]: https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/v0.10.0/shufflenetv2x2-pif-paf-paf25-edge401-191010-172527-ef704f06.pkl
[resnet18]: https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/v0.10.1/resnet18-pif-paf-paf25-edge401-191022-210137-84326f0f.pkl
[resnet50]: https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/v0.10.0/resnet50-pif-paf-paf25-edge401-191016-192503-d2b85396.pkl
[resnet101]: https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/v0.10.0/resnet101block5-pif-paf-paf25-edge401-191012-132602-a2bf7ecd.pkl

Pretrained model files are shared in the
__[openpifpaf-torchhub](https://github.com/vita-epfl/openpifpaf-torchhub/releases)__
repository and linked from the backbone names in the table above.
The pretrained models are downloaded automatically when
using the command line option `--checkpoint backbonenameasintableabove`.

To visualize logs:

```sh
python3 -m openpifpaf.logs \
  outputs/resnet50block5-pif-paf-edge401-190424-122009.pkl.log \
  outputs/resnet101block5-pif-paf-edge401-190412-151013.pkl.log \
  outputs/resnet152block5-pif-paf-edge401-190412-121848.pkl.log
```


# Train

See [datasets](docs/datasets.md) for setup instructions.
See [studies.ipynb](docs/studies.ipynb) for previous studies.

The exact training command that was used for a model is in the first
line of the training log file.

Train a ResNet model:

```sh
time CUDA_VISIBLE_DEVICES=0,1 python3 -m openpifpaf.train \
  --lr=1e-3 \
  --momentum=0.95 \
  --epochs=150 \
  --lr-decay 120 140 \
  --batch-size=16 \
  --basenet=resnet101 \
  --head-quad=1 \
  --headnets pif paf paf25 \
  --square-edge=401 \
  --lambdas 10 1 1 15 1 1 15 1 1
```

ShuffleNet models are trained without ImageNet pretraining:

```sh
time CUDA_VISIBLE_DEVICES=2,3 python3 -m openpifpaf.train \
  --lr=0.1 \
  --momentum=0.9 \
  --epochs=250 \
  --lr-warm-up-epochs=1 \
  --lr-decay 120 220 230 235 240 245 \
  --lr-decay-epochs=5 \
  --lr-decay-factor=0.5 \
  --batch-size=32 \
  --square-edge=385 \
  --lambdas 1 1 0.2   1 1 1 0.2 0.2    1 1 1 0.2 0.2 \
  --auto-tune-mtl \
  --weight-decay=1e-5 \
  --update-batchnorm-runningstatistics \
  --ema=0.01 \
  --basenet=shufflenetv2k32w \
  --headnets cif caf caf25

# for improved performance, take the epoch150 checkpoint and train with
# extended-scale and 10% orientation-invariant:
time CUDA_VISIBLE_DEVICES=0,1 python3 -m openpifpaf.train \
  --lr=0.1 \
  --momentum=0.9 \
  --epochs=250 \
  --lr-warm-up-epochs=1 \
  --lr-decay 120 220 230 235 240 245 \
  --lr-decay-epochs=5 \
  --lr-decay-factor=0.5 \
  --batch-size=32 \
  --square-edge=385 \
  --lambdas 1 1 0.2   1 1 1 0.2 0.2    1 1 1 0.2 0.2 \
  --auto-tune-mtl \
  --weight-decay=1e-5 \
  --update-batchnorm-runningstatistics \
  --ema=0.01 \
  --checkpoint outputs/shufflenetv2k32w-200424-175127-cif-caf-caf25.pkl.epoch150_ --extended-scale --orientation-invariant=0.1
```

You can refine an existing model with the `--checkpoint` option.

To produce evaluations at every epoch, check the directory for new
snapshots every 5 minutes:

```
while true; do \
  CUDA_VISIBLE_DEVICES=0 find outputs/ -name "resnet101block5-pif-paf-l1-190109-113346.pkl.epoch???" -exec \
    python3 -m openpifpaf.eval_coco --checkpoint {} -n 500 --long-edge=641 --skip-existing \; \
  ; \
  sleep 300; \
done
```


# Person Skeletons

COCO / kinematic tree / dense:

<img src="docs/skeleton_coco.png" height="250" /><img src="docs/skeleton_kinematic_tree.png" height="250" /><img src="docs/skeleton_dense.png" height="250" />

Created with `python3 -m openpifpaf.datasets.constants`.


# Video

```sh
python3 -m openpifpaf.video --checkpoint shufflenetv2k18w myvideotoprocess.mp4 --video-output --json-output
```


# Documentation Pages

* [datasets](docs/datasets.md)
* [predict.ipynb](docs/predict.ipynb), on [Google Colab](https://colab.research.google.com/drive/1H8T4ZE6wc0A9xJE4oGnhgHpUpAH5HL7W)
* [studies.ipynb](docs/studies.ipynb)
* [evaluation logs](docs/eval_logs.md)
* [history](HISTORY.md)
* [contributing](CONTRIBUTING.md)


# Related Projects

* [monoloco](https://github.com/vita-epfl/monoloco): "Monocular 3D Pedestrian Localization and Uncertainty Estimation" which uses OpenPifPaf for poses.
* [openpifpafwebdemo](https://github.com/vita-epfl/openpifpafwebdemo): web front-end.


# Citation

```
@InProceedings{kreiss2019pifpaf,
  author = {Kreiss, Sven and Bertoni, Lorenzo and Alahi, Alexandre},
  title = {PifPaf: Composite Fields for Human Pose Estimation},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```


[CC-BY-2.0]: https://creativecommons.org/licenses/by/2.0/
