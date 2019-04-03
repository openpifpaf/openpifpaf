# openpifpaf

[![Build Status](https://travis-ci.org/vita-epfl/openpifpaf.svg?branch=master)](https://travis-ci.org/vita-epfl/openpifpaf)

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


```
@article{kreiss2019pifpaf,
  title={PifPaf: Composite Fields for Human Pose Estimation},
  author={Kreiss, Sven and Bertoni, Lorenzo and Alahi, Alexandre},
  journal={CVPR, arXiv preprint arXiv:1903.06593},
  year={2019}
}
```

[arxiv.org/abs/1903.06593](https://arxiv.org/abs/1903.06593)


# Demo

![example image with overlaid pose skeleton](docs/coco/000000081988.jpg.skeleton.png)

Image credit: "[Learning to surf](https://www.flickr.com/photos/fotologic/6038911779/in/photostream/)" by fotologic which is licensed under [CC-BY-2.0].


Created with:

```sh
python3 -m openpifpaf.predict \
  --checkpoint outputs/resnet101block5-pifs-pafs-edge401-l1-190131-083451.pkl \
  data-mscoco/images/val2017/000000081988.jpg -o docs/coco/ --show
```

For more demos, see the
[openpifpafwebdemo](https://github.com/vita-epfl/openpifpafwebdemo) project and
the `openpifpaf.webcam` command.


# Install

Python 3 is required. Python 2 is not supported.
Do not clone this repository
and make sure there is no folder named `openpifpaf` in your current directory.

```sh
pip3 install openpifpaf
```

For a live demo, we recommend to try the
[openpifpafwebdemo](https://github.com/vita-epfl/openpifpafwebdemo) project.
Alternatively, `openpifpaf.webcam` provides a live demo as well.
It requires OpenCV. To use a globally installed
OpenCV from inside a virtual environment, create the virtualenv with the
`--system-site-packages` option and verify that you can do `import cv2`.

For development of the openpifpaf source code itself, you need to clone this repository and then:

```sh
pip3 install numpy cython
pip3 install --editable '.[train,test]'
```

The last command installs the Python package in the current directory
(signified by the dot) with the optional dependencies needed for training and
testing. The difference between release _v0.3.0_ and the master branch can be seen on
[GitHub compare](https://github.com/vita-epfl/openpifpaf/compare/v0.3.0...master).


# Interfaces

* `python3 -m openpifpaf.predict --help`
* `python3 -m openpifpaf.webcam --help`
* `python3 -m openpifpaf.train --help`
* `python3 -m openpifpaf.eval_coco --help`
* `python3 -m openpifpaf.logs --help`

Example commands to try:

```sh
# live demo
MPLBACKEND=macosx python3 -m openpifpaf.webcam --scale 0.1 --source=0

# single image
python3 -m openpifpaf.predict my_image.jpg --show
```


# Pre-trained Networks

Put these files into your `outputs` folder: [Google Drive](https://drive.google.com/drive/folders/1v8UNDjZbqeMZY64T33tSDOq1jtcBJBy7?usp=sharing>)

Visualize logs:

```sh
python3 -m pifpaf.logs \
  outputs/resnet50-pif-paf-rsmooth0.5-181209-192001.pkl.log \
  outputs/resnet101-pif-paf-rsmooth0.5-181213-224234.pkl.log \
  outputs/resnet152-pif-paf-l1-181230-201001.pkl.log
```


# Train

See [datasets](docs/datasets.md) for setup instructions.
See [studies.ipynb](docs/studies.ipynb) for previous studies.

Train a model:

```sh
python3 -m openpifpaf.train \
  --lr=1e-3 \
  --momentum=0.95 \
  --epochs=75 \
  --lr-decay 60 70 \
  --batch-size=8 \
  --basenet=resnet50block5 \
  --quad=1 \
  --headnets pif paf \
  --square-edge=401 \
  --regression-loss=laplace \
  --lambdas 30 2 2 50 3 3 \
  --crop-fraction=0.5 \
  --freeze-base=1
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

Created with `python3 -m openpifpaf.data`.


# Video

Processing a video frame by frame from `video.avi` to `video-pose.mp4` using ffmpeg:

```sh
ffmpeg -i video.avi -qscale:v 2 -vf scale=641:-1 -f image2 video-%05d.jpg
python3 -m openpifpaf.predict --checkpoint outputs/resnet101block5-pifs-pafs-edge401-l1-190213-100439.pkl video-*0.jpg
ffmpeg -framerate 24 -pattern_type glob -i 'video-*.jpg.skeleton.png' -vf scale=640:-1 -c:v libx264 -pix_fmt yuv420p video-pose.mp4
```


# Evaluations

See [evaluation logs](docs/eval_logs.md) for a long list.
This result was produced with `python3 -m openpifpaf.eval_coco --checkpoint outputs/resnet101block5-pif-paf-edge401-190313-100107.pkl --long-edge=641 --loader-workers=8`:

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.662
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.872
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.724
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.623
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.721
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.712
Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.895
Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.768
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.660
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.785
```


[CC-BY-2.0]: https://creativecommons.org/licenses/by/2.0/
