# openpifpaf

Continuously tested on Linux, MacOS and Windows: [![Build Status](https://travis-ci.org/vita-epfl/openpifpaf.svg?branch=master)](https://travis-ci.org/vita-epfl/openpifpaf)<br />
[CVPR 2019 paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Kreiss_PifPaf_Composite_Fields_for_Human_Pose_Estimation_CVPR_2019_paper.html)
<!-- [arxiv.org/abs/1903.06593](https://arxiv.org/abs/1903.06593) -->

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


# Example

![example image with overlaid pose predictions](https://github.com/vita-epfl/openpifpaf/raw/master/docs/coco/000000081988.jpg.predictions.png)

Image credit: "[Learning to surf](https://www.flickr.com/photos/fotologic/6038911779/in/photostream/)" by fotologic which is licensed under [CC-BY-2.0].<br />
Created with:
```sh
pip3 install openpifpaf matplotlib
python3 -m openpifpaf.predict coco/000000081988.jpg --image-output
```


# [Guide](https://vita-epfl.github.io/openpifpaf/intro.html)

Continue to our __[OpenPifPaf Guide](https://vita-epfl.github.io/openpifpaf/intro.html)__.


[CC-BY-2.0]: https://creativecommons.org/licenses/by/2.0/


# Commercial License

This software is available for licensing via the EPFL Technology Transfer
Office (https://tto.epfl.ch/, info.tto@epfl.ch).
