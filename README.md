# openpifpaf

Continuously tested on Linux, MacOS and Windows:
[![Tests](https://github.com/vita-epfl/openpifpaf/workflows/Tests/badge.svg?branch=main)](https://github.com/vita-epfl/openpifpaf/actions?query=workflow%3ATests)
[![deploy-guide](https://github.com/vita-epfl/openpifpaf/workflows/deploy-guide/badge.svg)](https://github.com/vita-epfl/openpifpaf/actions?query=workflow%3Adeploy-guide)
[![Downloads](https://pepy.tech/badge/openpifpaf)](https://pepy.tech/project/openpifpaf)
<br />
[__New__ 2021 paper](https://arxiv.org/abs/2103.02440):

> __OpenPifPaf: Composite Fields for Semantic Keypoint Detection and Spatio-Temporal Association__<br />
> _[Sven Kreiss](https://www.svenkreiss.com), [Lorenzo Bertoni](https://scholar.google.com/citations?user=f-4YHeMAAAAJ&hl=en), [Alexandre Alahi](https://scholar.google.com/citations?user=UIhXQ64AAAAJ&hl=en)_, 2021.
>
> Many image-based perception tasks can be formulated as detecting, associating
> and tracking semantic keypoints, e.g., human body pose estimation and tracking.
> In this work, we present a general framework that jointly detects and forms
> spatio-temporal keypoint associations in a single stage, making this the first
> real-time pose detection and tracking algorithm. We present a generic neural
> network architecture that uses Composite Fields to detect and construct a
> spatio-temporal pose which is a single, connected graph whose nodes are the
> semantic keypoints (e.g., a person's body joints) in multiple frames. For the
> temporal associations, we introduce the Temporal Composite Association Field
> (TCAF) which requires an extended network architecture and training method
> beyond previous Composite Fields. Our experiments show competitive accuracy
> while being an order of magnitude faster on multiple publicly available datasets
> such as COCO, CrowdPose and the PoseTrack 2017 and 2018 datasets. We also show
> that our method generalizes to any class of semantic keypoints such as car and
> animal parts to provide a holistic perception framework that is well suited for
> urban mobility such as self-driving cars and delivery robots.

Previous [CVPR 2019 paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Kreiss_PifPaf_Composite_Fields_for_Human_Pose_Estimation_CVPR_2019_paper.html).


# Example

![example image with overlaid pose predictions](https://github.com/vita-epfl/openpifpaf/raw/main/docs/coco/000000081988.jpg.predictions.jpeg)

Image credit: "[Learning to surf](https://www.flickr.com/photos/fotologic/6038911779/in/photostream/)" by fotologic which is licensed under [CC-BY-2.0].<br />
Created with:
```sh
pip3 install openpifpaf matplotlib
python3 -m openpifpaf.predict docs/coco/000000081988.jpg --image-min-dpi=200 --show-file-extension=jpeg --image-output
```


# [Guide](https://vita-epfl.github.io/openpifpaf/intro.html)

Continue to our __[OpenPifPaf Guide](https://vita-epfl.github.io/openpifpaf/intro.html)__.


[CC-BY-2.0]: https://creativecommons.org/licenses/by/2.0/


# Commercial License

This software is available for licensing via the EPFL Technology Transfer
Office (https://tto.epfl.ch/, info.tto@epfl.ch).
