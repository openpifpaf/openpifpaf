openpifpaf
==========

  We propose a new bottom-up method for multi-person 2D human pose estimation that is particularly well suited for self-driving cars and social robots. The new method, PifPaf, uses a Part Intensity Field (PIF) to localize body parts precisely, and a composite field of two vectors and a scalar - the Part Association Field (PAF) - to associate body parts with each other to form full human poses. It improves on previous methods at low resolution and in crowded, cluttered and occluded scenes. Our architecture is based on a fully convolutional, single-shot, box-free design. We produce state-of-the-art results on a modified COCO keypoint task for the transportation domain.

.. code-block::

  @article{
      PifPaf: Association Fields for Human Pose Estimation
  }


Install
-------

Create a virtualenv. Use ``--system-site-packages`` for OpenCV3 access
for ``openpifpaf.webcam``.

.. code-block:: sh

  python3 -m venv venv3 --system-site-packages


Inside virtualenv, install with optional dependencies:

.. code-block:: sh

  pip install numpy cython
  pip install 'openpifpaf[train,test]'

  # from source:
  pip install --editable '.[train,test]'


Interfaces
----------

* ``python -m openpifpaf.train --help``
* ``python -m openpifpaf.eval_coco --help``
* ``python -m openpifpaf.logs --help``
* ``python -m openpifpaf.predict --help``
* ``python -m openpifpaf.webcam --help``


Pre-trained Networks
--------------------

Put these files into your ``outputs`` folder: `Google Drive <https://drive.google.com/drive/folders/1v8UNDjZbqeMZY64T33tSDOq1jtcBJBy7?usp=sharing>`_

Visualize logs:

.. code-block::

  python -m pifpaf.logs \
    outputs/resnet50-pif-paf-rsmooth0.5-181209-192001.pkl.log \
    outputs/resnet101-pif-paf-rsmooth0.5-181213-224234.pkl.log \
    outputs/resnet152-pif-paf-l1-181230-201001.pkl.log


Train
-----

See `datasets <docs/datasets.md>`_ for setup instructions.
See `studies.ipynb <docs/studies.ipynb>`_ for previous studies.

Train a model:

.. code-block::

  python -m openpifpaf.train

  # or refine a pre-trained model
  python -m openpifpaf.train \
    --lr=1e-3 \
    --epochs=60 \
    --batch-size=8 \
    --basenet=resnet101 \
    --square-edge=321 \
    --regression-loss=l1 \
    --lambdas 30 1 30 1 1 \
    --freeze-base=1


Every 5 minutes, check the directory for new snapshots to evaluate:

.. code-block:: sh

  while true; do \
    CUDA_VISIBLE_DEVICES=0 find outputs/ -name "resnet101block5-pif-paf-l1-190109-113346.pkl.epoch???" -exec \
      python -m openpifpaf.eval_coco --checkpoint {} -n 500 --long-edge=641 --skip-existing \; \
    ; \
    sleep 300; \
  done



Demo
----

.. code-block:: sh

  python -m openpifpaf.predict \
    --checkpoint outputs/resnet101block5-pifs-pafs-edge401-l1-190131-083451.pkl \
    data-mscoco/images/val2017/000000081988.jpg -o docs/coco/ --show

Result:

.. figure:: docs/coco/000000081988.jpg.skeleton.png

  Image credit: "`Learning to surf <https://www.flickr.com/photos/fotologic/6038911779/in/photostream/>`_" by fotologic which is licensed under CC-BY-2.0_.

Processing a video from `video.avi` to `video-pose.mp4`:

.. code-block:: sh

    ffmpeg -i video.avi -qscale:v 2 -vf scale=641:-1 -f image2 video-%05d.jpg
    python -m openpifpaf.predict --checkpoint outputs/resnet101block5-pifs-pafs-edge401-l1-190213-100439.pkl video-*0.jpg
    ffmpeg -framerate 24 -pattern_type glob -i 'video-*.jpg.skeleton.png' -vf scale=640:-1 -c:v libx264 -pix_fmt yuv420p video-pose.mp4


Person Skeletons
----------------

COCO / kinematic tree / dense:

+----------------------+------------------------+-----------------------------+
| |COCO skeleton|      | |KinTree skeleton|     | |Dense skeleton|            |
+----------------------+------------------------+-----------------------------+

.. |COCO skeleton| image:: docs/skeleton_coco.png
  :height: 250px

.. |KinTree skeleton| image:: docs/skeleton_kinematic_tree.png
  :height: 250px

.. |Dense skeleton| image:: docs/skeleton_dense.png
  :height: 250px

Created with ``python -m openpifpaf.data``.


Evaluations
-----------

See `evaluation logs <docs/eval_logs.md>`_ for a long list.
This result was produced with ``python -m openpifpaf.eval_coco --checkpoint outputs/resnet152-pif-paf-l1-181230-201001.pkl --long-edge=641``:

.. code-block::

  removed outdated info



.. _CC-BY-2.0: https://creativecommons.org/licenses/by/2.0/
