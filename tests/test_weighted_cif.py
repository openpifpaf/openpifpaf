import numpy as np
import torch

import openpifpaf

from openpifpaf.plugins.coco.constants import (
    COCO_KEYPOINTS,
    COCO_PERSON_SKELETON,
    COCO_PERSON_SIGMAS,
    COCO_PERSON_SCORE_WEIGHTS,
    COCO_UPRIGHT_POSE,
)


def test_cif_ones_weight():
    cif = openpifpaf.headmeta.Cif('cif', 'cocokp',
                                  keypoints=COCO_KEYPOINTS,
                                  sigmas=COCO_PERSON_SIGMAS)
    x = torch.ones((2, 17, 5, 4, 4,)) * 5
    t = torch.ones((2, 17, 5, 4, 4,)) * 4

    # reference
    loss = openpifpaf.network.losses.composite.CompositeLoss(cif)
    loss_values = loss(x, t)
    loss_np_no_weight = np.array([l.numpy() for l in loss_values])

    # Weights explicitly set to 1.0
    cif.training_weights = [1.] * 17
    loss = openpifpaf.network.losses.composite.CompositeLoss(cif)
    loss_values = loss(x, t)
    loss_np = np.array([l.numpy() for l in loss_values])

    np.testing.assert_allclose(loss_np, loss_np_no_weight, atol=1e-7, rtol=1e-4)


def test_caf_ones_weight():
    x = torch.ones((2, 19, 8, 4, 4,)) * 5
    t = torch.ones((2, 19, 9, 4, 4,)) * 4
    caf = openpifpaf.headmeta.Caf('caf', 'cocokp',
                                  keypoints=COCO_KEYPOINTS,
                                  skeleton=COCO_PERSON_SKELETON,
                                  sigmas=COCO_PERSON_SIGMAS)

    # reference
    loss = openpifpaf.network.losses.composite.CompositeLoss(caf)
    loss_values = loss(x, t)
    loss_np_no_weight = np.array([l.numpy() for l in loss_values])

    # Weights explicitly set to 1.0
    caf.training_weights = [1.] * 19
    loss = openpifpaf.network.losses.composite.CompositeLoss(caf)
    loss_values = loss(x, t)
    loss_np = np.array([l.numpy() for l in loss_values])

    np.testing.assert_allclose(loss_np, loss_np_no_weight, atol=1e-7, rtol=1e-4)


def test_conf_equal_weight():
    # Weights = None
    cif = openpifpaf.headmeta.Cif('cif', 'cocokp',
                                  keypoints=COCO_KEYPOINTS,
                                  sigmas=COCO_PERSON_SIGMAS,
                                  pose=COCO_UPRIGHT_POSE,
                                  draw_skeleton=COCO_PERSON_SKELETON,
                                  score_weights=COCO_PERSON_SCORE_WEIGHTS)
    loss = openpifpaf.network.losses.composite.CompositeLoss(cif)
    x = torch.ones((2, 17, 5, 4, 4,)) * 5
    t = torch.ones((2, 17, 5, 4, 4,)) * 4
    loss_values = loss(x, t)
    loss_np_no_weight = np.array([l.numpy() for l in loss_values])

    w = 1.54  # an arbitrary float value
    cif = openpifpaf.headmeta.Cif('cif', 'cocokp',
                                  keypoints=COCO_KEYPOINTS,
                                  sigmas=COCO_PERSON_SIGMAS,
                                  pose=COCO_UPRIGHT_POSE,
                                  draw_skeleton=COCO_PERSON_SKELETON,
                                  score_weights=COCO_PERSON_SCORE_WEIGHTS,
                                  training_weights=[w] * 17)
    loss = openpifpaf.network.losses.composite.CompositeLoss(cif)
    x = torch.ones((2, 17, 5, 4, 4,)) * 5
    t = torch.ones((2, 17, 5, 4, 4,)) * 4
    loss_values = loss(x, t)
    loss_np = np.array([l.numpy() for l in loss_values])
    np.testing.assert_allclose(loss_np, loss_np_no_weight * w, atol=1e-4, rtol=1e-4)


def test_conf_zero_weight():
    cif = openpifpaf.headmeta.Cif('cif', 'cocokp',
                                  keypoints=COCO_KEYPOINTS,
                                  sigmas=COCO_PERSON_SIGMAS,
                                  pose=COCO_UPRIGHT_POSE,
                                  draw_skeleton=COCO_PERSON_SKELETON,
                                  score_weights=COCO_PERSON_SCORE_WEIGHTS,
                                  training_weights=[0.0] * 17)
    loss = openpifpaf.network.losses.composite.CompositeLoss(cif)
    x = torch.ones((2, 17, 5, 4, 4,)) * 5
    t = torch.ones((2, 17, 5, 4, 4,)) * 4
    loss_values = loss(x, t)
    loss_np = np.array([l.numpy() for l in loss_values])
    np.testing.assert_allclose(loss_np, np.zeros((3)), atol=1e-7, rtol=1e-4)
