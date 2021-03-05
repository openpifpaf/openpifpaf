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


def test_conf_ones_weight():
    # Weights = None
    cif = openpifpaf.headmeta.Cif('cif', 'cocokp',
                                  keypoints=COCO_KEYPOINTS,
                                  sigmas=COCO_PERSON_SIGMAS,
                                  pose=COCO_UPRIGHT_POSE,
                                  draw_skeleton=COCO_PERSON_SKELETON,
                                  score_weights=COCO_PERSON_SCORE_WEIGHTS)
    head_net = openpifpaf.network.heads.CompositeField3(cif, 1396)
    loss = openpifpaf.network.losses.composite.CompositeLoss(head_net)
    x = torch.ones((2, 17, 5, 4, 4,)) * 5
    t = torch.ones((2, 17, 5, 4, 4,)) * 4
    loss_values = loss(x, t)
    loss_np = np.array([l.numpy() for l in loss_values])
    correct_values = np.array([0.024449902, 830.4298, 66.78698])  # original loss
    np.testing.assert_allclose(loss_np, correct_values, atol=1e-7)

    # Weights explicitly set to 1.0
    cif = openpifpaf.headmeta.Cif('cif', 'cocokp',
                                  keypoints=COCO_KEYPOINTS,
                                  sigmas=COCO_PERSON_SIGMAS,
                                  pose=COCO_UPRIGHT_POSE,
                                  draw_skeleton=COCO_PERSON_SKELETON,
                                  score_weights=COCO_PERSON_SCORE_WEIGHTS,
                                  training_weights=[1.] * 17)
    head_net = openpifpaf.network.heads.CompositeField3(cif, 1396)
    loss = openpifpaf.network.losses.composite.CompositeLoss(head_net)
    x = torch.ones((2, 17, 5, 4, 4,)) * 5
    t = torch.ones((2, 17, 5, 4, 4,)) * 4
    loss_values = loss(x, t)
    loss_np = np.array([l.numpy() for l in loss_values])
    correct_values = np.array([0.024449902, 830.4298, 66.78698])  # original loss
    np.testing.assert_allclose(loss_np, correct_values, atol=1e-7)


def test_conf_equal_weight():
    w = 1.54  # an arbitary float value
    cif = openpifpaf.headmeta.Cif('cif', 'cocokp',
                                  keypoints=COCO_KEYPOINTS,
                                  sigmas=COCO_PERSON_SIGMAS,
                                  pose=COCO_UPRIGHT_POSE,
                                  draw_skeleton=COCO_PERSON_SKELETON,
                                  score_weights=COCO_PERSON_SCORE_WEIGHTS,
                                  training_weights=[w] * 17)
    head_net = openpifpaf.network.heads.CompositeField3(cif, 1396)
    loss = openpifpaf.network.losses.composite.CompositeLoss(head_net)
    x = torch.ones((2, 17, 5, 4, 4,)) * 5
    t = torch.ones((2, 17, 5, 4, 4,)) * 4
    loss_values = loss(x, t)
    loss_np = np.array([l.numpy() for l in loss_values])
    correct_values = np.array([0.024449902, 830.4298, 66.78698]) * w
    np.testing.assert_allclose(loss_np, correct_values, atol=1e-4)


def test_conf_zero_weight():
    cif = openpifpaf.headmeta.Cif('cif', 'cocokp',
                                  keypoints=COCO_KEYPOINTS,
                                  sigmas=COCO_PERSON_SIGMAS,
                                  pose=COCO_UPRIGHT_POSE,
                                  draw_skeleton=COCO_PERSON_SKELETON,
                                  score_weights=COCO_PERSON_SCORE_WEIGHTS,
                                  training_weights=[0.0] * 17)
    head_net = openpifpaf.network.heads.CompositeField3(cif, 1396)
    loss = openpifpaf.network.losses.composite.CompositeLoss(head_net)
    x = torch.ones((2, 17, 5, 4, 4,)) * 5
    t = torch.ones((2, 17, 5, 4, 4,)) * 4
    loss_values = loss(x, t)
    loss_np = np.array([l.numpy() for l in loss_values])
    np.testing.assert_allclose(loss_np, np.zeros((3)), atol=1e-7)
