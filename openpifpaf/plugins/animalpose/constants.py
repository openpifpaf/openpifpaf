
import os

import numpy as np

_CATEGORIES = ['cat', 'cow', 'dog', 'sheep', 'horse']  # only for preprocessing

ANIMAL_KEYPOINTS = [
    'Nose',         # 1
    'L_eye',        # 2
    'R_eye',        # 3
    'L_ear',        # 4
    'R_ear',        # 5
    'Throat',       # 6
    'Tail',         # 7
    'withers',      # 8
    'L_F_elbow',    # 9
    'R_F_elbow',    # 10
    'L_B_elbow',    # 11
    'R_B_elbow',    # 12
    'L_F_knee',     # 13
    'R_F_knee',     # 14
    'L_B_knee',     # 15
    'R_B_knee',     # 16
    'L_F_paw',      # 17
    'R_F_paw',      # 18
    'L_B_paw',      # 19
    'R_B_paw',      # 20
]


HFLIP = {
    'L_eye': 'R_eye',
    'R_eye': 'L_eye',
    'L_ear': 'R_ear',
    'R_ear': 'L_ear',
    'L_F_elbow': 'R_F_elbow',
    'R_F_elbow': 'L_F_elbow',
    'L_B_elbow': 'R_B_elbow',
    'R_B_elbow': 'L_B_elbow',
    'L_F_knee': 'R_F_knee',
    'R_F_knee': 'L_F_knee',
    'L_B_knee': 'R_B_knee',
    'R_B_knee': 'L_B_knee',
    'L_F_paw': 'R_F_paw',
    'R_F_paw': 'L_F_paw',
    'L_B_paw': 'R_B_paw',
    'R_B_paw': 'L_B_paw',
}


ALTERNATIVE_NAMES = [
    'Nose',         # 1
    'L_Eye',        # 2
    'R_Eye',        # 3
    'L_EarBase',    # 4
    'R_EarBase',    # 5
    'Throat',       # 6
    'TailBase',     # 7
    'Withers',      # 8
    'L_F_Elbow',    # 9
    'R_F_Elbow',    # 10
    'L_B_Elbow',    # 11
    'R_B_Elbow',    # 12
    'L_F_Knee',     # 13
    'R_F_Knee',     # 14
    'L_B_Knee',     # 15
    'R_B_Knee',     # 16
    'L_F_Paw',      # 17
    'R_F_Paw',      # 18
    'L_B_Paw',      # 19
    'R_B_Paw',      # 20
]


ANIMAL_SKELETON = [
    (1, 2), (1, 3), (1, 6), (2, 4), (3, 5), (2, 3), (6, 8), (6, 9), (6, 10), (7, 8),
    (7, 11), (7, 12),  # Torso + Face
    (10, 14), (14, 18), (9, 13), (13, 17), (12, 16), (16, 20), (11, 15), (15, 19)  # Legs
]


ANIMAL_SIGMAS = [
    0.026,  # nose
    0.025,  # eyes
    0.025,  # eyes
    0.035,  # ears
    0.035,  # ears
    0.079,  # throat
    0.079,  # tail
    0.079,  # withers
    0.072,  # elbows
    0.072,  # elbows
    0.072,  # elbows
    0.072,  # elbows
    0.087,  # knees
    0.087,  # knees
    0.087,  # knees
    0.087,  # knees
    0.089,  # ankles
    0.089,  # ankles
    0.089,  # ankles
    0.089,  # ankles
]

split, error = divmod(len(ANIMAL_KEYPOINTS), 4)
ANIMAL_SCORE_WEIGHTS = [5.0] * split + [3.0] * split + [1.0] * split + [0.5] * split + [0.1] * error

ANIMAL_CATEGORIES = ['animal']

ANIMAL_POSE = np.array([
    [0.0, 4.3, 2.0],  # 'nose',            # 1
    [-0.4, 4.7, 2.0],  # 'left_eye',        # 2
    [0.4, 4.7, 2.0],  # 'right_eye',       # 3
    [-0.7, 5.0, 2.0],  # 'left_ear',        # 4
    [0.7, 5.0, 2.0],  # 'right_ear',       # 5
    [0.2, 3.0, 2.0],  # 'throat',            # 6
    [6.7, 3.8, 2.0],  # 'tail',             # 7
    [0.8, 4.0, 2.0],  # 'withers',         # 8
    [1.0, 2.0, 2.0],  # 'L_F_elbow',      # 9
    [0.6, 2.2, 2.0],  # 'R_F_elbow',     # 10
    [5.8, 2.1, 2.0],  # 'L_B_elbow',      # 11
    [6.3, 2.3, 2.0],  # 'R_B_elbow',     # 12
    [0.8, 0.8, 2.0],  # 'L_F_Knee',     # 13
    [0.4, 1.0, 2.0],  # 'R_F_Knee',     # 14
    [6.0, 0.9, 2.0],  # 'L_B_Knee',     # 15
    [6.5, 1.1, 2.0],  # 'R_B_Knee',     # 16
    [1.0, 0.0, 2.0],  # 'L_F_Paw',      # 17
    [0.6, 0.2, 2.0],  # 'R_F_Paw',      # 18
    [6.0, 0.1, 2.0],  # 'L_B_Paw',      # 19
    [6.5, 0.3, 2.0],  # 'R_B_Paw',      # 20
])


assert len(ANIMAL_POSE) == len(ANIMAL_KEYPOINTS) == len(ALTERNATIVE_NAMES) == len(ANIMAL_SIGMAS) \
       == len(ANIMAL_SCORE_WEIGHTS), "dimensions!"


def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    bbox = ann.bbox()
    xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
    ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
    if aspect == 'equal':
        fig_w = 5.0
    else:
        fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])

    with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if aspect is not None:
            ax.set_aspect(aspect)

        keypoint_painter.annotation(ax, ann)


def draw_skeletons(pose):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    show.KeypointPainter.show_joint_scales = True
    show.KeypointPainter.font_size = 0
    keypoint_painter = show.KeypointPainter()

    ann = Annotation(
        keypoints=ANIMAL_KEYPOINTS, skeleton=ANIMAL_SKELETON, score_weights=ANIMAL_SCORE_WEIGHTS)
    ann.set(pose, np.array(ANIMAL_SIGMAS) * scale)
    os.makedirs('all-images', exist_ok=True)
    draw_ann(ann, filename='all-images/skeleton_animal.png', keypoint_painter=keypoint_painter)


def print_associations():
    for j1, j2 in ANIMAL_SKELETON:
        print(ANIMAL_SKELETON[j1 - 1], '-', ANIMAL_KEYPOINTS[j2 - 1])


if __name__ == '__main__':
    print_associations()
    draw_skeletons(ANIMAL_POSE)
