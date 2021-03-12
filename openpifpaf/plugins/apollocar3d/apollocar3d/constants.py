import numpy as np

from .transforms import transform_skeleton

CAR_KEYPOINTS = [
    'front_up_right',       # 1
    'front_up_left',        # 2
    'front_light_right',    # 3
    'front_light_left',     # 4
    'front_low_right',      # 5
    'front_low_left',       # 6
    'central_up_left',      # 7
    'front_wheel_left',     # 8
    'rear_wheel_left',      # 9
    'rear_corner_left',     # 10
    'rear_up_left',         # 11
    'rear_up_right',        # 12
    'rear_light_left',      # 13
    'rear_light_right',     # 14
    'rear_low_left',        # 15
    'rear_low_right',       # 16
    'central_up_right',     # 17
    'rear_corner_right',    # 18
    'rear_wheel_right',     # 19
    'front_wheel_right',    # 20
    'rear_plate_left',      # 21
    'rear_plate_right',     # 22
    'mirror_edge_left',     # 23
    'mirror_edge_right',    # 24

]

SKELETON_ORIG = [
    [49, 46], [49, 8], [49, 57], [8, 0], [8, 11], [57, 0], [57, 52], [0, 5], [52, 5], [5, 7],            # frontal
    [7, 20], [11, 23], [20, 23], [23, 25], [34, 32], [9, 11], [9, 7], [9, 20], [7, 0], [9, 0], [9, 8],  # L-lat
    [24, 33], [24, 25], [24, 11], [25, 32], [25, 28], [33, 32], [33, 46], [32, 29], [28, 29],            # rear
    [65, 64], [65, 25], [65, 28], [65, 20], [64, 29], [64, 32], [64, 37],  [29, 37], [28, 20],         # new rear
    [34, 37], [34, 46], [37, 50], [50, 52], [46, 48], [48, 37], [48, 49], [50, 57], [48, 57], [48, 50]  # R-la
    ]


KPS_MAPPING = [49, 8, 57, 0, 52, 5, 11, 7, 20, 23, 24, 33, 25, 32, 28, 29, 46, 34, 37, 50, 65, 64, 9, 48]

CAR_SKELETON = transform_skeleton(SKELETON_ORIG, KPS_MAPPING)

CAR_SIGMAS = [0.05] * len(KPS_MAPPING)

split, error = divmod(len(CAR_KEYPOINTS), 4)
CAR_SCORE_WEIGHTS = [10.0] * split + [3.0] * split + [1.0] * split + [0.1] * split + [0.1] * error
assert len(CAR_SCORE_WEIGHTS) == len(CAR_KEYPOINTS)

HFLIP = {
    'front_up_right': 'front_up_left',
    'front_light_right': 'front_light_left',
    'front_low_right': 'front_low_left',
    'central_up_left': 'central_up_right',
    'front_wheel_left': 'front_wheel_right',
    'rear_wheel_left': 'rear_wheel_right',
    'rear_corner_left': 'rear_corner_right',
    'rear_up_left': 'rear_up_right',
    'rear_light_left': 'rear_light_right',
    'rear_low_left': 'rear_low_right',
    'front_up_left': 'front_up_right',
    'front_light_left': 'front_light_right',
    'front_low_left': 'front_low_right',
    'central_up_right': 'central_up_left',
    'front_wheel_right': 'front_wheel_left',
    'rear_wheel_right': 'rear_wheel_left',
    'rear_corner_right': 'rear_corner_left',
    'rear_up_right': 'rear_up_left',
    'rear_light_right': 'rear_light_left',
    'rear_low_right': 'rear_low_left',
    'rear_plate_left': 'rear_plate_right',
    'rear_plate_right': 'rear_plate_left',
    'mirror_edge_left': 'mirror_edge_right',
    'mirror_edge_right': 'mirror_edge_left'
}

CAR_CATEGORIES = ['car']

p = 0.25

# CAR POSE is used for joint rescaling. x = [-3, 3] y = [0,4]
CAR_POSE = np.array([
    [-2.9, 4.0, 2.0],  # 'front_up_right',              # 1
    [2.9, 4.0, 2.0],   # 'front_up_left',               # 2
    [-2.0, 2.0, 2.0],  # 'front_light_right',           # 3
    [2.0, 2.0, 2.0],  # 'front_light_left',             # 4
    [-2.5, 0.0, 2.0],  # 'front_low_right',             # 5
    [2.5, 0.0,  2.0],  # 'front_low_left',              # 6
    [2.6, 4.2, 2.0],  # 'central_up_left'     # 7
    [3.2, 0.2,  2.0],  # 'front_wheel_left',           # 8
    [3.0, 0.3,  2.0],   # 'rear_wheel_left'      # 9
    [3.1, 2.1, 2.0],   # 'rear_corner_left',          # 10
    [2.4, 4.3, 2.0],  # 'rear_up_left',       # 11
    [-2.4, 4.3, 2.0],  # 'rear_up_right'      # 12
    [2.5, 2.2, 2.0],   # 'rear_light_left',             # 13
    [-2.5, 2.2, 2.0],  # 'rear_light_right',            # 14
    [2.1, 0.1, 2.0],  # 'rear_low_left',            # 15
    [-2.1, 0.1, 2.0],  # 'rear_low_right',          # 16
    [-2.6, 4.2, 2.0],  # 'central_up_right'    # 17
    [-3.1, 2.1, 2.0],  # 'rear_corner_right',         # 18
    [-3.0, 0.3, 2.0],  # 'rear_wheel_right'       # 19
    [-3.2, 0.2, 2.0],  # 'front_wheel_right',          # 20
    [1.0, 1.3, 2.0],  # 'rear_plate_left',              # 21
    [-1.0, 1.3, 2.0],  # 'rear_plate_right',            # 22
    [2.8, 3, 2.0],  # 'mirror_edge_left'          # 23
    [-2.8, 3, 2.0],  # 'mirror_edge_right'        # 24
])

CAR_POSE_FRONT = np.array([
    [-2.0, 4.0, 2.0],  # 'front_up_right',         # 1
    [2.0, 4.0, 2.0],   # 'front_up_left',        # 2
    [-1.3, 2.0, 2.0],  # 'front_light_right',    # 3
    [1.3, 2.0, 2.0],  # 'front_light_left',     # 4
    [-2.2, 0.0, 2.0],  # 'front_low_right',       # 5
    [2.2, 0.0,  2.0],  # 'front_low_left',       # 6
    [2.0 - p/2, 4.0 + p, 1.0],  # 'central_up_left',      # 7
    [2.0 + p, 0.1 - p/2,  1.0],  # 'front_wheel_left',     # 8
    [2, 0.1,  0.0],  # 'rear_wheel_left',      # 9
    [2.6, 1.7, 0.0],   # 'rear_corner_left',          # 10
    [2.0, 4.1, 0.0],  # 'rear_up_left',         # 11
    [-2.0, 4.0, 0.0],  # 'rear_up_right',        # 12
    [2.1, 1.9, 0.0],   # 'rear_light_left',      # 13
    [-2.1, 1.9, 0.0],  # 'rear_right_right',     # 14
    [2.4, 0.1, 0.0],  # 'rear_low_left',        # 15
    [-2.4, 0.1, 0.0],  # 'rear_low_right',       # 16
    [-2.0 + p/2, 4.0 + p, 1.0],  # 'central_up_right',     # 17
    [-2.6, 1.75, 0.0],  # 'rear_corner_right',           # 18
    [-2, 0.0, 0.0],  # 'rear_wheel_right',     # 19
    [-2 - p, 0.0 - p/2, 1.0],  # 'front_wheel_right',     # 20
])

CAR_POSE_REAR = np.array([
    [-2.0, 4.0, 0.0],  # 'front_up_right',         # 1
    [2.0, 4.0, 0.0],   # 'front_up_left',        # 2
    [-1.3, 2.0, 0.0],  # 'front_light_right',    # 3
    [1.3, 2.0, 0.0],  # 'front_light_left',     # 4
    [-2.2, 0.0, 0.0],  # 'front_low_right',       # 5
    [2.2, 0.0,  0.0],  # 'front_low_left',       # 6
    [-2.0+p, 4.0+p, 2.0],  # 'central_up_left',      # 7
    [2, 0.0,  0.0],  # 'front_wheel_left',     # 8
    [2, 0.0,  0.0],  # 'rear_wheel_left',      # 9
    [-1.6-p, 2.2-p, 2.0],   # 'rear_corner_left',     # 10
    [-2.0, 4.0, 2.0],  # 'rear_up_left',         # 11
    [2.0, 4.0, 2.0],  # 'rear_up_right',        # 12
    [-1.6, 2.2, 2.0],   # 'rear_light_left',      # 13
    [1.6, 2.2, 2.0],  # 'rear_right_right',     # 14
    [-2.4, 0.0, 2.0],  # 'rear_low_left',        # 15
    [2.4, 0.0, 2.0],  # 'rear_low_right',       # 16
    [2.0-p, 4.0+p, 2.0],  # 'central_up_right',     # 17
    [1.6+p, 2.2-p, 2.0],  # 'rear_corner_right', # 18
    [-2, 0.0, 0.0],  # 'rear_wheel_right',     # 19
    [-2, 0.0, 0.0],  # 'front_wheel_right',     # 20
])

CAR_POSE_LEFT = np.array([
    [-2.0, 4.0, 0.0],  # 'front_up_right',         # 1
    [0 - 5*p, 4.0 - p/2, 2.0],   # 'front_up_left',        # 2
    [-1.3, 2.0, 0.0],  # 'front_light_right',    # 3
    [1.3, 2.0, 0.0],  # 'front_light_left',     # 4
    [-2.2, 0.0, 0.0],  # 'front_low_right',       # 5
    [-4-3*p, 0.0,  2.0],   # 'front_low_left',       # 6
    [0, 4.0, 2.0],  # 'central_up_left',      # 7
    [-4, 0.0,  2.0],  # 'front_wheel_left',     # 8
    [4, 0.0,  2.0],  # 'rear_wheel_left',      # 9
    [5, 2, 2.0],  # 'rear_corner_left',     # 10
    [0 + 5*p, 4.0-p/2, 2.0],  # 'rear_up_left',  # 11
    [2.0, 4.0, 0.0],  # 'rear_up_right',        # 12
    [5+p, 2+p, 1.0],   # 'rear_light_left',      # 13
    [1.6, 2.2, 0.0],  # 'rear_right_right',     # 14
    [-2.4, 0.0, 0.0],  # 'rear_low_left',        # 15
    [2.4, 0.0, 0.0],  # 'rear_low_right',       # 16
    [2.0, 4.0, 0.0],  # 'central_up_right',     # 17
    [1.6, 2.2, 0.0],  # 'rear_corner_right', # 18
    [-2, 0.0, 0.0],  # 'rear_wheel_right',     # 19
    [-2, 0.0, 0.0],  # 'front_wheel_right',     # 20
])


CAR_POSE_RIGHT = np.array([
    [0 + 5*p, 4.0-p/2, 2.0],  # 'front_up_right',         # 1
    [0, 4.0, 0.0],   # 'front_up_left',        # 2
    [-1.3, 2.0, 0.0],  # 'front_light_right',    # 3
    [1.3, 2.0, 0.0],  # 'front_light_left',     # 4
    [4 + 3*p, 0.0,  2.0],  # 'front_low_right',       # 5
    [-4-3, 0.0,  0.0],   # 'front_low_left',       # 6
    [0, 4.0, 0.0],  # 'central_up_left',      # 7
    [-4, 0.0,  0.0],  # 'front_wheel_left',     # 8
    [4, 0.0,  0.0],  # 'rear_wheel_left',      # 9
    [5, 2, 0.0],  # 'rear_corner_left',     # 10
    [0 + 5, 4.0, 0.0],  # 'rear_up_left',  # 11
    [0 - 5*p, 4.0-p/2, 2.0],  # 'rear_up_right',        # 12
    [5, 2, 0.0],   # 'rear_light_left',      # 13
    [-5-p, 2.0+p, 2.0],  # 'rear_light_right',     # 14
    [-2.4, 0.0, 0.0],  # 'rear_low_left',        # 15
    [2.4, 0.0, 0.0],  # 'rear_low_right',       # 16
    [0.0, 4.0, 2.0],  # 'central_up_right',     # 17
    [-5, 2.0, 2.0],  # 'rear_corner_right', # 18
    [-4, 0.0, 2.0],  # 'rear_wheel_right',     # 19
    [4, 0.0, 2.0],  # 'front_wheel_right',     # 20
])


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
    keypoint_painter = show.KeypointPainter()

    ann = Annotation(keypoints=CAR_KEYPOINTS, skeleton=CAR_SKELETON, score_weights=CAR_SCORE_WEIGHTS)
    ann.set(pose, np.array(CAR_SIGMAS) * scale)
    draw_ann(ann, filename='docs/skeleton_car.png', keypoint_painter=keypoint_painter)


def print_associations():
    for j1, j2 in CAR_SKELETON:
        print(CAR_KEYPOINTS[j1 - 1], '-', CAR_KEYPOINTS[j2 - 1])


if __name__ == '__main__':
    print_associations()

    # c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
    # rotate = np.array(((c, -s), (s, c)))
    # rotated_pose = np.copy(COCO_DAVINCI_POSE)
    # rotated_pose[:, :2] = np.einsum('ij,kj->ki', rotate, rotated_pose[:, :2])
    draw_skeletons(CAR_POSE)
    # draw_skeletons(CAR_POSE_FRONT)
    # draw_skeletons(CAR_POSE_REAR)
    # draw_skeletons(CAR_POSE_RIGHT)
    # draw_skeletons(CAR_POSE_LEFT)
