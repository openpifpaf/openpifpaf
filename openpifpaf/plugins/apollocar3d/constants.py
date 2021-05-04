import os

import numpy as np
try:
    import matplotlib.cm as mplcm
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    pass

import openpifpaf

from .transforms import transform_skeleton

CAR_KEYPOINTS_24 = [
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
    [49, 46], [49, 8], [49, 57], [8, 0], [8, 11], [57, 0],
    [57, 52], [0, 5], [52, 5], [5, 7],  # frontal
    [7, 20], [11, 23], [20, 23], [23, 25], [34, 32],
    [9, 11], [9, 7], [9, 20], [7, 0], [9, 0], [9, 8],  # L-lat
    [24, 33], [24, 25], [24, 11], [25, 32], [25, 28],
    [33, 32], [33, 46], [32, 29], [28, 29],  # rear
    [65, 64], [65, 25], [65, 28], [65, 20], [64, 29],
    [64, 32], [64, 37], [29, 37], [28, 20],  # new rear
    [34, 37], [34, 46], [37, 50], [50, 52], [46, 48], [48, 37],
    [48, 49], [50, 57], [48, 57], [48, 50]
]


KPS_MAPPING = [49, 8, 57, 0, 52, 5, 11, 7, 20, 23, 24, 33, 25, 32, 28,
               29, 46, 34, 37, 50, 65, 64, 9, 48]

CAR_SKELETON_24 = transform_skeleton(SKELETON_ORIG, KPS_MAPPING)

CAR_SIGMAS_24 = [0.05] * len(KPS_MAPPING)

split, error = divmod(len(CAR_KEYPOINTS_24), 4)
CAR_SCORE_WEIGHTS_24 = [10.0] * split + [3.0] * split + \
    [1.0] * split + [0.1] * split + [0.1] * error
assert len(CAR_SCORE_WEIGHTS_24) == len(CAR_KEYPOINTS_24)

HFLIP_24 = {
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

CAR_CATEGORIES_24 = ['car']

p = 0.25
FRONT = -6.0
BACK = 4.5

# CAR POSE is used for joint rescaling. x = [-3, 3] y = [0,4]
CAR_POSE_24 = np.array([
    [-2.9, 4.0, FRONT * 0.5],  # 'front_up_right',              # 1
    [2.9, 4.0, FRONT * 0.5],   # 'front_up_left',               # 2
    [-2.0, 2.0, FRONT],  # 'front_light_right',           # 3
    [2.0, 2.0, FRONT],  # 'front_light_left',             # 4
    [-2.5, 0.0, FRONT],  # 'front_low_right',             # 5
    [2.5, 0.0, FRONT],  # 'front_low_left',              # 6
    [2.6, 4.2, 0.0],  # 'central_up_left'     # 7
    [3.2, 0.2, FRONT * 0.7],  # 'front_wheel_left',           # 8
    [3.0, 0.3, BACK * 0.7],   # 'rear_wheel_left'      # 9
    [3.1, 2.1, BACK * 0.5],   # 'rear_corner_left',          # 10
    [2.4, 4.3, BACK * 0.35],  # 'rear_up_left',       # 11
    [-2.4, 4.3, BACK * 0.35],  # 'rear_up_right'      # 12
    [2.5, 2.2, BACK],   # 'rear_light_left',             # 13
    [-2.5, 2.2, BACK],  # 'rear_light_right',            # 14
    [2.1, 0.1, BACK],  # 'rear_low_left',            # 15
    [-2.1, 0.1, BACK],  # 'rear_low_right',          # 16
    [-2.6, 4.2, 0.0],  # 'central_up_right'    # 17
    [-3.1, 2.1, BACK * 0.5],  # 'rear_corner_right',         # 18
    [-3.0, 0.3, BACK * 0.7],  # 'rear_wheel_right'       # 19
    [-3.2, 0.2, FRONT * 0.7],  # 'front_wheel_right',          # 20
    [1.0, 1.3, BACK],  # 'rear_plate_left',              # 21
    [-1.0, 1.3, BACK],  # 'rear_plate_right',            # 22
    [2.8, 3, FRONT * 0.35],  # 'mirror_edge_left'          # 23
    [-2.8, 3, FRONT * 0.35],  # 'mirror_edge_right'        # 24
])

CAR_POSE_FRONT_24 = np.array([
    [-2.0, 4.0, 2.0],  # 'front_up_right',         # 1
    [2.0, 4.0, 2.0],   # 'front_up_left',        # 2
    [-1.3, 2.0, 2.0],  # 'front_light_right',    # 3
    [1.3, 2.0, 2.0],  # 'front_light_left',     # 4
    [-2.2, 0.0, 2.0],  # 'front_low_right',       # 5
    [2.2, 0.0, 2.0],  # 'front_low_left',       # 6
    [2.0 - p / 2, 4.0 + p, 1.0],  # 'central_up_left',      # 7
    [2.0 + p, 0.1 - p / 2, 1.0],  # 'front_wheel_left',     # 8
    [2, 0.1, 0.0],  # 'rear_wheel_left',      # 9
    [2.6, 1.7, 0.0],   # 'rear_corner_left',          # 10
    [2.0, 4.1, 0.0],  # 'rear_up_left',         # 11
    [-2.0, 4.0, 0.0],  # 'rear_up_right',        # 12
    [2.1, 1.9, 0.0],   # 'rear_light_left',      # 13
    [-2.1, 1.9, 0.0],  # 'rear_right_right',     # 14
    [2.4, 0.1, 0.0],  # 'rear_low_left',        # 15
    [-2.4, 0.1, 0.0],  # 'rear_low_right',       # 16
    [-2.0 + p / 2, 4.0 + p, 1.0],  # 'central_up_right',     # 17
    [-2.6, 1.75, 0.0],  # 'rear_corner_right',           # 18
    [-2, 0.0, 0.0],  # 'rear_wheel_right',     # 19
    [-2 - p, 0.0 - p / 2, 1.0],  # 'front_wheel_right',     # 20
])

CAR_POSE_REAR_24 = np.array([
    [-2.0, 4.0, 0.0],  # 'front_up_right',         # 1
    [2.0, 4.0, 0.0],   # 'front_up_left',        # 2
    [-1.3, 2.0, 0.0],  # 'front_light_right',    # 3
    [1.3, 2.0, 0.0],  # 'front_light_left',     # 4
    [-2.2, 0.0, 0.0],  # 'front_low_right',       # 5
    [2.2, 0.0, 0.0],  # 'front_low_left',       # 6
    [-2.0 + p, 4.0 + p, 2.0],  # 'central_up_left',      # 7
    [2, 0.0, 0.0],  # 'front_wheel_left',     # 8
    [2, 0.0, 0.0],  # 'rear_wheel_left',      # 9
    [-1.6 - p, 2.2 - p, 2.0],   # 'rear_corner_left',     # 10
    [-2.0, 4.0, 2.0],  # 'rear_up_left',         # 11
    [2.0, 4.0, 2.0],  # 'rear_up_right',        # 12
    [-1.6, 2.2, 2.0],   # 'rear_light_left',      # 13
    [1.6, 2.2, 2.0],  # 'rear_right_right',     # 14
    [-2.4, 0.0, 2.0],  # 'rear_low_left',        # 15
    [2.4, 0.0, 2.0],  # 'rear_low_right',       # 16
    [2.0 - p, 4.0 + p, 2.0],  # 'central_up_right',     # 17
    [1.6 + p, 2.2 - p, 2.0],  # 'rear_corner_right', # 18
    [-2, 0.0, 0.0],  # 'rear_wheel_right',     # 19
    [-2, 0.0, 0.0],  # 'front_wheel_right',     # 20
])

CAR_POSE_LEFT_24 = np.array([
    [-2.0, 4.0, 0.0],  # 'front_up_right',         # 1
    [0 - 5 * p, 4.0 - p / 2, 2.0],   # 'front_up_left',        # 2
    [-1.3, 2.0, 0.0],  # 'front_light_right',    # 3
    [1.3, 2.0, 0.0],  # 'front_light_left',     # 4
    [-2.2, 0.0, 0.0],  # 'front_low_right',       # 5
    [-4 - 3 * p, 0.0, 2.0],   # 'front_low_left',       # 6
    [0, 4.0, 2.0],  # 'central_up_left',      # 7
    [-4, 0.0, 2.0],  # 'front_wheel_left',     # 8
    [4, 0.0, 2.0],  # 'rear_wheel_left',      # 9
    [5, 2, 2.0],  # 'rear_corner_left',     # 10
    [0 + 5 * p, 4.0 - p / 2, 2.0],  # 'rear_up_left',  # 11
    [2.0, 4.0, 0.0],  # 'rear_up_right',        # 12
    [5 + p, 2 + p, 1.0],   # 'rear_light_left',      # 13
    [1.6, 2.2, 0.0],  # 'rear_right_right',     # 14
    [-2.4, 0.0, 0.0],  # 'rear_low_left',        # 15
    [2.4, 0.0, 0.0],  # 'rear_low_right',       # 16
    [2.0, 4.0, 0.0],  # 'central_up_right',     # 17
    [1.6, 2.2, 0.0],  # 'rear_corner_right', # 18
    [-2, 0.0, 0.0],  # 'rear_wheel_right',     # 19
    [-2, 0.0, 0.0],  # 'front_wheel_right',     # 20
])


CAR_POSE_RIGHT_24 = np.array([
    [0 + 5 * p, 4.0 - p / 2, 2.0],  # 'front_up_right',         # 1
    [0, 4.0, 0.0],   # 'front_up_left',        # 2
    [-1.3, 2.0, 0.0],  # 'front_light_right',    # 3
    [1.3, 2.0, 0.0],  # 'front_light_left',     # 4
    [4 + 3 * p, 0.0, 2.0],  # 'front_low_right',       # 5
    [-4 - 3, 0.0, 0.0],   # 'front_low_left',       # 6
    [0, 4.0, 0.0],  # 'central_up_left',      # 7
    [-4, 0.0, 0.0],  # 'front_wheel_left',     # 8
    [4, 0.0, 0.0],  # 'rear_wheel_left',      # 9
    [5, 2, 0.0],  # 'rear_corner_left',     # 10
    [0 + 5, 4.0, 0.0],  # 'rear_up_left',  # 11
    [0 - 5 * p, 4.0 - p / 2, 2.0],  # 'rear_up_right',        # 12
    [5, 2, 0.0],   # 'rear_light_left',      # 13
    [-5 - p, 2.0 + p, 2.0],  # 'rear_light_right',     # 14
    [-2.4, 0.0, 0.0],  # 'rear_low_left',        # 15
    [2.4, 0.0, 0.0],  # 'rear_low_right',       # 16
    [0.0, 4.0, 2.0],  # 'central_up_right',     # 17
    [-5, 2.0, 2.0],  # 'rear_corner_right', # 18
    [-4, 0.0, 2.0],  # 'rear_wheel_right',     # 19
    [4, 0.0, 2.0],  # 'front_wheel_right',     # 20
])

CAR_KEYPOINTS_66 = [
    "top_left_c_left_front_car_light",      # 0
    "bottom_left_c_left_front_car_light",   # 1
    "top_right_c_left_front_car_light",     # 2
    "bottom_right_c_left_front_car_light",  # 3
    "top_right_c_left_front_fog_light",     # 4
    "bottom_right_c_left_front_fog_light",  # 5
    "front_section_left_front_wheel",       # 6
    "center_left_front_wheel",              # 7
    "top_right_c_front_glass",              # 8
    "top_left_c_left_front_door",           # 9
    "bottom_left_c_left_front_door",        # 10
    "top_right_c_left_front_door",          # 11
    "middle_c_left_front_door",             # 12
    "front_c_car_handle_left_front_door",   # 13
    "rear_c_car_handle_left_front_door",    # 14
    "bottom_right_c_left_front_door",       # 15
    "top_right_c_left_rear_door",           # 16
    "front_c_car_handle_left_rear_door",    # 17
    "rear_c_car_handle_left_rear_door",     # 18
    "bottom_right_c_left_rear_door",        # 19
    "center_left_rear_wheel",               # 20
    "rear_section_left_rear_wheel",         # 21
    "top_left_c_left_rear_car_light",       # 22
    "bottom_left_c_left_rear_car_light",    # 23
    "top_left_c_rear_glass",                # 24
    "top_right_c_left_rear_car_light",      # 25
    "bottom_right_c_left_rear_car_light",   # 26
    "bottom_left_c_trunk",                  # 27
    "Left_c_rear_bumper",                   # 28
    "Right_c_rear_bumper",                  # 29
    "bottom_right_c_trunk",                 # 30
    "bottom_left_c_right_rear_car_light",   # 31
    "top_left_c_right_rear_car_light",      # 32
    "top_right_c_rear_glass",               # 33
    "bottom_right_c_right_rear_car_light",  # 34
    "top_right_c_right_rear_car_light",     # 35
    "rear_section_right_rear_wheel",        # 36
    "center_right_rear_wheel",              # 37
    "bottom_left_c_right_rear_car_door",    # 38
    "rear_c_car_handle_right_rear_car_door",    # 39
    "front_c_car_handle_right_rear_car_door",   # 40
    "top_left_c_right_rear_car_door",       # 41
    "bottom_left_c_right_front_car_door",   # 42
    "rear_c_car_handle_right_front_car_door",   # 43
    "front_c_car_handle_right_front_car_door",  # 44
    "middle_c_right_front_car_door",        # 45
    "top_left_c_right_front_car_door",      # 46
    "bottom_right_c_right_front_car_door",  # 47
    "top_right_c_right_front_car_door",     # 48
    "top_left_c_front_glass",               # 49
    "center_right_front_wheel",             # 50
    "front_section_right_front_wheel",      # 51
    "bottom_left_c_right_fog_light",        # 52
    "top_left_c_right_fog_light",           # 53
    "bottom_left_c_right_front_car_light",  # 54
    "top_left_c_right_front_car_light",     # 55
    "bottom_right_c_right_front_car_light",  # 56
    "top_right_c_right_front_car_light",     # 57
    "top_right_c_front_lplate",             # 58
    "top_left_c_front_lplate",              # 59
    "bottom_right_c_front_lplate",           # 60
    "bottom_left_c_front_lplate",          # 61
    "top_left_c_rear_lplate",               # 62
    "top_right_c_rear_lplate",              # 63
    "bottom_right_c_rear_lplate",           # 64
    "bottom_left_c_rear_lplate", ]            # 65


HFLIP_ids = {
    0: 57,
    1: 56,
    2: 55,
    3: 54,
    4: 53,
    5: 52,
    6: 51,
    7: 50,
    8: 49,
    9: 48,
    10: 47,
    11: 46,
    12: 45,
    13: 44,
    14: 43,
    15: 42,
    16: 41,
    17: 40,
    18: 39,
    19: 38,
    20: 37,
    21: 36,
    22: 35,
    23: 34,
    24: 33,
    25: 32,
    26: 31,
    27: 30,
    28: 29,
    59: 58,
    61: 60,
    62: 63,
    65: 64
}

HFLIP_66 = {}
checklist = []
for ind in HFLIP_ids:
    HFLIP_66[CAR_KEYPOINTS_66[ind]] = CAR_KEYPOINTS_66[HFLIP_ids[ind]]
    HFLIP_66[CAR_KEYPOINTS_66[HFLIP_ids[ind]]] = CAR_KEYPOINTS_66[ind]
    checklist.append(ind)
    checklist.append(HFLIP_ids[ind])
assert sorted(checklist) == list(range(len(CAR_KEYPOINTS_66)))
assert len(HFLIP_66) == len(CAR_KEYPOINTS_66)

CAR_CATEGORIES_66 = ['car']

SKELETON_LEFT = [
    [59, 61], [59, 1], [61, 5], [0, 1], [0, 2], [2, 3], [3, 1], [3, 4], [4, 5],  # front
    [5, 6], [6, 7], [4, 7], [2, 9], [9, 8], [8, 11], [7, 10], [6, 10], [9, 10],  # side front part
    [11, 12], [11, 24], [9, 12], [10, 15], [12, 15],
    [9, 13], [13, 14], [14, 12], [14, 15],  # side middle part
    [24, 16], [12, 16], [12, 17], [17, 18], [18, 16],
    [15, 19], [19, 20], [19, 18], [20, 21], [16, 21],  # side back part
    [16, 22], [21, 28], [22, 23], [23, 28], [22, 25], [25, 26],
    [23, 26], [26, 27], [25, 62], [27, 65], [62, 65], [28, 65]]

SKELETON_RIGHT = [[HFLIP_ids[bone[0]], HFLIP_ids[bone[1]]] for bone in SKELETON_LEFT]

SKELETON_CONNECT = [
    [28, 29], [62, 63], [65, 64], [24, 33], [46, 11],
    [48, 9], [59, 58], [60, 61], [0, 57], [49, 8]]

SKELETON_ALL = SKELETON_LEFT + SKELETON_RIGHT + SKELETON_CONNECT

CAR_SKELETON_66 = [(bone[0] + 1, bone[1] + 1) for bone in SKELETON_ALL]  # COCO style skeleton

CAR_SIGMAS_66 = [0.05] * len(CAR_KEYPOINTS_66)

split, error = divmod(len(CAR_KEYPOINTS_66), 4)
CAR_SCORE_WEIGHTS_66 = [10.0] * split + [3.0] * split + \
    [1.0] * split + [0.1] * split + [0.1] * error
assert len(CAR_SCORE_WEIGHTS_66) == len(CAR_KEYPOINTS_66)


# number plate offsets
P_X = 0.3
P_Y_TOP = -0.2
P_Y_BOTTOM = -0.4

# z for front
FRONT_Z = -2.0
FRONT_Z_SIDE = -1.8
FRONT_Z_CORNER = -1.7
FRONT_Z_WHEEL = -1.4
FRONT_Z_DOOR = -1.0

# lights x offset
LIGHT_X_INSIDE = 0.8
X_OUTSIDE = 1.0

# y offsets
TOP_CAR = 0.5
BOTTOM_LINE = -0.75
TOP_LINE = 0.1

# z for the back
BACK_Z_WHEEL = 1.0
BACK_Z = 1.5
BACK_Z_SIDE = 1.3

CAR_POSE_HALF = np.array([
    [-LIGHT_X_INSIDE, 0.0, FRONT_Z],    # 0
    [-LIGHT_X_INSIDE, -0.2, FRONT_Z],  # 1
    [-X_OUTSIDE, 0.0, FRONT_Z_SIDE],  # 2
    [-X_OUTSIDE, -0.2, FRONT_Z_SIDE],  # 3
    [-X_OUTSIDE, P_Y_BOTTOM, FRONT_Z_SIDE],  # 4
    [-X_OUTSIDE, P_Y_BOTTOM - 0.2, FRONT_Z_SIDE],  # 5
    [-X_OUTSIDE, BOTTOM_LINE, FRONT_Z_CORNER],  # 6
    [-X_OUTSIDE, BOTTOM_LINE + 0.1, FRONT_Z_WHEEL],  # 7
    [-X_OUTSIDE + 0.1, TOP_CAR, FRONT_Z_DOOR + 0.5],  # 8
    [-X_OUTSIDE, TOP_LINE, FRONT_Z_DOOR],  # 9
    [-X_OUTSIDE, BOTTOM_LINE, FRONT_Z_DOOR],  # 10
    [-X_OUTSIDE + 0.1, TOP_CAR, 0.1],  # 11
    [-X_OUTSIDE, TOP_LINE, 0.05],  # 12
    [-X_OUTSIDE, 0.0, -0.1],  # 13
    [-X_OUTSIDE, 0.0, 0.0],  # 14
    [-X_OUTSIDE, BOTTOM_LINE, 0.0],  # 15
    [-X_OUTSIDE, TOP_LINE, BACK_Z_WHEEL],  # 16
    [-X_OUTSIDE, 0.0, BACK_Z_WHEEL * 0.8],  # 17
    [-X_OUTSIDE, 0.0, BACK_Z_WHEEL * 0.9],  # 18
    [-X_OUTSIDE, BOTTOM_LINE, BACK_Z_WHEEL * 0.6],  # 19
    [-X_OUTSIDE, BOTTOM_LINE + 0.1, BACK_Z_WHEEL],  # 20
    [-X_OUTSIDE, BOTTOM_LINE, BACK_Z_SIDE - 0.2],  # 21
    [-X_OUTSIDE, 0.0, BACK_Z_SIDE],  # 22
    [-X_OUTSIDE, -0.2, BACK_Z_SIDE],  # 23
    [-X_OUTSIDE + 0.1, TOP_CAR - 0.1, BACK_Z_WHEEL],  # 24
    [-LIGHT_X_INSIDE, 0.0, BACK_Z],  # 25
    [-LIGHT_X_INSIDE, -0.2, BACK_Z],  # 26
    [-LIGHT_X_INSIDE + 0.1, -0.3, BACK_Z],  # 27
    [-X_OUTSIDE + 0.1, BOTTOM_LINE, BACK_Z]] + \
    [[np.nan, np.nan, np.nan]] * 30 + \
    [[-P_X, P_Y_TOP, FRONT_Z]] + \
    [[np.nan, np.nan, np.nan]] + \
    [[-P_X, P_Y_BOTTOM, FRONT_Z],  # 61
     [-P_X, P_Y_TOP, BACK_Z]] + \
    [[np.nan, np.nan, np.nan]] * 2 + \
    [[-P_X, P_Y_BOTTOM, BACK_Z]])  # 65

CAR_POSE_66 = CAR_POSE_HALF
for key in HFLIP_ids:
    CAR_POSE_66[HFLIP_ids[key], :] = CAR_POSE_HALF[key, :]
    CAR_POSE_66[HFLIP_ids[key], 0] = -CAR_POSE_HALF[key, 0]
assert not np.any(CAR_POSE_66 == np.nan)

training_weights_local_centrality = [
    0.890968488270775,
    0.716506138617812,
    1.05674590410869,
    0.764774195768455,
    0.637682585483328,
    0.686680807728366,
    0.955422595797394,
    0.936714585642375,
    1.34823795445326,
    1.38308992581967,
    1.32689945125819,
    1.38838655605483,
    1.18980184904613,
    1.02584355494795,
    0.90969156732068,
    1.24732068576104,
    1.11338768064342,
    0.933815217550391,
    0.852297518872114,
    1.04167641424727,
    1.01668968075247,
    1.34625964088011,
    0.911796331039028,
    0.866206536337413,
    1.55957820407853,
    0.730844382675724,
    0.651138644197359,
    0.758018559633786,
    1.31842501396691,
    1.32186116654782,
    0.744347016851606,
    0.636390683664723,
    0.715244950821949,
    1.63122349407032,
    0.849835699185461,
    0.910488007220499,
    1.44244151650561,
    1.14150437331681,
    1.19808610191343,
    0.960186788642886,
    1.05023623286937,
    1.19761709710598,
    1.3872216313401,
    1.01256700741214,
    1.1167909667759,
    1.27893496336199,
    1.54475684725655,
    1.40343733870633,
    1.45552060866114,
    1.47264222155031,
    0.970060423999993,
    0.944450314768933,
    0.623987071240172,
    0.5745237907704,
    0.66890646050993,
    0.978411632994504,
    0.587396395188292,
    0.76307999741129,
    0.609793563449648,
    0.67983566494545,
    0.685883538168462,
    0.753587600664775,
    0.770335133588157,
    0.764713638033368,
    0.792364155965385,
    0.796435233566833
]


def get_constants(num_kps):
    if num_kps == 24:
        CAR_POSE_24[:, 2] = 2.0
        return [CAR_KEYPOINTS_24, CAR_SKELETON_24, HFLIP_24, CAR_SIGMAS_24,
                CAR_POSE_24, CAR_CATEGORIES_24, CAR_SCORE_WEIGHTS_24]
    if num_kps == 66:
        CAR_POSE_66[:, 2] = 2.0
        return [CAR_KEYPOINTS_66, CAR_SKELETON_66, HFLIP_66, CAR_SIGMAS_66,
                CAR_POSE_66, CAR_CATEGORIES_66, CAR_SCORE_WEIGHTS_66]
    # using no if-elif-else construction due to pylint no-else-return error
    raise Exception("Only poses with 24 or 66 keypoints are available.")


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


def draw_skeletons(pose, sigmas, skel, kps, scr_weights):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    show.KeypointPainter.show_joint_scales = True
    keypoint_painter = show.KeypointPainter()
    ann = Annotation(keypoints=kps, skeleton=skel, score_weights=scr_weights)
    ann.set(pose, np.array(sigmas) * scale)
    os.makedirs('docs', exist_ok=True)
    draw_ann(ann, filename='docs/skeleton_car.png', keypoint_painter=keypoint_painter)


def plot3d_red(ax_2D, p3d, skeleton):
    skeleton = [(bone[0] - 1, bone[1] - 1) for bone in skeleton]

    rot_p90_x = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    p3d = p3d @ rot_p90_x

    fig = ax_2D.get_figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_axis_off()
    ax_2D.set_axis_off()

    ax.view_init(azim=-90, elev=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_range = np.array([p3d[:, 0].max() - p3d[:, 0].min(),
                          p3d[:, 1].max() - p3d[:, 1].min(),
                          p3d[:, 2].max() - p3d[:, 2].min()]).max() / 2.0
    mid_x = (p3d[:, 0].max() + p3d[:, 0].min()) * 0.5
    mid_y = (p3d[:, 1].max() + p3d[:, 1].min()) * 0.5
    mid_z = (p3d[:, 2].max() + p3d[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)  # pylint: disable=no-member

    for ci, bone in enumerate(skeleton):
        c = mplcm.get_cmap('tab20')((ci % 20 + 0.05) / 20)  # Same coloring as Pifpaf preds
        ax.plot(p3d[bone, 0], p3d[bone, 1], p3d[bone, 2], color=c)

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig

    return FuncAnimation(fig, animate, frames=360, interval=100)


def print_associations():
    print("\nAssociations of the car skeleton with 24 keypoints")
    for j1, j2 in CAR_SKELETON_24:
        print(CAR_KEYPOINTS_24[j1 - 1], '-', CAR_KEYPOINTS_24[j2 - 1])
    print("\nAssociations of the car skeleton with 66 keypoints")
    for j1, j2 in CAR_SKELETON_66:
        print(CAR_KEYPOINTS_66[j1 - 1], '-', CAR_KEYPOINTS_66[j2 - 1])


def main():
    print_associations()
# =============================================================================
#     draw_skeletons(CAR_POSE_24, sigmas = CAR_SIGMAS_24, skel = CAR_SKELETON_24,
#                    kps = CAR_KEYPOINTS_24, scr_weights = CAR_SCORE_WEIGHTS_24)
#     draw_skeletons(CAR_POSE_66, sigmas = CAR_SIGMAS_66, skel = CAR_SKELETON_66,
#                    kps = CAR_KEYPOINTS_66, scr_weights = CAR_SCORE_WEIGHTS_66)
# =============================================================================
    with openpifpaf.show.Canvas.blank(nomargin=True) as ax_2D:
        anim_66 = plot3d_red(ax_2D, CAR_POSE_66, CAR_SKELETON_66)
        anim_66.save('openpifpaf/plugins/apollocar3d/docs/CAR_66_Pose.gif', fps=30)
    with openpifpaf.show.Canvas.blank(nomargin=True) as ax_2D:
        anim_24 = plot3d_red(ax_2D, CAR_POSE_24, CAR_SKELETON_24)
        anim_24.save('openpifpaf/plugins/apollocar3d/docs/CAR_24_Pose.gif', fps=30)


if __name__ == '__main__':
    main()
