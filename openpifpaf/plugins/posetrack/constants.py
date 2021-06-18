import numpy as np


KEYPOINTS = [
    'nose',
    'head_bottom',
    'head_top',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
]

SIGMAS = [
    0.026,  # 1 nose
    0.08,  # 2 head_bottom ==> changed versus COCO
    0.06,  # 3 head_top ==> changed versus COCO
    0.035,  # 4 ears ==> never annotated
    0.035,  # 5 ears ==> never annotated
    0.079,  # 6 shoulders
    0.079,  # 7 shoulders
    0.072,  # 8 elbows
    0.072,  # 9 elbows
    0.062,  # 10 wrists
    0.062,  # 11 wrists
    0.107,  # 12 hips
    0.107,  # 13 hips
    0.087,  # 14 knees
    0.087,  # 15 knees
    0.089,  # 16 ankles
    0.089,  # 17 ankles
]

UPRIGHT_POSE = np.array([
    [0.2, 9.3, 2.0],  # 'nose',            # 1
    [-0.05, 9.0, 2.0],  # 'head_bottom',        # 2
    [0.05, 10.0, 2.0],  # 'head_top',       # 3
    [-0.7, 9.5, 2.0],  # 'left_ear',        # 4
    [0.7, 9.5, 2.0],  # 'right_ear',       # 5
    [-1.4, 8.0, 2.0],  # 'left_shoulder',   # 6
    [1.4, 8.0, 2.0],  # 'right_shoulder',  # 7
    [-1.75, 6.0, 2.0],  # 'left_elbow',      # 8
    [1.75, 6.2, 2.0],  # 'right_elbow',     # 9
    [-1.75, 4.0, 2.0],  # 'left_wrist',      # 10
    [1.75, 4.2, 2.0],  # 'right_wrist',     # 11
    [-1.26, 4.0, 2.0],  # 'left_hip',        # 12
    [1.26, 4.0, 2.0],  # 'right_hip',       # 13
    [-1.4, 2.0, 2.0],  # 'left_knee',       # 14
    [1.4, 2.1, 2.0],  # 'right_knee',      # 15
    [-1.4, 0.0, 2.0],  # 'left_ankle',      # 16
    [1.4, 0.1, 2.0],  # 'right_ankle',     # 17
])

SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 6],
    [2, 7],
    [2, 3],
    [1, 2],
    [1, 3],
    [1, 4],  # 4 is never annotated
    [1, 5],  # 5 is never annotated
]

DENSER_CONNECTIONS = [
    [6, 7],  # shoulders
    [8, 9],  # elbows
    [10, 11],  # wrists
    [14, 15],  # knees
    [16, 17],  # ankles
    [6, 10],  # shoulder - wrist
    [7, 11],
    [10, 12],  # wrists - hips
    [11, 13],
    [2, 10],  # headbottom - wrists
    [2, 11],
    [12, 15],  # hip knee cross
    [13, 14],
    [14, 17],  # knee ankle cross
    [15, 16],
    [6, 13],  # shoulders hip cross
    [7, 12],
    [6, 3],  # shoulders head top
    [7, 3],
    [6, 1],  # shoulders head nose
    [7, 1],
    [8, 2],  # elbows head_bottom
    [9, 2],  # elbows head_bottom
]

KEYPOINTS2017 = [
    'right_ankle',
    'right_knee',
    'right_hip',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_wrist',
    'right_elbow',
    'right_shoulder',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'head_bottom',
    'nose',
    'head_top',
]

POSETRACK_INDEX2017TO2018 = [KEYPOINTS.index(kp_name) for kp_name in KEYPOINTS2017]
