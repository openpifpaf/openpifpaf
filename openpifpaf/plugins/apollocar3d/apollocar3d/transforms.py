
from collections import defaultdict


def skeleton_mapping(kps_mapping):
    """Map the subset of keypoints from 0 to n-1"""
    map_sk = defaultdict(lambda: 100)  # map to 100 the keypoints not used
    for i, j in zip(kps_mapping, range(len(kps_mapping))):
        map_sk[i] = j
    return map_sk


def transform_skeleton(skeleton_orig, kps_mapping):
    """
    Transform the original apollo skeleton of 66 joints into a skeleton from 1 to n
    """
    map_sk = skeleton_mapping(kps_mapping)
    # skeleton = [[dic_sk[i], dic_sk[j]] for i, j in SKELETON]  # TODO
    skeleton = []
    for i, j in skeleton_orig:
        skeleton.append([map_sk[i] + 1, map_sk[j] + 1])   # skeleton starts from 1
    return skeleton
