import openpifpaf
from openpifpaf.datasets.constants import (
    COCO_KEYPOINTS,
    COCO_PERSON_SKELETON,
    COCO_UPRIGHT_POSE,
)


def main():
    ann = openpifpaf.Annotation(keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON)
    ann.set(COCO_UPRIGHT_POSE)

    # favicon
    keypoint_painter = openpifpaf.show.KeypointPainter(
        line_width=48, marker_size=0)
    openpifpaf.datasets.constants.draw_ann(
        ann,
        keypoint_painter=keypoint_painter,
        aspect='equal',
        margin=0.8,
        frameon=False,
        filename='favicon.png',
    )

    # logo
    keypoint_painter = openpifpaf.show.KeypointPainter(
        line_width=12)
    openpifpaf.datasets.constants.draw_ann(
        ann,
        keypoint_painter=keypoint_painter,
        frameon=False,
        filename='logo.png',
    )


if __name__ == '__main__':
    main()
