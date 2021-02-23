import openpifpaf
from openpifpaf.plugins.coco.constants import (
    COCO_KEYPOINTS,
    COCO_PERSON_SKELETON,
    COCO_UPRIGHT_POSE,
)


def main():
    ann = openpifpaf.Annotation(keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON)
    ann.set(COCO_UPRIGHT_POSE, fixed_score='')

    # favicon
    keypoint_painter = openpifpaf.show.KeypointPainter(
        line_width=48, marker_size=0)
    with openpifpaf.show.Canvas.annotation(ann, filename='favicon.png',
                                           margin=0.8, frameon=False, fig_w=5) as ax:
        ax.set_aspect('equal')
        keypoint_painter.annotation(ax, ann)

    # logo
    keypoint_painter = openpifpaf.show.KeypointPainter(
        line_width=12)
    with openpifpaf.show.Canvas.annotation(ann, filename='logo.png') as ax:
        keypoint_painter.annotation(ax, ann)


if __name__ == '__main__':
    main()
