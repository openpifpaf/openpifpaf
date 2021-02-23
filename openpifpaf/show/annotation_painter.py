from .painters import KeypointPainter, CrowdPainter, DetectionPainter

PAINTERS = {
    'Annotation': KeypointPainter,
    'AnnotationCrowd': CrowdPainter,
    'AnnotationDet': DetectionPainter,
}


class AnnotationPainter:
    def __init__(self, *,
                 xy_scale=1.0,
                 painters=None):
        self.painters = {annotation_type: painter(xy_scale=xy_scale)
                         for annotation_type, painter in PAINTERS.items()}

        if painters:
            for annotation_type, painter in painters.items():
                self.painters[annotation_type] = painter

    def annotations(self, ax, annotations, *,
                    color=None, colors=None, texts=None, subtexts=None, **kwargs):
        for i, ann in enumerate(annotations):
            if colors is not None:
                this_color = colors[i]
            elif color is not None:
                this_color = color
            elif getattr(ann, 'id_', None):
                this_color = ann.id_
            else:
                this_color = i

            text = None
            if texts is not None:
                text = texts[i]

            subtext = None
            if subtexts is not None:
                subtext = subtexts[i]

            painter = self.painters[ann.__class__.__name__]
            painter.annotation(ax, ann, color=this_color, text=text, subtext=subtext, **kwargs)
