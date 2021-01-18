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
            this_color = color
            if this_color is None:
                this_color = i
            if colors is not None:
                this_color = colors[i]
            elif hasattr(ann, 'id_'):
                this_color = ann.id_

            text = None
            text_is_score = False
            if texts is not None:
                text = texts[i]
            elif hasattr(ann, 'id_'):
                text = '{}'.format(ann.id_)
            elif getattr(ann, 'score', None):
                text = '{:.0%}'.format(ann.score)
                text_is_score = True

            subtext = None
            if subtexts is not None:
                subtext = subtexts[i]
            elif not text_is_score and getattr(ann, 'score', None):
                subtext = '{:.0%}'.format(ann.score)

            painter = self.painters[ann.__class__.__name__]
            painter.annotation(ax, ann, color=this_color, text=text, subtext=subtext, **kwargs)
