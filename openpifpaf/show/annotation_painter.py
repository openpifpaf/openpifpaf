from collections import defaultdict

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
        self.painters = {}
        for annotation_type, painter in PAINTERS.items():
            if painters and annotation_type in painters:
                self.painters[annotation_type] = painters[annotation_type]
            else:
                self.painters[annotation_type] = painter(xy_scale=xy_scale)

    def annotations(self, ax, annotations, *,
                    color=None, colors=None, texts=None, subtexts=None):
        by_classname = defaultdict(list)
        for ann_i, ann in enumerate(annotations):
            by_classname[ann.__class__.__name__].append((ann_i, ann))

        for classname, i_anns in by_classname.items():
            anns = [ann for _, ann in i_anns]
            this_colors = [colors[i] for i, _ in i_anns] if colors else None
            this_texts = [texts[i] for i, _ in i_anns] if texts else None
            this_subtexts = [subtexts[i] for i, _ in i_anns] if subtexts else None
            self.painters[classname].annotations(
                ax, anns,
                color=color, colors=this_colors, texts=this_texts, subtexts=this_subtexts)
