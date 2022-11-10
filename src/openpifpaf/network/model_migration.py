from . import tracking_heads
from .nets import model_defaults
from .tracking_base import TrackingBase
from ..signal import Signal

MODEL_MIGRATION = set()


# pylint: disable=protected-access,too-many-branches
def model_migration(net_cpu):
    model_defaults(net_cpu)

    if not hasattr(net_cpu, 'process_heads'):
        net_cpu.process_heads = None

    for m in net_cpu.modules():
        if not hasattr(m, '_non_persistent_buffers_set'):
            m._non_persistent_buffers_set = set()

        if m.__class__.__name__ == 'InvertedResidualK' and not hasattr(m, 'branch1'):
            m.branch1 = None
        if m.__class__.__name__ == 'GELU' and not hasattr(m, 'approximate'):
            m.approximate = 'none'

    if not hasattr(net_cpu, 'head_nets') and hasattr(net_cpu, '_head_nets'):
        net_cpu.head_nets = net_cpu._head_nets

    for hn_i, hn in enumerate(net_cpu.head_nets):
        if not hn.meta.base_stride:
            hn.meta.base_stride = net_cpu.base_net.stride
        if hn.meta.head_index is None:
            hn.meta.head_index = hn_i
        if hn.meta.name == 'cif' and 'score_weights' not in vars(hn.meta):
            hn.meta.score_weights = [3.0] * 3 + [1.0] * (hn.meta.n_fields - 3)

        if not hasattr(hn, 'n_fields'):
            hn.n_fields = hn.meta.n_fields
        if not hasattr(hn, 'n_confidences'):
            hn.n_confidences = hn.meta.n_confidences
        if not hasattr(hn, 'n_vectors'):
            hn.n_vectors = hn.meta.n_vectors
        if not hasattr(hn, 'n_scales'):
            hn.n_scales = hn.meta.n_scales
        if not hasattr(hn, 'vector_offsets'):
            hn.vector_offsets = hn.meta.vector_offsets
        if not hasattr(hn, 'upsample_stride'):
            hn.upsample_stride = hn.meta.upsample_stride

    for mm in MODEL_MIGRATION:
        mm(net_cpu)


def fix_feature_cache(model):
    for m in model.modules():
        if not isinstance(m, TrackingBase):
            continue
        m.reset()


def subscribe_cache_reset(model):
    for m in model.modules():
        if not isinstance(m, TrackingBase):
            continue
        Signal.subscribe('eval_reset', m.reset)


def tcaf_shared_preprocessing(model):
    for m in model.modules():
        if not isinstance(m, tracking_heads.Tcaf):
            continue

        # pylint: disable=protected-access
        tracking_heads.Tcaf._global_feature_reduction = m.feature_reduction
        tracking_heads.Tcaf._global_feature_compute = m.feature_compute
        return


MODEL_MIGRATION.add(fix_feature_cache)
MODEL_MIGRATION.add(subscribe_cache_reset)
MODEL_MIGRATION.add(tcaf_shared_preprocessing)
