import openpifpaf

from .wholebody import Wholebody


def register():
    openpifpaf.DATAMODULES['wholebody'] = Wholebody
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k16-wholebody'] = 'http://github.com/DuncanZauss/' \
        'openpifpaf_assets/releases/download/v0.1.0/wb_shufflenet16_mixed_foot.pkl.epoch550'
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k30-wholebody'] = 'http://github.com/DuncanZauss/' \
        'openpifpaf_assets/releases/download/v0.1.0/wb_shufflenet30_mixed_foot.pkl.epoch350'
