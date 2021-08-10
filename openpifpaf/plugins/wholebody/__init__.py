import openpifpaf

from .wholebody import Wholebody


def register():
    openpifpaf.DATAMODULES['wholebody'] = Wholebody
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k16-wholebody'] = 'http://github.com/DuncanZauss/' \
        'openpifpaf_assets/releases/download/v0.1.0/sk16_wholebody.pkl'
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k30-wholebody'] = 'http://github.com/DuncanZauss/' \
        'openpifpaf_assets/releases/download/v0.1.0/sk30_wholebody.pkl'
