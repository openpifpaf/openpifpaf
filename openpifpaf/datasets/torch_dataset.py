import logging

import torch.utils.data

from .. import transforms


LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))


class TorchDataset(torch.utils.data.Dataset):
    """Wraps a torch-based parent dataset and applies OpenPifPaf transforms."""

    def __init__(self, parent, *, preprocess=None):
        super().__init__()
        self.parent = parent
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM

    def __getitem__(self, index):
        parent_data = self.parent[index]

        meta = {
            'dataset_index': index,
        }

        # preprocess image and annotations
        image, anns, meta = self.preprocess(parent_data, meta)
        LOG.debug(meta)

        return image, anns, meta

    def __len__(self):
        return len(self.parent)
