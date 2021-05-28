import torch


def collate_images_anns_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    anns = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    return images, anns, metas


def collate_images_targets_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    targets = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return images, targets, metas

### AMA
def collate_images_targets_inst_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    targets = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    # masks = [b[2] for b in batch]
    metas = [b[2] for b in batch]
    return images,targets, metas

def collate_images_targets_inst_meta_eval(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    # targets = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    # masks = [b[2] for b in batch]
    anns = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    targets = torch.utils.data.dataloader.default_collate([b[3] for b in batch])
    # targets = [b[3] for b in batch]
    return images,anns, metas, targets

def collate_images_targets_inst_meta_views(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    targets = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    # masks = [b[2] for b in batch]
    metas = [b[2] for b in batch]
    views = [b[3] for b in batch]
    keys = [b[4] for b in batch]
    return images,targets, metas, views, keys