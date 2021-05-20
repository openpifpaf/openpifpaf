import torch


def collate_images_anns_meta(batch):
    anns = [b[-2] for b in batch]
    metas = [b[-1] for b in batch]

    if len(batch[0]) == 4:
        # raw images are also in this batch
        images = [b[0] for b in batch]
        processed_images = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
        return images, processed_images, anns, metas

    processed_images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    return processed_images, anns, metas


def collate_images_targets_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    targets = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return images, targets, metas


def collate_tracking_images_targets_meta(batch):
    images = torch.utils.data.dataloader.default_collate([
        im for group in batch for im in group[0]])

    targets = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    metas = [b[2] for b in batch]

    return images, targets, metas
