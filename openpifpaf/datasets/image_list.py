import PIL
import torch


from .. import transforms


class ImageList(torch.utils.data.Dataset):
    def __init__(self, image_paths, preprocess=None, with_raw_image=False):
        super().__init__()
        self.image_paths = image_paths
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.with_raw_image = with_raw_image

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with open(image_path, 'rb') as f:
            image = PIL.Image.open(f).convert('RGB')

        anns = []
        meta = {
            'dataset_index': index,
            'file_name': image_path,
        }
        processed_image, anns, meta = self.preprocess(image, anns, meta)
        if self.with_raw_image:
            return image, processed_image, anns, meta
        return processed_image, anns, meta

    def __len__(self):
        return len(self.image_paths)


class PilImageList(torch.utils.data.Dataset):
    def __init__(self, images, preprocess=None, with_raw_image=False):
        super().__init__()
        self.images = images
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.with_raw_image = with_raw_image

    def __getitem__(self, index):
        image = self.images[index].copy().convert('RGB')

        anns = []
        meta = {
            'dataset_index': index,
        }
        processed_image, anns, meta = self.preprocess(image, anns, meta)
        if self.with_raw_image:
            return image, processed_image, anns, meta
        return processed_image, anns, meta

    def __len__(self):
        return len(self.images)


class NumpyImageList(torch.utils.data.Dataset):
    def __init__(self, images, preprocess=None, with_raw_image=False):
        super().__init__()
        self.images = images
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.with_raw_image = with_raw_image

    def __getitem__(self, index):
        image = PIL.Image.fromarray(self.images[index]).copy()

        anns = []
        meta = {
            'dataset_index': index,
        }
        processed_image, anns, meta = self.preprocess(image, anns, meta)
        if self.with_raw_image:
            return image, processed_image, anns, meta
        return processed_image, anns, meta

    def __len__(self):
        return len(self.images)
