from abc import abstractmethod
import logging
import multiprocessing
import sys
import time

import torch

from ... import visualizer
from ... import network

LOG = logging.getLogger(__name__)


class DummyPool():
    @staticmethod
    def starmap(f, iterable):
        return [f(*i) for i in iterable]


class Generator:
    def __init__(self, worker_pool=None):
        if worker_pool is None or worker_pool == 0:
            worker_pool = DummyPool()
        if isinstance(worker_pool, int):
            LOG.info('creating decoder worker pool with %d workers', worker_pool)
            assert not sys.platform.startswith('win'), (
                'not supported, use --decoder-workers=0 '
                'on windows'
            )
            worker_pool = multiprocessing.Pool(worker_pool)

        self.worker_pool = worker_pool

        self.last_decoder_time = 0.0
        self.last_nn_time = 0.0

    def __getstate__(self):
        return {
            k: v for k, v in self.__dict__.items()
            if k not in ('worker_pool',)
        }

    @staticmethod
    def fields_batch(model, image_batch, *, device=None, oracle_masks=None, target_batch=None):
        """From image batch to field batch."""
        start = time.time()
        # print('Oracle masks', oracle_masks)
        def apply(f, items):
            """Apply f in a nested fashion to all items that are not list or tuple."""
            if items is None:
                return None
            if isinstance(items, (list, tuple)):
                return [apply(f, i) for i in items]
            if isinstance(items, dict):
                return {k: apply(f, v) for k, v in items.items()}
            return f(items)

        with torch.no_grad():
            if device is not None:
                image_batch = image_batch.to(device, non_blocking=True)

            with torch.autograd.profiler.record_function('model'):
                heads = model(image_batch)

            # to numpy
            with torch.autograd.profiler.record_function('tonumpy'):
                heads = apply(lambda x: x.cpu().numpy(), heads)

        if oracle_masks is not None:
            import numpy as np
            def classes_from_target(semantic_target):
                ''' converting [B,H,W] to [B,C,H,W] '''
                
                B, H, W = semantic_target.shape
                C = semantic_target.max() + 1
                target = np.zeros((B, C, H, W))
                for cc in range(C):
                    target[:,cc,:,:] = np.where(semantic_target == cc, 1, 0)
                return target

            def denan(t):
                return torch.where(torch.isnan(t), torch.zeros_like(t), t)

            # if 'semantic' in oracle_masks or 'offset' in oracle_masks:
            pan_target = {}
            pan_target['semantic'] = classes_from_target(target_batch[1]['semantic'].numpy())
            pan_target['offset'] = target_batch[1]['offset'].numpy()
            if len(target_batch) == 3:
                gt = [torch.cat([denan(target_batch[0][0])[:,:,None],
                            denan(target_batch[0][1])[:,:,:2]+network.index_field_torch(target_batch[0][1].shape[-2:]),
                            torch.ones_like(target_batch[0][2][:,:,None])*0,
                            denan(target_batch[0][2])[:,:,None]
                            ], dim=2),
                    pan_target,
                    torch.cat([denan(target_batch[2][0])[:,:,None],
                            denan(target_batch[2][1])[:,:,:2]+network.index_field_torch(target_batch[2][1].shape[-2:]),
                            torch.ones_like(target_batch[2][2][:,:,None])*0,
                            denan(target_batch[2][2])[:,:,None]
                            ], dim=2),
                            ]
            elif len(target_batch) == 4:
                gt = [torch.cat([denan(target_batch[0][0])[:,:,None],
                                denan(target_batch[0][1])[:,:,:2]+network.index_field_torch(target_batch[0][1].shape[-2:]),
                                torch.ones_like(target_batch[0][2][:,:,None])*0,
                                denan(target_batch[0][2])[:,:,None]
                                ], dim=2),
                        pan_target,
                        torch.cat([denan(target_batch[2][0])[:,:,None],
                                denan(target_batch[2][1])[:,:,:2]+network.index_field_torch(target_batch[2][1].shape[-2:]),
                                torch.ones_like(target_batch[2][2][:,:,None])*0,
                                denan(target_batch[2][2])[:,:,None]
                                ], dim=2),
                        torch.cat([denan(target_batch[3][0])[:,:,None],
                                denan(target_batch[3][1])[:,:,:2]+network.index_field_torch(target_batch[3][1].shape[-2:]),
                                torch.ones_like(target_batch[3][2][:,:,None])*0,
                                denan(target_batch[3][2])[:,:,None]
                                ], dim=2),
                                ]

            if 'semantic' in oracle_masks:
                print('ORACLE: semantic replaced')
                heads[1]['semantic'] = gt[1]['semantic']
            if 'offset' in oracle_masks:
                print('ORACLE: offset replaced')
                heads[1]['offset'] = gt[1]['offset']
            # print('head type',heads[0].shape)
            # print('gt type',gt[0].shape)
            # print('type heads', type(heads))
            # print('type gt', type(gt))
            # import matplotlib.pyplot as plt
            # import scipy
            # con = scipy.ndimage.zoom(heads[0][0,-1,0,:,:], (8, 8))
            # plt.imshow(con, cmap='jet')
            # plt.colorbar()
            # image = image_batch[0].cpu().permute(1,2,0).numpy()
            # image = (image - image.min())/(image.max() - image.min())
            # plt.imshow(image, alpha=.2)
            # plt.show()
            # plt.savefig('image/before_.png')
            K = len(heads[0][0])
            if K == 18:
                if 'centroid' in oracle_masks and 'keypoints' in oracle_masks:
                    import copy
                    heads[0][:,:,:,:,:] = copy.deepcopy(gt[0][:,:,:,:,:].numpy())    # [B,K,5,W_L,H_L]
                elif 'centroid' in oracle_masks:
                    import copy
                    heads[0][:,-1,:,:,:] = copy.deepcopy(gt[0][:,-1,:,:,:].numpy())    # [B,K,5,W_L,H_L]
            elif K == 17:
                if 'keypoints' in oracle_masks:
                    import copy
                    print('ORACLE: keypoints replaced')
                    heads[0] = copy.deepcopy(gt[0].numpy())    # [B,K,5,W_L,H_L]
                if 'centroid' in oracle_masks:
                    import copy
                    print('ORACLE: centroid replaced')
                    heads[3] = copy.deepcopy(gt[3].numpy())    # [B,K,5,W_L,H_L]

            # con = scipy.ndimage.zoom(heads[0][0,-1,0,:,:], (8, 8))
            # plt.imshow(con, cmap='jet')
            # # plt.colorbar()
            # plt.imshow(image, alpha=.2)
            # plt.show()
            # plt.savefig('image/after_.png')
            # raise

        # index by frame (item in batch)
        head_iter = apply(iter, heads)
        heads = []
        while True:
            try:
                heads.append(apply(next, head_iter))
            except StopIteration:
                break

        LOG.debug('nn processing time: %.3fs', time.time() - start)
        return heads

    @abstractmethod
    def __call__(self, fields, *, initial_annotations=None):
        """For single image, from fields to annotations."""
        raise NotImplementedError()

    def batch(self, model, image_batch, *, device=None, oracle_masks=None, target_batch=None):
        """From image batch straight to annotations batch."""
        start_nn = time.perf_counter()
        fields_batch = self.fields_batch(model, image_batch, device=device, oracle_masks=oracle_masks, target_batch=target_batch)
        # print('fields batch',len(fields_batch[0]))
        # print('fields batch',fields_batch)
        self.last_nn_time = time.perf_counter() - start_nn

        if not isinstance(self.worker_pool, DummyPool):
            # remove debug_images to save time during pickle
            image_batch = [None for _ in fields_batch]

        LOG.debug('parallel execution with worker %s', self.worker_pool)
        start_decoder = time.perf_counter()
        result = self.worker_pool.starmap(
            self._mappable_annotations, zip(fields_batch, image_batch))
        self.last_decoder_time = time.perf_counter() - start_decoder

        LOG.debug('time: nn = %.3fs, dec = %.3fs', self.last_nn_time, self.last_decoder_time)
        return result

    def _mappable_annotations(self, fields, debug_image):
        if debug_image is not None:
            visualizer.BaseVisualizer.processed_image(debug_image)

        return self(fields)
