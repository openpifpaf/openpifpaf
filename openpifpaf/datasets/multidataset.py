import torch

class MultiDataset(object):
    def __init__(self, heads, dataloaders): # datloaders: list of tuples of train dl and val dl
        # self.train_dl = []
        # self.val_dl = []
        # # print('len dataloaders',dataloaders)
        # for train_dl, val_dl in dataloaders:
        #     self.train_dl.append(train_dl)
        #     self.val_dl.append(train_dl)
        self.heads = heads
        # print('len dataloaders',len(self.train_dl))
        # print('len dataloaders',len(self.val_dl))

        self.train_dl = dataloaders
        cifs = ['cif', 'cifcent', 'cifcentball']
        # cif_mode = ['cif' in self.config, 'cifcent' in self.config, 'cifcentball' in self.config]
        self.cif_mode = cifs[True]

    # def trainIter(self):
    #     self.train_iters = [iter(tr_dl) for tr_dl in self.train_dl]
    
    def __iter__(self):
        self.train_iters = [iter(tr_dl) for tr_dl in self.train_dl]
        return self

    def __next__(self):
        try:
            list_scenes = [next(tr_iter) for tr_iter in self.train_iters]
        except StopIteration:
            raise StopIteration

        # target_pan = None
        
        data = torch.cat([img for img,_,_ in list_scenes], dim=0)
        meta = [met for _,_,met in list_scenes]
        targets = []

        for h_ix, head in enumerate(self.heads):
            if head in ['cif', 'cifcent', 'cifcentball']:
                cif_conf = torch.cat([trg[h_ix][0] for _,trg,_ in list_scenes], dim=0)
                cif_vec = torch.cat([trg[h_ix][1] for _,trg,_ in list_scenes], dim=0)
                cif_scale = torch.cat([trg[h_ix][2] for _,trg,_ in list_scenes], dim=0)
                targets.append((cif_conf, cif_vec, cif_scale))
            elif head == 'pan':
                target_pan = dict()
                target_pan['semantic'] = torch.cat([trg[h_ix]['semantic'] for _,trg,_ in list_scenes], dim=0)
                target_pan['offset']  = torch.cat([trg[h_ix]['offset'] for _,trg,_ in list_scenes], dim=0)
                target_pan['semantic_weights']  = torch.cat([trg[h_ix]['semantic_weights'] for _,trg,_ in list_scenes], dim=0)
                target_pan['offset_weights'] = torch.cat([trg[h_ix]['offset_weights'] for _,trg,_ in list_scenes], dim=0)
                targets.append(target_pan)
            elif head == 'ball':
                cif_conf = torch.cat([trg[h_ix][0] for _,trg,_ in list_scenes], dim=0)
                cif_vec = torch.cat([trg[h_ix][1] for _,trg,_ in list_scenes], dim=0)
                cif_scale = torch.cat([trg[h_ix][2] for _,trg,_ in list_scenes], dim=0)
                targets.append((cif_conf, cif_vec, cif_scale))
        
        # cif_mode = [cif_mode]
        # if self.cif_mode:
        #     # cif_mode = cifs[True]
        #     # print('cif mode')
        #     cif_conf = torch.cat([trg[self.cif_mode][0] for _,trg,_ in list_scenes], dim=0)
        #     cif_vec = torch.cat([trg[self.cif_mode][1] for _,trg,_ in list_scenes], dim=0)
        #     cif_scale = torch.cat([trg[self.cif_mode][2] for _,trg,_ in list_scenes], dim=0)
        #     targets[self.cif_mode] = (cif_conf, cif_vec, cif_scale)
        # if 'pan' in self.config:
        #     target_pan = dict()
        #     target_pan['semantic'] = torch.cat([trg['pan']['semantic'] for _,trg,_ in list_scenes], dim=0)
        #     target_pan['offset']  = torch.cat([trg['pan']['offset'] for _,trg,_ in list_scenes], dim=0)
        #     target_pan['semantic_weights']  = torch.cat([trg['pan']['semantic_weights'] for _,trg,_ in list_scenes], dim=0)
        #     target_pan['offset_weights'] = torch.cat([trg['pan']['offset_weights'] for _,trg,_ in list_scenes], dim=0)
        #     targets['pan'] = target_pan

        # if 'ball' in self.config:
        #     cif_ball_conf = torch.cat([trg['ball'][0] for _,trg,_ in list_scenes], dim=0)
        #     cif_ball_vec = torch.cat([trg['ball'][1] for _,trg,_ in list_scenes], dim=0)
        #     cif_ball_scale = torch.cat([trg['ball'][2] for _,trg,_ in list_scenes], dim=0)
        #     targets[self.cif_mode] = (cif_ball_conf, cif_ball_vec, cif_ball_scale)
        # if target_pan == None:
        #     return data, [(cif_conf, cif_vec, cif_scale)], 0
        # if cif_ball_conf is not None:
        #     return data, [(cif_conf, cif_vec, cif_scale), target_pan, (cif_ball_conf, cif_ball_vec, cif_ball_scale)], 0
        # else:
        #     return data, [(cif_conf, cif_vec, cif_scale), target_pan], 0
        return data, targets, meta


    # def getTrainNext(self, i):
    #     # whichone = i % len(self.train_dl)
    #     # return next(self.train_iters[whichone])

    #     list_scenes = [next(tr_iter) for tr_iter in self.train_iters]
    #     target_pan = None
    #     data = torch.cat([img for img,_,_ in list_scenes], dim=0)
    #     if 'cif' in self.config or 'cifcent' in self.config or 'cifcentball' in self.config:
    #         cif_conf = torch.cat([trg[0][0] for _,trg,_ in list_scenes], dim=0)
    #         cif_vec = torch.cat([trg[0][1] for _,trg,_ in list_scenes], dim=0)
    #         cif_scale = torch.cat([trg[0][2] for _,trg,_ in list_scenes], dim=0)
    #     if 'pan' in self.config:
    #         target_pan = dict()
    #         target_pan['semantic'] = torch.cat([trg[1]['semantic'] for _,trg,_ in list_scenes], dim=0)
    #         target_pan['offset']  = torch.cat([trg[1]['offset'] for _,trg,_ in list_scenes], dim=0)
    #         target_pan['semantic_weights']  = torch.cat([trg[1]['semantic_weights'] for _,trg,_ in list_scenes], dim=0)
    #         target_pan['offset_weights'] = torch.cat([trg[1]['offset_weights'] for _,trg,_ in list_scenes], dim=0)
    #     if target_pan == None:
    #         return data, [(cif_conf, cif_vec, cif_scale)]
    #     else:
    #         return data, [(cif_conf, cif_vec, cif_scale), target_pan]
    #     raise
        

        
    # def valIter(self):
    #     self.val_iters = [iter(vl_dl) for vl_dl in self.val_dl]
    
    # def getValNext(self, i):
    #     whichone = i % len(self.val_dl)
    #     return next(self.val_iters[whichone])

    # def getValLen(self):
    #     return min([len(d) for d in self.val_dl])


    def __len__(self):
        return min([len(d) for d in self.train_dl])