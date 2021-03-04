import torch

class MultiDataset(object):
    def __init__(self, args, dataloaders): # datloaders: list of tuples of train dl and val dl
        # self.train_dl = []
        # self.val_dl = []
        # # print('len dataloaders',dataloaders)
        # for train_dl, val_dl in dataloaders:
        #     self.train_dl.append(train_dl)
        #     self.val_dl.append(train_dl)
        self.config = args.headnets
        # print('len dataloaders',len(self.train_dl))
        # print('len dataloaders',len(self.val_dl))

        self.train_dl = dataloaders

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

        target_pan = None
        data = torch.cat([img for img,_,_ in list_scenes], dim=0)
        if 'cif' in self.config or 'cifcent' in self.config or 'cifcentball' in self.config:
            cif_conf = torch.cat([trg[0][0] for _,trg,_ in list_scenes], dim=0)
            cif_vec = torch.cat([trg[0][1] for _,trg,_ in list_scenes], dim=0)
            cif_scale = torch.cat([trg[0][2] for _,trg,_ in list_scenes], dim=0)
        if 'pan' in self.config:
            target_pan = dict()
            target_pan['semantic'] = torch.cat([trg[1]['semantic'] for _,trg,_ in list_scenes], dim=0)
            target_pan['offset']  = torch.cat([trg[1]['offset'] for _,trg,_ in list_scenes], dim=0)
            target_pan['semantic_weights']  = torch.cat([trg[1]['semantic_weights'] for _,trg,_ in list_scenes], dim=0)
            target_pan['offset_weights'] = torch.cat([trg[1]['offset_weights'] for _,trg,_ in list_scenes], dim=0)
        if target_pan == None:
            return data, [(cif_conf, cif_vec, cif_scale)], 0
        else:
            return data, [(cif_conf, cif_vec, cif_scale), target_pan], 0
        raise


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