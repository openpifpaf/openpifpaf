import torch

class MultiDataset(object):
    def __init__(self, heads, dataloaders): # datloaders: list of tuples of train dl and val dl

        self.heads = heads

        self.train_dl = dataloaders
        cifs = ['cif', 'cifcent', 'cifcentball']
        
    
    def __iter__(self):
        self.train_iters = [iter(tr_dl) for tr_dl in self.train_dl]
        return self

    def __next__(self):
        try:
            list_scenes = [next(tr_iter) for tr_iter in self.train_iters]
        except StopIteration:
            raise StopIteration

        
        
        data = torch.cat([img for img,_,_ in list_scenes], dim=0)
        meta = []
        for _,_,met in list_scenes:
            meta += met
        
        targets = []

        for h_ix, head in enumerate(self.heads):
            if head in ['cif', 'cifcent', 'cifcentball']:
                
                cif_conf = torch.cat([trg[h_ix][0] for _,trg,_ in list_scenes], dim=0)
                cif_vec = torch.cat([trg[h_ix][1] for _,trg,_ in list_scenes], dim=0)
                cif_scale = torch.cat([trg[h_ix][2] for _,trg,_ in list_scenes], dim=0)
                targets.append((cif_conf, cif_vec, cif_scale))
            elif head == 'pan':
                # print('data pan')
                target_pan = dict()
                target_pan['semantic'] = torch.cat([trg[h_ix]['semantic'] for _,trg,_ in list_scenes], dim=0)
                target_pan['offset']  = torch.cat([trg[h_ix]['offset'] for _,trg,_ in list_scenes], dim=0)
                target_pan['semantic_weights']  = torch.cat([trg[h_ix]['semantic_weights'] for _,trg,_ in list_scenes], dim=0)
                target_pan['offset_weights'] = torch.cat([trg[h_ix]['offset_weights'] for _,trg,_ in list_scenes], dim=0)
                targets.append(target_pan)
            elif head == 'cent':
                
                cif_conf = torch.cat([trg[h_ix][0] for _,trg,_ in list_scenes], dim=0)
                cif_vec = torch.cat([trg[h_ix][1] for _,trg,_ in list_scenes], dim=0)
                cif_scale = torch.cat([trg[h_ix][2] for _,trg,_ in list_scenes], dim=0)
                targets.append((cif_conf, cif_vec, cif_scale))
            elif head == 'ball':
                
                cif_conf = torch.cat([trg[h_ix][0] for _,trg,_ in list_scenes], dim=0)
                cif_vec = torch.cat([trg[h_ix][1] for _,trg,_ in list_scenes], dim=0)
                cif_scale = torch.cat([trg[h_ix][2] for _,trg,_ in list_scenes], dim=0)
                targets.append((cif_conf, cif_vec, cif_scale))
        
    
        return data, targets, meta


    def __len__(self):
        return min([len(d) for d in self.train_dl])