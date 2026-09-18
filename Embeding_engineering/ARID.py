
import torch.nn.functional as F
import torch

import numpy as np

train = 'data/ARID/MAJOR-ARID-FEATS-VIDEOCHAT2/ARIDannotesfeats_train.txt'
val = 'data/ARID/MAJOR-ARID-FEATS-VIDEOCHAT2/ARIDannotesfeats_val.txt'
test = 'data/ARID/MAJOR-ARID-FEATS-VIDEOCHAT2/ARIDannotesfeats_test.txt'
root = 'data/ARID/MAJOR-ARID-FEATS-VIDEOCHAT2/'
root2 = 'data/ARID/ARID-FEATS-VIDEOCHAT2/'
batch_size = 1

device = torch.device('cpu')#('cuda')

from Embeding_engineering.ARID_loader import AL
dataset_tr = AL(train, root, root2)
dl = torch.utils.data.DataLoader(dataset_tr, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
dataset = AL(val, root, root2)
vdl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

dataset_ts = AL(test, root, root2)
tst = torch.utils.data.DataLoader(dataset_ts, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
dataloader = {'train':dl, 'val':vdl, 'test':tst}



def get_arid_feats(phase):
    k = 0 
    eps= 1e-8
    
    for feat, output, index, name in dataloader[phase]:
            
            vid_feats = feat[0].to(device)
            
            

            f = vid_feats
  
            out = output.to(device)
            index = index.to(device)

            X = torch.cat((X, f), dim=0) if k > 0 else f
            Y = torch.cat((Y, out[0]), dim=0) if k > 0 else out[0]
            X_idx = torch.cat((X_idx, index), dim=0) if k > 0 else index
            
            k += 1
    print(X.size(), Y.size(), X_idx.size())
    X = X.reshape([X.size(0),X.size(1)*X.size(2)])
    Y = Y.reshape([Y.size(0),Y.size(1)*Y.size(2)])
    Y = Y.to(torch.float64)
    if phase == 'test':
           
            print(name)
            return X, Y, X_idx, name
    else:
        return X, Y, X_idx
   
