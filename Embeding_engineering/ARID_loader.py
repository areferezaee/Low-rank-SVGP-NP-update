import torch
import torch.utils.data as data_utl

import numpy as np
import random

import os
#import lintel

import json

import torch
#from Embeding_engineering.comp_output import get_delta_Z

class AL(data_utl.Dataset):

    def __init__(self, split_file, root, secondroot):
        with open(split_file, 'r') as f:
            self.data = f.readlines()
        
        self.maindata = []
        self.feats = []
        self.all_feats = []
        self.all_normgrads = []
        self.normgrads = []
        self.all_gts = []
        self.gts = []
        self.all_feats2 = []
        for o in range(len(self.data)):
            line = self.data[o].strip()
            #print(line, o)
            if not line:
                continue
            parts = line.split()
            entry = parts[0]
            # Determine feature file names for each dataset type
            feature_name = 'z_'+entry
            #ablation = 'QFormer_'+entry
            
            #normalized_grad = 'normalized_grad_z_'+entry
            gt_name = 'z_gt_'+entry
            
            feat_path = os.path.join(root, feature_name)
            
            
            gt_path = os.path.join(root, gt_name)
            
            if not os.path.exists(feat_path):
                    print(feat_path)
                    continue
            totalfeat = torch.load(feat_path)            
            totalgts = torch.load(gt_path)
            self.all_feats.append(totalfeat)
            self.all_gts.append(totalgts)
            self.maindata.append(line)
            self.feats.append(feature_name)
            self.gts.append(gt_name)

        
       
        
        self.split_file = split_file
        self.root = root
        
        
        
    def __getitem__(self, index):
        eta = 1.0
        feat = self.feats[index] 
        gts = self.gts[index]


        gtz = self.all_gts[index]

        dff = self.all_feats[index]
        self.delta_z = gtz - dff

        
        return dff, self.delta_z, index, feat
        
    def __len__(self):
        return len(self.maindata)#.keys())

    


    
