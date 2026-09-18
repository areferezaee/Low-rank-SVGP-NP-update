#Arefe-MAH
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader#, non_deterministic
from tqdm import trange
from utils import *
from torchvision import transforms
from GP_Error_Estimator.estimator import Estimator
from io import BytesIO
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gpytorch


from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
roc_auc_score,
confusion_matrix,
classification_report
)

from sklearn.preprocessing import label_binarize
from argparse import ArgumentParser
import torch
import time


from Embeding_engineering.ARID import get_arid_feats

# =========================
# Hyperparameters
# =========================

def get_parser():

   parser = argparse.ArgumentParser(description='Err_Est GP - trainer')
   parser.add_argument('--script-name', default='Err_Est')
   parser.add_argument('--dataset', type=str, default='ARID')
   parser.add_argument('--optimizer', default='adam')

   parser.add_argument('-lr', default=1e-5, type=float, help='learning rate')
   parser.add_argument('--natural-lr', default=.1, type=float,
                    help='natural GA learning rate. If not using stochastic updates - may use a value of 1.')
   parser.add_argument('--batch_size', type=int, default=1, help='batch size')
   parser.add_argument('--test-batch-size', type=int, default=1, help='test batch size')
   parser.add_argument('--train_strategy', type=str, default='joint')
   parser.add_argument('--kernel-function', type=str, default=['LinearKernel', 'LinearKernel', 'LinearKernel', 'LinearKernel', 'LinearKernel', 'LinearKernel'])
   parser.add_argument('--num-inducing-points', type=int, default=5)
   parser.add_argument('--model-path', type=str, default='model_path/Darklowlightmodel.pth')
   parser.add_argument('--OnlyTest-Model', type=str, default='NotYet', choices=['NotYet', 'Now'])
   parser.add_argument('--outputscale', type=float, default=1., help='output scale')
   parser.add_argument('--eval-every', type=int, default=1, help='num. epochs between test set eval')
   parser.add_argument('--seed', default=42, type=int, help='random seed')
   parser.add_argument('--num-workers', default=1, type=int, help='num wortkers')
   parser.add_argument('--gpus', type=str, default='0')

   return parser

# =========================
# Main
# =========================

class Main:
    def __init__(self, args):
        
        self.args = args
        set_seed(self.args.seed)
        
        self.n_token = 1
        self.device = torch.device('cpu')#('cuda')
        self.batch_size = args.batch_size
        self.mode= 'train'
        self.leng = 3072
        self.div = 1
        print(f"Using device: {self.device}")

        #======= Data =======
        

        if self.args.dataset =='ARID':
          


          self.X_train, self.Y_train, self.Xtrain_idx = get_arid_feats('train')
          self.X_val, self.Y_val, self.Xval_idx = get_arid_feats('val')   
          self.X_test, self.Y_test, self.Xtest_idx, self.name = get_arid_feats('test')
          self.ztest = self.X_test 
          self.Y_test_major = self.Y_test
          
          self.args.feature_length = torch.tensor([self.X_train.size(1)])#, self.X_train.size(2)])
          print('It is using for Xbar',self.X_train.size(1))#*self.X_train.size(2))
        self.allGT=[]
        self.allpreds=[]
        #======= Model =======
        # build initial model
        
        self.estimator = Estimator(self.args, self.device, pretrained=False)
        self.estimator.to(self.device)
        # ==========
        # Initialization Train
        # ==========
        self.num_epochs = 20
        self.writer = SummaryWriter()
        self.start_time = time.time()
        
        self.params_optimizer = optim.Adam(self.estimator.parameters(), lr=1e-3)
    # =========================
    # Compute Loss
    # =========================
    def major_loss(self, pred_z):
        
        
        mainY = self.Y_test_major
        predY = pred_z
        mse = torch.mean((mainY - predY) ** 2)
        mae = torch.mean(torch.abs(mainY - predY))
        
        predY = predY.reshape([predY.size(0), 96, 3072])
        
        #torch.save(predY, 'pred_bar_z/pred_bar_z_drink_6_17.pt')
        torch.save(predY, 'dlta_bar_Z/finaldlta_bar_z_'+ self.name[0])
        return mse, mae
    # =========================
    # train-val-test 
    # =========================
    
    def train(self, strategy):
       #estimator = estimator.cuda()   
       #max_grad = 100
       self.estimator.train()
       total_loss = 0.0
       if strategy == 'joint':
          self.params_optimizer.zero_grad()
       with torch.autograd.detect_anomaly():

             if strategy == 'hierarchical':
                self.params_optimizer.zero_grad()
             loss = self.estimator(self.X_train, self.Y_train.squeeze(1), self.Xtrain_idx, mode='train', tk_idx=0)
             
             if strategy == 'hierarchical':
                loss.backward(retain_graph=True)
                self.params_optimizer.step()
             
             total_loss += loss
       # optimize GP hyper-parameters 
       tot = self.n_token*self.div
       tot_loss = total_loss/tot
       if strategy == 'joint':
         
         tot_loss.backward(retain_graph=True )
         self.params_optimizer.step()
       torch.save(self.estimator,'model_path/Darklowlightmodel.pth')
       return tot_loss
       

    def val(self, strategy):
      self.estimator.eval()
      total_loss1 = 0.0
      total_loss2 = 0.0
      total_loss3 = 0.0
      
          
      loss1, loss2, loss3, mu = self.estimator(self.X_val, self.Y_val, self.Xval_idx, mode='val', tk_idx=0) 
      print(self.Y_val.mean().item(), self.Y_val.std().item())
      print('mu v mean', mu.mean().item(), mu.std().item())
      
      total_loss1 += loss1
      total_loss2 += loss2
      total_loss3 += loss3
      tot = self.n_token*self.div
      
      return total_loss1/tot, total_loss2/tot, total_loss3/tot
      

    def test(self, strategy):

      self.estimator.eval()
      total_loss1 = 0.0
      total_loss2 = 0.0
      total_loss3 = 0.0
      
         
      loss1, loss2, loss3, mu = self.estimator(self.X_test, self.Y_test, self.Xtest_idx, mode='test', tk_idx=0)
          
      total_loss1 += loss1
      total_loss2 += loss2
      total_loss3 += loss3
          
      
      
      pred_z = mu
      
      print(self.Y_test.mean().item(), self.Y_test.std().item())
      print('mu t mean', pred_z.mean().item(), pred_z.std().item())
      tot = self.n_token*self.div
      return total_loss1/tot, total_loss2/tot, total_loss3/tot, pred_z

   

    # === start ===
    def start(self):
      strategy = self.args.train_strategy
      for epoch in range(self.num_epochs):
        if epoch == (self.num_epochs - 1):
           self.mode= 'test'
           loss1, loss2, loss3, predz = self.test(strategy)
           mse, mae = self.major_loss(predz) 
           print('TEST:loss1, loss2, loss3', loss1, loss2, loss3)
           print('Pred bar z acc:mse, mae:', mse, mae)
        elif self.args.OnlyTest_Model=='Now':
           self.estimator = torch.load(self.args.model_path,  weights_only=False)
           loss1, loss2, loss3, predz = self.test(strategy)
           mse, mae = self.major_loss(predz) 
           print('TEST:loss1, loss2, loss3', loss1, loss2, loss3)
           print('Pred bar z acc:mse, mae:', mse, mae)
           break
        else:
  
           self.mode= 'train'
           loss = self.train(strategy)
           print('TRAIN:loss:', loss)
           self.mode= 'val'
           loss1, loss2, loss3 = self.val(strategy)
           print('VAL:lloss1, loss2, loss3', loss1, loss2, loss3)
      

         

def main():
    parser = get_parser()
    args = parser.parse_args()

    DarkLowligihtM = Main(args)
    DarkLowligihtM.start()


if __name__ == "__main__":
    main()

