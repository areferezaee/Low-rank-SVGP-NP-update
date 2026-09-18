import torch.nn as nn
from utils import *
from gpytorch.models import ApproximateGP, GP
import gpytorch
from GP_Error_Estimator.interface import Interface_VI
import torch

class Model(gpytorch.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.device = torch.device('cpu')#('cuda')
        self.criterion = nn.CrossEntropyLoss()
        self.Interface_VI = Interface_VI(args, self.device)

    def _init_Xbar(self, X, Y):
        raise NotImplementedError("not yet implemented")

    def forward(self, x, y, x_idx, mode='train', tk_idx=0):
        raise NotImplementedError("not yet implemented")


  

class Estimator(Model):
    def __init__(self, args, device, pretrained=True):
        super(Estimator, self).__init__(args)
        self.learn_location = 'True'

        Xbar_dim = (self.args.num_inducing_points, self.args.feature_length[0])#[-1])#, self.args.feature_length[1])
        if self.learn_location:
            self.Xbar = nn.Parameter(torch.randn(Xbar_dim), requires_grad=True)
        else:
            self.Xbar = torch.randn(Xbar_dim).to(device)
        
    def _init_Xbar(self, X, Y):
        
          with torch.no_grad():
            
            num_inducing = self.args.num_inducing_points
            
            Xbar = self.Xbar 
            self.Xbar = Xbar
    
    def forward(self, x, y, x_idx, mode='train', tk_idx=0):       
      if mode == 'train':
        z = x
        lengthscale = 10.
        num_inducing_inputs = self.args.num_inducing_points
        
        self.Interface_VI.setting(kernel_function=self.args.kernel_function, num_inducing_points=num_inducing_inputs,lengthscale=lengthscale,
                               natural_lr=self.args.natural_lr, outputscale=self.args.outputscale,
                               dtype=self.Xbar.dtype, num_data=self.args.feature_length[-1])
                               #low_rank_rank=4,
                               #low_rank_jitter=1e-5)
                              
        
        
        loss = self.Interface_VI.train_core(z, y, x_idx, self.Xbar,tk_idx=0)
        return loss

      else:  
        z = x
            
        with torch.no_grad():
             mu, sigma = self.Interface_VI.eval_core(z, self.Xbar)
             mse = torch.mean((y - mu) ** 2)
             mae = torch.mean(torch.abs(y - mu))
             eps = 1e-8

             nlpd = 0.5 * ( torch.log(2 * torch.pi * sigma + eps) + (y - mu) ** 2 / (sigma + eps))
             nlpd = nlpd.mean()
             if mode == 'test':
               print("\n===== Nescessary for Alpha in teseting one sample=====")
               gamma = mu.max() - mu.min()
               betta = torch.sqrt(sigma.sum()) - 171.
               r = gamma/betta
               
               if r < 20. and betta >= 0.81:
                  alpha = 0.4 #0.3 #0.5
               else:
                  alpha = .1 #1.1 #1.11 #.9
               print('Alpha is:', alpha)
        
        return mse, mae, nlpd, mu





  

    
