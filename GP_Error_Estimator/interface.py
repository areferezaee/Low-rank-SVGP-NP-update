from GP_Error_Estimator.primary_mdl import Primary_MDL
from torch import nn
import torch.optim as optim
from utils import *
import gpytorch
import torch

class Interface(gpytorch.Module):
    def __init__(self, args, device):
        super(Interface, self).__init__()
        self.model = None
        self.device = None
        self.args = args
        self.id = 0
        self.depth = 0
        self.num_data = 1

        self.X_support = None
        self.Y_support = None
        self.state = None




class Interface_VI(Interface):



    def setting(self, kernel_function, num_inducing_points, lengthscale, natural_lr,
                  outputscale, dtype, num_data):

        
        self.num_data = num_data
        
        

        dtype=torch.float32
        
        self.model = Primary_MDL(kernel_func=kernel_function,dtype=dtype,num_inducing_points=num_inducing_points,
                                num_data=num_data,natural_lr=natural_lr)
                                #low_rank_rank=4,
                                #low_rank_jitter=1e-5)
        self.model.model._set_params(outputscale=outputscale, lengthscale=lengthscale)

        self.model.to(self.device)
        

    def train_core(self, X, Y, batch_idx, Z, tk_idx):
        
        train_data = torch.cat((Z, X), dim=0)
        
        loss = - self.model.forward_mll(train_data, Y, batch_idx)

        avg_loss = loss.item() / self.num_data
        
       
        self.model.ELBO.update()  
        #print("AFTER UPDATE eta:",self.model.ELBO.eta[tk_idx].abs().max().item())
        return loss / self.num_data
        

    def eval_core(self, X, Z):
        
        X_star = torch.cat((Z, X), dim=0)
       
        mu, sigma = self.model.predictive_posterior(X_star)

        return mu, sigma

