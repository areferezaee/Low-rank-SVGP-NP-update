from gpytorch import kernels
from gpytorch import constraints
import gpytorch
import torch
from torch import nn
import torch.nn.functional as F
import gpytorch


class GPModel(gpytorch.Module):

    def __init__(self, jitter_val=1e-3):
        super().__init__()
        # mean and cov functions
        self.jitter_val = jitter_val
        self.num_modules = 9

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        
        g = int(x1.size(1)/self.num_modules)#256#256#512
        
        mean_x = self.mean_module(x2)
       
        covars = []

        for i in range(self.num_modules):
             start = i * g
             end = (i + 1) * g

             covar = getattr(self, f"covar_module{i}")(x1[:, start:end], x2[:, start:end]).add_jitter(jitter_val=self.jitter_val).to_dense()

             covars.append(covar)

             covar_x = sum(covars)

       
        
        
        return mean_x, covar_x

    def _set_params(self, outputscale=1., lengthscale=1.):
        # init hyperparameters
        for helper in range(self.num_modules):
          module = getattr(self, f"covar_module{helper}")

          module.outputscale = outputscale

          if self.kernel_function[helper] == 'LinearKernel':
             module.base_kernel.variance = 1e-7
        

        
     
class CoreGPModel(GPModel):
    def __init__(self, kernel_function, jitter_val=1e-3):
        super(CoreGPModel, self).__init__(jitter_val)

        self.mean_module = gpytorch.means.ConstantMean()
        self.kernel_function = []
        for hh in range(self.num_modules):
          self.kernel_function.append("LinearKernel")
        for hh in range(self.num_modules):
          if self.kernel_function[hh] == "LinearKernel":
            setattr(self, f"ker_fun{hh}", kernels.LinearKernel())

          setattr(self, f"covar_module{hh}", gpytorch.kernels.ScaleKernel( getattr(self, f"ker_fun{hh}")))


        
        
       
       
