import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi
import math
import numpy as np
import pylops_gpu





class Id(nn.Module):
    """
    Identity operator
    """
    
    def __init__(self):
        super().__init__()
        
         
    def forward(self, x): 
        return x
    
    
    def apply(self, x):
        return x
    
    
    def applyJacobianT(self, v, x=None):
        return v
    
    
    
    
class PlaneWave(nn.Module):
    
    def __init__(self, frequency):
        
        self.frequency = frequency
        
    
    def forward(self, Nx, Nz):
        
        # Propagates the wave over a distance N
        Vectx = torch.arange(Nx) - Nx/2
        Vectz = torch.arange(Nz) - Nz/2
            
        X, Z = torch.meshgrid(Vectx,Vectz)
        u = torch.exp(1j*(self.frequency[0]*Z + self.frequency[1]*X))
        
        return u
        
        
    def apply(self, Nx, Nz):
        with torch.no_grad():
            u = self.forward(Nx, Nz)        
        
        return u
    
    
class BatchedPlaneWave(nn.Module):
    def __init__(self, Nx, Nz):
        super(BatchedPlaneWave, self).__init__()
        '''
        For Nx, Nz fixed, compute PlaneWave propagation in space for given frequencies
        '''
        self.Nx = Nx
        self.Nz = Nz

    def forward(self, freqs):
        # Freqs is a [Nx2] array of x, z frequences
        Nx = self.Nx
        Nz = self.Nz
        r = len(freqs)
        Vectx = torch.arange(Nx) - Nx/2
        Vectz = torch.arange(Nz) - Nz/2
        X, Z = torch.meshgrid(Vectx, Vectz)
        X_, Z_ = torch.tile(X[None, ...], (r, 1, 1)), torch.tile(Z[None, ...], (r, 1, 1))
        return torch.exp(1j * (freqs[:, 0:1, None] * Z_ + freqs[:, 1:2, None] * X_))


class SelectorPatch1d(nn.Module):
    
    def __init__(self, sz, idxmin, idxmax):
        
        super().__init__()
        
        self.sizein = sz
        self.idxmin = idxmin
        self.idxmax = idxmax
        self.sizeout= idxmax-idxmin
        
        
    def forward(self, x):
        # x: Nb x K
        y = x[:, self.idxmin:self.idxmax]
        
        return y
    
    
    def apply(self, x):
        with torch.no_grad():
            y = self.forward(x)
        
        return y
    
    
    def applyJacobianT(self, v, x=None):
        # v: Nb x K
        with torch.no_grad():
            Nb = v.shape[0]
            y = torch.zeros(Nb, self.sizein, dtype=torch.complex64, device=v.device)
            y[:, self.idxmin:self.idxmax] = v
        
        return y
    
    
    

class SelectorPatch1d_par(nn.Module):
    
    def __init__(self, sz, num, idxmin, idxmax):
        
        super().__init__()
        
        self.sizein = sz
        self.num = num
        self.idxmin = idxmin
        self.idxmax = idxmax
        self.sizeout= idxmax-idxmin
        
        
    def forward(self, x):
        # x: Nb x num x K
        y = x[:, :, self.idxmin:self.idxmax]
        
        return y
    
    
    def apply(self, x):
        with torch.no_grad():
            y = self.forward(x)
        
        return y
    
    
    def applyJacobianT(self, v, x=None):
        # v: Nb x num x K
        with torch.no_grad():
            Nb = v.shape[0]
            y = torch.zeros(Nb, self.num, self.sizein, dtype=torch.complex64, device=v.device)
            y[:, :, self.idxmin:self.idxmax] = v
        
        return y
    
    
    
    
class StackMap(nn.Module):
    
    def __init__(self, module_list):
        
        super().__init__()
    
        self.module_list = module_list
        self.num = len(module_list)
        self.sizein = module_list[0].sizein    # 2D
        self.sizeout = module_list[0].sizeout  # 1D
        
    
    def forward(self, x):
        Nb = x.shape[0]
        y = torch.zeros([Nb, self.num, self.sizeout], dtype=torch.complex64, device=x.device)
        
        for idx in range(self.num):
            y[:, idx, :] = self.module_list[idx](x)
            
        return y
    
    
    def apply(self, x):
        with torch.no_grad():
            Nb = x.shape[0]
            y = torch.zeros([Nb, self.num, self.sizeout], dtype=torch.complex64, device=x.device)
        
            for idx in range(self.num):
                y[:, idx, :] = self.module_list[idx].apply(x)
            
        return y
    
    
    def applyJacobianT(self, v, x=None):
        Nb = v.shape[0]
        y = torch.zeros([Nb, *self.sizein], device=v.device)
        with torch.no_grad():
            if x is None:
                for idx in range(self.num):
                    y = y + self.module_list[idx].applyJacobianT(v[:, idx, :])
            else:
                for idx in range(self.num):
                    y = y + self.module_list[idx].applyJacobianT(v[:, idx, :], x)
                    
        return y
    
    
    
    
class LinOpGrad():
    
    def __init__(self, sizein, device):
        
        self.H = sizein[2]
        self.W = sizein[3]
        self.device = device
        
        self.Dop_0 = pylops_gpu.FirstDerivative(self.H*self.W, dims=(self.H, self.W), dir=0, device=device, togpu=(True, True))
        self.Dop_1 = pylops_gpu.FirstDerivative(self.H*self.W, dims=(self.H, self.W), dir=1, device=device, togpu=(True, True))
    
    
    def apply(self, x):
        x = x.to(torch.float)
        with torch.no_grad():
            x = x.to(self.device)
            out_0 = self.Dop_0*(x.view(-1))
            out_1 = self.Dop_1*(x.view(-1))
            out = torch.cat([torch.reshape(out_0, (1, self.H, self.W)), torch.reshape(out_1, (1, self.H, self.W))], dim=0)
            #out = out.to(vol.device)
            
        return out
        
        
    def applyJacobianT(self, y):
        
        with torch.no_grad():
            y = y.to(self.device)
            out = torch.reshape(self.Dop_0.H*(y[0,...].view(-1)), (self.H, self.W)) + torch.reshape(self.Dop_1.H*(y[1,...].view(-1)), (self.H, self.W))
            #out = out.to(y.device)
            out = out.unsqueeze(0)
            out = out.unsqueeze(0)
            
        return out # 1 x 1 x K x K