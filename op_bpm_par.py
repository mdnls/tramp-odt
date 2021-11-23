import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np
from op_utils import PlaneWave, SelectorPatch1d_par
import math
from math import pi
from easydict import EasyDict
import scipy.signal



def buildforwardBPM_par(Nx, Nz, thetas, beamtype='tukey', use_autograd=False):
    
    Nxext = 4*Nx  #the wave is propagated on a larger window
    Nzext = Nz
    '''
    4x padding in *only* the X direction. Represented by free space, scattering through padded space has no effect 
    '''

    # BPM cannot do angles larger or eq than abs(pi/4)
    Ntheta = thetas.shape[0] # parallelization is over theta. More thetas means more info content, similar to # measurements.
    # Coverage matters, broader the better. Should be 20 angles unif between -pi/4 to pi/4
    # Corresponds to augmented problem with independent/block operating forward operators

    lambda0 = 0.450
    dx = 0.1
    dz = 0.1

    n0 = 1.525

    Lx = dx*Nx
    Lz = dz*Nz
    Lxext = dx*Nxext
    Lzext = dz*Nzext

    SensorDist = -Lz/2   # BPM prop the wave from left to right
                         # this param refocuses the wave given sensordist
    padfact = Nxext/Nx
    siz = [Nx,Nz]
    k0 = 2*pi/lambda0
    k = k0*n0

    uin = torch.zeros(Nxext, Nzext, Ntheta, dtype=torch.complex64)

    vec_z = dz*torch.arange(-Nzext/2 + 1, Nzext/2 + 1)
    vec_x = dx*torch.arange(-Nxext/2 + 1, Nxext/2 + 1)
    XX, ZZ = torch.meshgrid(vec_x, vec_z)

    sigmaBeam = 0.5

    for kk in range(Ntheta):
        angx = thetas[kk]
        curr_k = [k*dz*math.cos(angx), k*dz*math.sin(angx)]

        # Can assume our beam is ain=1
        if (beamtype == 'GB'):
            Bx = sigmaBeam*Lxext
            Bx = Bx/math.cos(angx)
            Bz = sigmaBeam*Lzext
            Bz = -Bz/math.sin(angx)
            ain = torch.exp(-torch.pow(XX/Bx + ZZ/Bz, 2))
        
        elif (beamtype == 'tukey'):
            tukeywin = torch.unsqueeze(torch.tensor(scipy.signal.tukey(Nxext, sigmaBeam)), 1)
            ain = tukeywin.expand(-1, Nzext)
            
        else:
            # ain = 1 is a good choice for idealized setting. Controls amplitude of the initial wave.
            ain = 1.0
        
        u = PlaneWave(curr_k)
        uin[:,:,kk] = ain*(u.apply(Nxext,Nzext))
    

    uin = torch.squeeze(uin[:,int(uin.shape[1]/2)-1,:], dim=1)
    #uin = uin[:,int(uin.shape[1]/2)-1,:]

    '''
    What one would expect: 
    1. Sample (ie. evaluate coordinates of) initial wave at z=-L/2 
    
    Empirical heuristics: 
    1. Sample initial wave at z=0
    2. Propagate through free space from z=0 *backward* to z=-L/2. 
    3. Because there is no scattering, propagation through many z slices can be done all at once as a power of 
        the fixed operator.
    '''
    distUin = -Lz/2

    uem = uin[int(uin.shape[0]/2) - int(Nx/2):int(uin.shape[0]/2) + int(Nx/2), :]   # Measurements without the sample
    
    # SetFields
    distuM = 0
    tmp_Nx = uin.shape[0]
    tmp_padfact = 1
    dkx = 2*pi/(tmp_padfact*tmp_Nx*dx)
    tmp_Nx = tmp_padfact*tmp_Nx

    Kxx = torch.pow(dkx*torch.cat([torch.arange(int(tmp_Nx/2)), torch.arange(-int(tmp_Nx/2), 0)]), 2)

    k = 2*pi/(lambda0)*n0
    complex_Kxx = Kxx.numpy() + 0j
    dphi = torch.tensor(np.real(complex_Kxx/(k + np.real(np.sqrt(k**2 - complex_Kxx)))))

    uin_fft = torch.fft.fft(uin, dim=0)

    rep = torch.exp(-1j*(dphi*distUin))

    fft_prod = uin_fft*torch.unsqueeze(rep, 1).expand(rep.shape[0], Ntheta)
    uin_out = torch.fft.ifft(fft_prod, dim=0)

    '''
    SelectorPatch is removing padding by 'selecting' a subset of coordinates 
    '''
    M = SelectorPatch1d_par(Nxext, Ntheta, int(Nxext/2) - int(Nx/2), int(Nxext/2) + int(Nx/2))

    '''
    _par is parallelized over thetas 
    '''
    #H = OpBPM_par(siz, Lx, Lz, k0, k, uin_out, SensorDist, padfact, M, thetas, use_autograd=use_autograd)
    H = None
    # setting up the params dictionary
    param_dict = {"Nx" : Nx, "Nz" : Nz, "Nxext" : Nxext, "Nzext" : Nzext,
                  "thetas" : thetas, "lambda0" : lambda0, "dx" : dx, "dz" : dz,
                  "n0" : n0, "Lx" : Lx, "Lz" : Lz, "Lxext" : Lxext, "Lzext" : Lzext,
                  "SensorDist": SensorDist, "padfact" : padfact, "siz" : siz, "k0" : k,
                  "k" : k, "sigmaBeam" : sigmaBeam, "distUin" : distUin}
    params = EasyDict(param_dict)
    
    return M, params, uem, uin_out




class BPM_Func_par(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, Nx, Nz, Nx_ext, uin, f1, k0, drz, cos_thetas, pad_sz, f2, M, f3, f4):
        # x: N x C x H x W
        
        ctx.cos_thetas = cos_thetas
        Nb = x.shape[0]
        num = cos_thetas.shape[0]
        #expx = torch.zeros([Nz, Nb, int(Nx_ext)], dtype=torch.cfloat, device=x.device)
        expx = torch.zeros([Nb, num, int(Nx_ext), Nz], dtype=torch.complex64, device=x.device)
        #u = torch.zeros([Nz, Nb, uin.shape[0]], dtype=torch.cfloat, device=x.device)
        u = torch.zeros([Nb, num, uin.shape[1], Nz], dtype=torch.complex64, device=x.device)
        
        #uouthat = uin.expand(Nb, uin.shape[0])
        uouthat = uin.expand(Nb, num, uin.shape[1])
        
        f1 = f1.expand(Nb, num, f1.shape[0])
        
        cos_thetas = cos_thetas.expand(Nb, num, Nx)
        
        uouthat = uouthat.to(x.device)
        f1 = f1.to(x.device)
        cos_thetas = cos_thetas.to(x.device)
        
        x = x.expand(Nb, num, Nx, Nz)
        
        for ind_z in range(Nz):
            uouthat = torch.fft.ifft((torch.fft.fft(uouthat, dim=2))*f1, dim=2)   
            curr_expx = F.pad(torch.exp(1j*k0*x[:,:, :, ind_z]*drz/cos_thetas), [pad_sz, pad_sz], value=1)
            uouthat = uouthat*curr_expx
            
            expx[:, :, :, ind_z] = curr_expx.clone()
            u[:, :, :, ind_z] = uouthat.clone()
            
        f2 = f2.expand(Nb, num, f2.shape[0])
        f2 = f2.to(x.device)
        uouthat = torch.fft.ifft((torch.fft.fft(uouthat, dim=2))*f2, dim=2)
        uouthat = M(uouthat) 
        
        f3 = f3.expand(Nb, num, f3.shape[0])
        f3 = f3.to(x.device)
        f4 = f4.expand(Nb, num, f4.shape[0])
        f4 = f4.to(x.device)
        
        #ctx.save_for_backward(expx, M, f3, u, f1, k0, drz, theta, f4, uin)
        ctx.Nx = Nx
        ctx.Nz = Nz
        ctx.Nx_ext = Nx_ext
        ctx.M = M
        ctx.f3 = f3
        ctx.f1 = f1
        ctx.k0 = k0
        ctx.drz = drz
        ctx.f4 = f4
        ctx.uin = uin.to(x.device)
        ctx.expx = expx
        ctx.u = u
        ctx.num = num
        

        return uouthat


    @staticmethod
    def backward(ctx, r):
        
        Nb = r.shape[0]
        #Nx, Nz, Nx_ext, expx, M, f3, u, f1, k0, drz, theta, f4, uin = ctx.saved_tensors
        Nx = ctx.Nx
        Nz = ctx.Nz
        Nx_ext = ctx.Nx_ext 
        M = ctx.M
        f3 = ctx.f3 
        f1 = ctx.f1
        k0 = ctx.k0
        drz = ctx.drz 
        f4 = ctx.f4 
        uin = ctx.uin
        expx = ctx.expx 
        u = ctx.u
        num = ctx.num
        cos_thetas = ctx.cos_thetas
        
        r = M.applyJacobianT(r)
        r = torch.fft.ifft((torch.fft.fft(r, dim=2))*f3, dim=2)   
        gradD = torch.zeros([Nb, num, int(Nx_ext), Nz], device=r.device)
        cos_thetas = cos_thetas.expand(Nb, num, int(Nx_ext))
        cos_thetas = cos_thetas.to(r.device)
            
        for ind_z in range(Nz-1, 0, -1):
            
            s = torch.conj(torch.fft.ifft((torch.fft.fft(u[:, :, :, ind_z-1], dim=2))*f1, dim=2))
            s = s*r*torch.conj(1j*(expx[:, :, :, ind_z])*(k0*drz/cos_thetas))
                
            gradD[:, :, :, ind_z] = torch.real(s)
                
            r = (torch.conj(expx[:, :, :, ind_z]))*r
            r = torch.fft.ifft((torch.fft.fft(r, dim=2))*f4, dim=2)
        
        s = torch.conj(torch.fft.ifft((torch.fft.fft(uin.expand(Nb, num, uin.shape[1]), dim=2))*f1, dim=2))
        s = s*r*torch.conj(1j*(expx[:, :, :, 0]*(k0*drz/cos_thetas)))
        
        gradD[:, :, :, 0] = torch.real(s)
        gradD = gradD[:, :, int(Nx_ext/2) - int(Nx/2):int(Nx_ext/2) + int(Nx/2), :]
        gradD = torch.sum(gradD, dim=1, keepdim=True)

        return gradD, None, None, None, None, None, None, None, None, None, None, None, None, None, None
    
    
    
    
class OpBPM_par(nn.Module):
    
    def __init__(self, siz, Lx, Lz, k0, k, uin, SensorDist, padfact, M, thetas, use_autograd=False):

        super().__init__()
        
        self.sizein = siz
        self.Nx = siz[0]
        self.Nz = siz[1]
            
        self.k = k
        self.k0 = k0
        self.Lz = Lz
        self.Lx = Lx
            
        self.drx = self.Lx/self.Nx
        self.drz = self.Lz/self.Nz
            
        self.rx = self.drx*torch.arange(-self.Nx/2+1, self.Nx/2 + 1)
        self.rx = self.rx.unsqueeze(1)
        
        self.rz = self.drz*torch.arange(1, self.Nz+1) 
        self.rz = self.rz.unsqueeze(0)
            
        self.Rxx = self.rx
            
        self.dkx = 2*pi/self.Lx
        self.dkz = 2*pi/self.Lz
            
        
        self.kx = self.dkx*torch.cat([torch.arange(self.Nx/2), torch.arange(-self.Nx/2, 0)])
        self.kx = self.kx.unsqueeze(1)
        
        self.kz = self.dkz*torch.cat([torch.arange(self.Nz/2), torch.arange(-self.Nz/2, 0)])
        self.kz = self.kz.unsqueeze(1)
        
        Kxx = torch.pow(self.kx, 2)
        complex_Kxx = Kxx.numpy() + 0j
            
        self.dphi = torch.tensor(np.real(complex_Kxx/(self.k + np.sqrt((self.k)**2 - complex_Kxx)))) 
            
        self.uin = torch.transpose(uin, 0, 1) # num x K
        self.thetas = thetas # num
        
        self.cos_thetas = torch.unsqueeze(torch.cos(self.thetas), 1)
        
        self.SensorDist = SensorDist
            
        self.padfact = padfact
        self.Nx_ext = self.padfact*self.Nx
        self.dkx_ext = 2*pi/(self.Nx_ext*self.drx)
            
        Kxx_ext = torch.pow(self.dkx_ext*torch.cat([torch.arange(self.Nx_ext/2), torch.arange(-self.Nx_ext/2, 0)]), 2)
        complex_Kxx_ext = Kxx_ext.numpy() + 0j    
        self.dphi_ext = torch.tensor(np.real(complex_Kxx_ext/(self.k + np.sqrt((self.k)**2 - complex_Kxx_ext))))
        
        self.f1 = torch.exp(-1j*(self.dphi_ext*self.drz))
        self.f2 = torch.exp(-1j*(self.dphi_ext*self.SensorDist))
        self.f3 = torch.exp(-1j*(-self.dphi_ext*self.SensorDist))
        self.f4 = torch.exp(-1j*(-self.dphi_ext*self.drz))
        
        self.M = M
        self.sizeout = self.M.sizeout
        
        self.pad_sz = int((self.Nx_ext-self.Nx)/2)
        
        self.use_autograd = use_autograd
        
    def forward(self, x):
        # x: N x C x H x W
        if (self.use_autograd):
            Nb = x.shape[0]
            num = self.cos_thetas.shape[0]
            uouthat = self.uin.expand(Nb, num, self.uin.shape[1])
            f1 = self.f1.expand(Nb, num, self.f1.shape[0])
            cos_thetas = self.cos_thetas.expand(Nb, num, self.Nx)
            uouthat = uouthat.to(x.device)
            f1 = f1.to(x.device)
            cos_thetas = cos_thetas.to(x.device)
        
            x = x.expand(Nb, num, self.Nx, self.Nz)
        
            for ind_z in range(self.Nz):
                uouthat = torch.fft.ifft((torch.fft.fft(uouthat, dim=2))*f1, dim=2)
                expx = F.pad(torch.exp(1j*self.k0*x[:,:, :, ind_z]*self.drz/cos_thetas), [self.pad_sz, self.pad_sz], value=1)
                uouthat = uouthat*expx

            f2 = self.f2.expand(Nb, num, self.f2.shape[0])
            f2 = f2.to(x.device)
            uouthat = torch.fft.ifft((torch.fft.fft(uouthat, dim=2))*f2, dim=2)
            uouthat = self.M(uouthat) 
        
        else:
            
            uouthat = BPM_Func_par.apply(x, self.Nx, self.Nz, self.Nx_ext, self.uin, self.f1, self.k0, self.drz, self.cos_thetas, self.pad_sz, self.f2, self.M, self.f3, self.f4)
        
        return uouthat  # Nb x num x K
        
        
    def apply(self, x):
        
        with torch.no_grad():
            
            Nb = x.shape[0]
            num = self.cos_thetas.shape[0]
            self.ctx_expx = torch.zeros([Nb, num, int(self.Nx_ext), self.Nz], dtype=torch.complex64, device=x.device)
            self.ctx_u = torch.zeros([Nb, num, self.uin.shape[1], self.Nz], dtype=torch.complex64, device=x.device)
            
            uouthat = self.uin.expand(Nb, num, self.uin.shape[1])
            self.ctx_f1 = self.f1.expand(Nb, num, self.f1.shape[0])
            cos_thetas = self.cos_thetas.expand(Nb, num, self.Nx)
        
            uouthat = uouthat.to(x.device)
            self.ctx_f1 = self.ctx_f1.to(x.device)
            cos_thetas = cos_thetas.to(x.device)
            
            x = x.expand(Nb, num, self.Nx, self.Nz)
        
            for ind_z in range(self.Nz):
                uouthat = torch.fft.ifft((torch.fft.fft(uouthat, dim=2))*self.ctx_f1, dim=2)    
                curr_expx = F.pad(torch.exp(1j*self.k0*x[:,:, :, ind_z]*self.drz/cos_thetas), [self.pad_sz, self.pad_sz], value=1)
                uouthat = uouthat*curr_expx
            
                self.ctx_expx[:, :, :, ind_z] = curr_expx.clone()
                self.ctx_u[:, :, :, ind_z] = uouthat.clone()
            
             
            f2 = self.f2.expand(Nb, num, self.f2.shape[0])
            f2 = f2.to(x.device)
            uouthat = torch.fft.ifft((torch.fft.fft(uouthat, dim=2))*f2, dim=2)
            uouthat = self.M(uouthat) 
        
            self.ctx_f3 = self.f3.expand(Nb, num, self.f3.shape[0])
            self.ctx_f3 = self.ctx_f3.to(x.device)
            self.ctx_f4 = self.f4.expand(Nb, num, self.f4.shape[0])
            self.ctx_f4 = self.ctx_f4.to(x.device)
            self.ctx_uin = self.uin.to(x.device)
            
        return uouthat
    
    
    def applyJacobianT(self, v, x=None):
        
        with torch.no_grad():
            if x is not None:
                uouthat = self.apply(x)
        
            Nb = v.shape[0]
            num = self.cos_thetas.shape[0]
            cos_thetas = self.cos_thetas.expand(Nb, num, int(self.Nx_ext))
            cos_thetas = cos_thetas.to(v.device)
            
            v = self.M.applyJacobianT(v)
            v = torch.fft.ifft((torch.fft.fft(v, dim=2))*self.ctx_f3, dim=2)   
            gradD = torch.zeros([Nb, num, int(self.Nx_ext), self.Nz], device=v.device)
            
            for ind_z in range(self.Nz-1, 0, -1):
            
                s = torch.conj(torch.fft.ifft((torch.fft.fft(self.ctx_u[:, :, :, ind_z-1], dim=2))*self.ctx_f1, dim=2))
                s = s*v*torch.conj(1j*(self.ctx_expx[:, :, :, ind_z])*(self.k0*self.drz/cos_thetas))
                
                gradD[:, :, :, ind_z] = torch.real(s)
                
                v = (torch.conj(self.ctx_expx[:, :, :, ind_z]))*v
                v = torch.fft.ifft((torch.fft.fft(v, dim=2))*self.ctx_f4, dim=2)
            
            s = torch.conj(torch.fft.ifft((torch.fft.fft(self.ctx_uin.expand(Nb, num, self.ctx_uin.shape[1]), dim=2))*self.ctx_f1, dim=2))
            s = s*v*torch.conj(1j*(self.ctx_expx[:, :, :, 0]*(self.k0*self.drz/cos_thetas)))
        
            gradD[:, :, :, 0] = torch.real(s)
            gradD = gradD[:, :, int(self.Nx_ext/2) - int(self.Nx/2):int(self.Nx_ext/2) + int(self.Nx/2), :]
            gradD = torch.sum(gradD, dim=1, keepdim=True)
                    
        return gradD

        
        
if __name__ == "__main__":
    cnf = {
        "dx": 0.1,
        "dz": 0.1,
        "n0": 1.525,
        "lambda0": 0.450,
        "n_thetas": 20,
        "sigmaBeam": 0.5,
        "x_pad_factor": 4,
        "z_pad_factor": 1,
        "Nx": 16,
        "Nz": 16
    }
    M, params, uem, uin_out = buildforwardBPM_par(cnf['Nx'], cnf['Nz'], torch.FloatTensor(np.linspace(-np.pi / 4, np.pi / 4, 20)),
                                             beamtype="", use_autograd=False)
    b = OpBPM_par(siz=[cnf['Nx'], cnf['Nz']], Lx=params['Lx'], Lz=params['Lz'], k=params['k'], uin=uin_out,
                  SensorDist=params['SensorDist'], padfact=params['padfact'], M=M, thetas=params['thetas'], k0=params['k0'])
    x = torch.FloatTensor(np.random.normal(size=(1, cnf['n_thetas'], cnf['Nx'], cnf['Nz'])))
    b.apply(x)
    print('F')

    
    
    

    
    
