import numpy as np
import torch
from op_utils import BatchedPlaneWave
from op_bpm_par import buildforwardBPM_par
from tramp.channels import BatchConvChannel, ProductChannel
from tramp.variables import SISOVariable as V, SILeafVariable as O
from tramp.priors import GaussianPrior, VonMisesPrior
from tramp.likelihoods import GaussianLikelihood

class BPM():
    def __init__(self, cnf):
        '''
        Coordinate system:

                x
          Lx/2  /\
                |   .#.
              0 |   ###
                |   '#'
         -Lx/2 -|----|----> z
              -Lz/2  0  Lz/2

        '''
        cnf['thetas'] = np.linspace(-np.pi/4, np.pi/4, cnf['n_thetas'])
        cnf['Lx'] = cnf['dx'] * cnf['Nx']
        cnf['Lz'] = cnf['dz'] * cnf['Nz']
        cnf['Nx_pad'] = cnf['Nx'] * cnf['x_pad_factor'] # (!) #_pad previously *ext
        cnf['Lx_pad'] = cnf['dx'] * cnf['Nx_pad']
        cnf['Nz_pad'] = cnf['Nz'] * cnf['z_pad_factor']
        cnf['Lz_pad'] = cnf['dz'] * cnf['Nz_pad']
        cnf['sensorDist'] = -cnf['Lz']/2
        cnf['dims'] = [cnf['Nx'], cnf['Nz']] # (!)dims previously siz
        cnf['k0'] = 2 * np.pi / cnf['lambda0']
        cnf['k'] = cnf['k0'] * cnf['n0']
        self.cnf = cnf
        self.diffraction = self._diffraction_spectrum()
        
    def _init_amplitude(self):
        return 1

    def _diffraction_spectrum(self):
        # cnf to context:
        dx, dz, n0, lambda0, n_thetas, sigmaBeam, x_pad_factor, z_pad_factor, Nx, Nz, \
        thetas, Lx, Lz, Nx_pad, Lx_pad, Nz_pad, Lz_pad, sensorDist, dims, k0, k = self.cnf.values()

        dkx = 2 * np.pi / (Nx_pad * dx)
        Kxx = torch.pow(dkx * torch.cat([torch.arange(Nx_pad//2), torch.arange(-Nx_pad//2, 0)]), 2).to(torch.complex64)

        k = 2 * np.pi / lambda0 * n0
        dphi = torch.real(Kxx / (k + np.real(np.sqrt(k ** 2 - Kxx))))

        diffraction_spectrum = torch.exp(-1j * (dphi * dz)) # this is f1
        return diffraction_spectrum

    def _init_illumination(self):
        # cnf to context:
        dx, dz, n0, lambda0, n_thetas, sigmaBeam, x_pad_factor, z_pad_factor, Nx, Nz, \
        thetas, Lx, Lz, Nx_pad, Lx_pad, Nz_pad, Lz_pad, sensorDist, dims, k0, k = self.cnf.values()

        # compute initial illumination by backpropagating from x=0 to x=-Lz/2
        # (!) thetas are now indexed by the first dimension, like a batch dimension

        k_per_theta = torch.FloatTensor(np.stack((k * dz * np.cos(thetas), k * dz * np.sin(thetas)), axis=1))
        wave_model = BatchedPlaneWave(Nx_pad, Nz_pad)
        ain = self._init_amplitude()
        init_wave = ain * wave_model(k_per_theta)
        uin = init_wave[:, :, Nz//2 - 1]

        lwr = Nx_pad//2 - Nx//2
        upr = Nx_pad//2 + Nx//2
        uem = uin[:, lwr:upr]

        rep = torch.pow(self._diffraction_spectrum(), -Nz/2)
        uin_fft = torch.fft.fft(uin, dim=1)

        fft_prod = uin_fft * torch.unsqueeze(rep, 0).expand(n_thetas, Nx_pad)
        uin_out = torch.fft.ifft(fft_prod, dim=1)

        return uem, uin_out

    def _init_graph(self):
        # 1. Create a MCC layer representing the diffraction operation
        # 2. Initialize variables at each step of z representing phases
        # 3. Wire these things together

        '''
        Data of interest: uouthat, of shape
        Convolution step:
            - Spectrum is ctx_f1
        '''

        uem, uin = self._init_illumination()
        model = GaussianLikelihood(y=uin, var=0.01)
        dim = [2, self.cnf['n_thetas'], self.cnf['Nx']]

        phase_variables = [VonMisesPrior(size=dim, b=100) @ V(id=f"D{i}") for i in range(self.cnf['Nz'])]
        for ind_z in range(self.Nz):
            model = model @ BatchConvChannel(filter=self.diffraction, N=self.cnf['n_thetas'], real=False) @ V(id=f"z{ind_z}")
            model = (model + phase_variables[ind_z]) @ ProductChannel(shape=dim, layer_idx=ind_z)

        return model

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
    _, _, uem, uin_out = buildforwardBPM_par(cnf['Nx'], cnf['Nz'], np.linspace(-np.pi/4, np.pi/4, 20), beamtype="", use_autograd=False)
    B = BPM(cnf)
    uem2, uin_out2 = B._init_illumination()
    uin_out = uin_out.T
    uem = uem.T
    '''
    TODO:
    Compare uem and uin_out to their rewritten counterparts and debug the difference. 
    '''
    print("F")
