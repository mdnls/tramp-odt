from tramp.algos import ExpectationPropagation
from tramp.variables import SILeafVariable as O, SISOVariable as V
from tramp.channels import ProductChannel, GaussianChannel, ConvChannel
from tramp.priors import GaussianPrior, VonMisesPrior
from tramp.likelihoods import GaussianLikelihood
import numpy as np


class DiffractionChannel(ConvChannel):
    def __init__(self, dz, Nx, Ny, c=1):
        '''
        dz - size of step per layer in transverse direction
        Nx, Ny, Nz - length of each axis of scanning domain
        c - a scaling constant, eg c=k0 * n0
        '''
        freq = lambda x, y: (x ** 2 + y ** 2) / (c + np.sqrt(c ** 2 - x ** 2 - y ** 2))
        x_coords = np.stack([np.arange(Nx)] * Ny, axis=0)  # x axis indexes columns
        y_coords = np.stack([np.arange(Ny)] * Nx, axis=1)  # y axis indexes rows
        filter = np.fft.ifftn(np.exp(1j * freq(x_coords, y_coords) * dz))
        super().__init__(filter=filter, real=False)

class Teacher():
    def __init__(self, reg, dz, dim):
        '''
        reg - parameter ranging [0, infty) controlling regularization strength
        dz - step size in transverse axis
        dim - [Nx, Ny]
        '''
        self.l = reg
        self.dz = dz
        self.dim = dim
        self.shape = [2,] + list(self.dim)
        self.model = self.build_model()

    def build_model(self):
        D = DiffractionChannel(dz=self.dz, Nx=self.dim[1], Ny=self.dim[0], c=self.dim[0]*self.dim[1])
        z_prior = GaussianPrior(size=self.shape, mean=np.zeros(self.shape), var=1) @ V(id="x1") @ D @ V(id="z1")
        s_prior = VonMisesPrior(size=self.shape, b=self.l + 0j) @ V(id="s1")
        model = (z_prior + s_prior) @ ProductChannel(shape=self.shape, layer_idx=1) @ V(id="x2") @ GaussianChannel(var=1) @ O(id="y")
        model = model.to_model()
        return model

class Student():
    def __init__(self, sample, initvar, finvar, reg, dz, dim):
        self.sample = sample
        self.initvar = initvar
        self.finvar = finvar
        self.l = reg
        self.dz = dz
        self.dim = dim
        self.shape = [2,] + list(self.dim)
        self.model = self.build_model()

    def build_model(self):
        D = DiffractionChannel(dz=self.dz, Nx=self.dim[1], Ny=self.dim[0], c=self.dim[0]*self.dim[1])
        z_prior = GaussianPrior(size=self.shape, mean=self.sample['x1'], var=self.initvar) @ V(id="x1") @ D @ V(id="z1")
        s_prior = VonMisesPrior(size=self.shape, b=self.l + 0j) @ V(id="s1")
        model = (z_prior + s_prior) @ ProductChannel(shape=self.shape, layer_idx=1) @ V(id="x2") @ GaussianLikelihood(y=self.sample['y'], y_name="y", var=self.finvar)
        model = model.to_model()
        return model


if __name__ == "__main__":
    Nx = 2
    Ny = 2
    teacher = Teacher(reg=1, dz=0.01, dim=[Ny, Nx])
    sample = teacher.model.sample()
    student = Student(sample=sample, initvar=0.01, finvar=0.01, reg=1, dz=0.01, dim=[Ny, Nx])

    ep = ExpectationPropagation(student.model)
    ep.iterate(max_iter=10, damping=0)
    
    data_ep = ep.get_variables_data('all')
    print("True s:")
    print(sample['s1'])
    print("Estimated s:")
    print(data_ep['s1']['r'])
    print(data_ep)
