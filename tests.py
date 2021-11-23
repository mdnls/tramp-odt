import unittest
from BPM import BPM
from op_bpm_par import buildforwardBPM_par
import numpy as np

class WidgetTestCase(unittest.TestCase):
    def setUp(self):
        self.cnf = {
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

    def test_init(self):
        init1 = buildforwardBPM_par(Nx=self.cnf['Nx'], Nz=self.cnf['Nz'],
                                    thetas=np.linspace(-np.pi/4, np.pi/4, self.cnf['n_thetas']),
                                    beamtype="", use_autograd=False)
        uem1 = init1[2].T
        u_init1 = init1[3].T
        B = BPM(self.cnf)
        uem2, u_init2 = B._init_illumination()
        self.assertTrue(np.allclose(uem1, uem2))
        self.assertTrue(np.allclose(u_init1, u_init2))

