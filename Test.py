from tramp.algos import ExpectationPropagation
from tramp.variables import SILeafVariable as O, SISOVariable as V
from tramp.channels import ProductChannel, MC_ProductChannel, GaussianChannel, ConvChannel
from tramp.priors import GaussianPrior, VonMisesPrior
from tramp.likelihoods import GaussianLikelihood
import numpy as np
from tramp.utils.misc import complex2array, array2complex


z_prior = GaussianPrior(size=(2, 10), mean=np.zeros((2, 10)), var=1)
s_prior = VonMisesPrior(size=(2, 10), b=100j)

z = np.random.normal(scale=1, size=(2, 10))

s = np.ones((2, 10))
s[0, :] = 0

y_cpx = array2complex(z) * array2complex(s)
y = complex2array(y_cpx)

student = ((z_prior @ V(id="z")) + (s_prior @ V(id="s"))) @ \
          MC_ProductChannel(shape=(2, 10), n_samples=10000) @ V(id="y") @ GaussianLikelihood(y=y, var=0.01)
student = student.to_model_dag()
student = student.to_model()


ep = ExpectationPropagation(student)
ep.iterate(max_iter=10, damping=0)

data_ep = ep.get_variables_data('all')
print("Estimated: ")
for k, v in data_ep.items():
    print(k)
    print(v['r'])
print("True z: ")
print(z)
print("True s: ")
print(s)
print("True y: ")
print(y)
