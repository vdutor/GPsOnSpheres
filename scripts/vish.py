import numpy as np
import gpflow

from gspheres.kernels.spherical_matern import SphericalMatern
from gspheres.fundamental_set import num_harmonics
from gspheres.vish import SphericalHarmonicFeatures, WsabiLSVGP


# Set up data
X = np.random.standard_normal((10, 2))
Xnew = 2 * np.random.standard_normal((50, 2))
# Add bias dimension
bias = 3
X = np.concatenate((X, bias * np.ones((10, 1))), axis=1)
Xnew = np.concatenate((Xnew, bias * np.ones((50, 1))), axis=1)
# Map to hypersphere
X = X / np.linalg.norm(X, axis=1, keepdims=True)
Xnew = Xnew / np.linalg.norm(Xnew, axis=1, keepdims=True)


def f(x):
    return np.exp(2 * np.pi * np.linalg.norm(x, axis=1) / 4 * 2)


F = f(X)
Fnew = f(Xnew)
noise_scale = 0.1
np.random.seed(1)
Y = F + np.random.randn(*F.shape) * noise_scale

data = (X, Y.reshape(-1, 1))


# Set up model
degrees = 3
truncation_level = np.sum([num_harmonics(3, d) for d in range(degrees)])
m = WsabiLSVGP(
    kernel=SphericalMatern(nu=0.5, truncation_level=truncation_level, dimension=3),
    likelihood=gpflow.likelihoods.Gaussian(variance=noise_scale ** 2),
    inducing_variable=SphericalHarmonicFeatures(dimension=3, degrees=degrees),
    num_data=len(X),
    whiten=False
)
gpflow.set_trainable(m.kernel, False)
gpflow.set_trainable(m.likelihood, False)
gpflow.set_trainable(m.inducing_variable, False)

opt = gpflow.optimizers.Scipy()
opt.minimize(
    m.training_loss_closure(data, compile=False), m.trainable_variables, compile=False
)

print(f'ELBO: {m.elbo(data).numpy().item()}')
