import numpy as np
import tensorflow as tf
import gpflow

from gspheres.kernels.spherical_matern import SphericalMatern
from gspheres.fundamental_set import num_harmonics
from gspheres.vish import (
    map_to_sphere, WsabiLSVGP, SphericalHarmonicFeatures, MixedFeatures
)


# Set up data
X = np.random.standard_normal((10, 2))
Xnew = 2 * np.random.standard_normal((50, 2))
# Add bias dimension
bias = 3
X = map_to_sphere(X, bias)
Xnew = map_to_sphere(Xnew, bias)


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
    # inducing_variable=SphericalHarmonicFeatures(dimension=3, degrees=degrees),
    inducing_variable=MixedFeatures(
        dimension=3, degrees=degrees, locations=tf.constant(X[-2:, :])
    ),
    num_data=len(X),
    whiten=False
)
gpflow.set_trainable(m.kernel, False)
gpflow.set_trainable(m.likelihood, False)
gpflow.set_trainable(m.inducing_variable, False)

opt = gpflow.optimizers.Scipy()
opt.minimize(
    m.training_loss_closure(data, compile=False),
    m.trainable_variables,
    compile=False,
    options=dict(maxiter=100)
)

print(f'ELBO: {m.elbo(data).numpy().item()}')
