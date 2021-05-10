import numpy as np
import tensorflow as tf
import gpflow
from gpflow import default_jitter
from gpflow.models import SVGP, SGPR
from gpflow.covariances.dispatch import Kuf, Kuu

from gspheres.kernels.spherical_matern import SphericalMatern
from gspheres.kernels.chord_matern import ChordMatern
from gspheres.fundamental_set import num_harmonics
from gspheres.vish import (
    map_to_sphere, WsabiLSVGP, SphericalHarmonicFeatures, MixedFeatures
)

# 'SVGP', 'SGPR'
MODEL = 'SGPR'

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
    # kernel=SphericalMatern(nu=0.5, truncation_level=truncation_level, dimension=3),
    kernel=ChordMatern(nu=0.5, dimension=3),
    likelihood=gpflow.likelihoods.Gaussian(variance=noise_scale ** 2),
    inducing_variable=SphericalHarmonicFeatures(dimension=3, degrees=degrees),
    # inducing_variable=MixedFeatures(
    #     dimension=3, degrees=degrees, locations=tf.constant(X[-2:, :])
    # ),
    num_data=len(X),
    whiten=False
)
# Set q_mu and q_sqrt if using SGPR.
if MODEL == 'SGPR':
    # Slight modification of gpflow.models.SGPR.compute_qu
    X_data, Y_data = data

    kuf = Kuf(m.inducing_variable, m.kernel, X_data)
    kuu = Kuu(m.inducing_variable, m.kernel, jitter=default_jitter()).to_dense()

    sig = kuu + (m.likelihood.variance ** -1) * tf.matmul(kuf, kuf, transpose_b=True)
    sig_sqrt = tf.linalg.cholesky(sig)

    sig_sqrt_kuu = tf.linalg.triangular_solve(sig_sqrt, kuu)

    cov = tf.linalg.matmul(sig_sqrt_kuu, sig_sqrt_kuu, transpose_a=True)
    q_sqrt = tf.linalg.cholesky(cov)

    err = Y_data - m.mean_function(X_data)
    mu = (
            tf.linalg.matmul(
                sig_sqrt_kuu,
                tf.linalg.triangular_solve(sig_sqrt, tf.linalg.matmul(kuf, err)),
                transpose_a=True,
            )
            / m.likelihood.variance
    )

    m.q_mu.assign(mu)
    m.q_sqrt.assign(tf.reshape(q_sqrt, (1, *q_sqrt.shape)))

# Set up optimiser and variables to optimise.
opt = gpflow.optimizers.Scipy()
if 'SVGP' in MODEL:
    # gpflow.set_trainable(m.kernel, False)
    gpflow.set_trainable(m.kernel.lengthscales, False)
    # gpflow.set_trainable(m.likelihood, False)
    gpflow.set_trainable(m.inducing_variable, False)
else:
    # gpflow.set_trainable(m.kernel, False)
    gpflow.set_trainable(m.kernel.lengthscales, False)
    # gpflow.set_trainable(m.likelihood, False)
    gpflow.set_trainable(m.inducing_variable, False)
    gpflow.set_trainable(m.q_mu, False)
    gpflow.set_trainable(m.q_sqrt, False)
# Optimise.
opt.minimize(
    m.training_loss_closure(data, compile=False),
    m.trainable_variables,
    compile=False,
    options=dict(maxiter=2)
)

print(f'ELBO: {m.elbo(data).numpy().item()}')
