import pytest
import numpy as np
import tensorflow as tf
import gpflow

from gspheres.kernels import ChordMatern, TruncatedChordMatern, SphericalMatern
from gspheres.vish import VishGPR, SphericalHarmonicFeatures


def get_data(num_data=100):
    noise_variance = 0.01
    X = np.random.randn(num_data, 2)
    kernel = gpflow.kernels.ArcCosine(order=1, bias_variance=1, weight_variances=[1.0, 1.0])
    Kxx = kernel(X).numpy() + noise_variance ** 0.5 * np.eye(num_data)
    Y = np.linalg.cholesky(Kxx) @ np.random.randn(num_data, 1)
    return X, Y


def get_sgpr(data, kernel_type='chord_matern'):
    max_degree = 5
    if kernel_type == 'chord_matern':
        kernel = ChordMatern(nu=0.5, dimension=3, bias_variance=1, weight_variances=[1.0, 1.0])
    elif kernel_type == 'truncated_chord_matern':
        kernel = TruncatedChordMatern(nu=0.5, dimension=3, degrees=max_degree, bias_variance=1, weight_variances=[1.0, 1.0])
    elif kernel_type == 'spherical_matern':
        kernel = SphericalMatern(nu=0.5, dimension=3, degrees=max_degree, bias_variance=1, weight_variances=[1.0, 1.0])
    else:
        raise NotImplementedError
    _ = kernel.eigenvalues(max_degree)
    model = VishGPR(
        data=data,
        kernel=kernel,
        inducing_variable=SphericalHarmonicFeatures(dimension=3, degrees=max_degree),
        noise_variance=0.01,
    )
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss_closure(), model.trainable_variables)
    return model


@pytest.mark.parametrize('kernel_type', ('chord_matern', 'truncated_chord_matern', 'spherical_matern'))
def test_variance_at_data(kernel_type):
    """Check that posterior variance is below likelihood variance at
    data points.
    """
    X, Y = get_data()
    model = get_sgpr((X, Y), kernel_type)
    variances = model.predict_f(tf.constant(X))[1]
    noise_variance = model.likelihood.variance
    assert tf.math.reduce_all(variances < noise_variance).numpy()
