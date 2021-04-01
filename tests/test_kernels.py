import numpy as np
import pytest
import tensorflow as tf
from gspheres.kernels import ChordMatern, SphericalMatern
from gspheres.spherical_harmonics import SphericalHarmonics, num_harmonics
from gspheres.utils import chain, l2norm


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
@pytest.mark.parametrize("kernel_class", ["spherical", "chord"])
def test_kernels(nu, kernel_class):
    dimension = 3
    max_degree = 30

    X = np.random.randn(10, dimension)
    X = X / l2norm(X)

    if kernel_class == "spherical":
        kernel = SphericalMatern(nu, max_degree, dimension)
    elif kernel_class == "chord":
        kernel = ChordMatern(nu, dimension)
        kernel.variance.assign(1.42)

    eigenfeatures = SphericalHarmonics(dimension, max_degree)
    phi_X = eigenfeatures(X)  # [N, L]
    eigenvalues_per_level = kernel.eigenvalues(max_degree)
    num_harmonics_per_level = tf.convert_to_tensor(
        [num_harmonics(dimension, n) for n in range(max_degree)]
    )
    eigenvalues = chain(eigenvalues_per_level, num_harmonics_per_level)  # [L]
    value1 = tf.reduce_sum(
        eigenvalues[None, None, :] * phi_X[:, None, :] * phi_X[None, :, :], axis=2
    )  # [N, N]
    value2 = kernel.K(X)
    np.testing.assert_allclose(value1, value2, rtol=3e-2, atol=3e-2, equal_nan=False)


if __name__ == "__main__":
    test_kernels()
