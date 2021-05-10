import gpflow.kernels
import tensorflow as tf
from gpflow.inducing_variables import InducingVariables
from gpflow import covariances as cov
from gpflow.base import TensorLike

from gspheres.fundamental_set import num_harmonics
from gspheres.spherical_harmonics import SphericalHarmonics
from gspheres.utils import chain


class MixedFeatures(InducingVariables):
    """Wraps SphericalHarmonics."""
    def __init__(self, dimension, degrees, locations):
        self.dimension = dimension
        self.max_degree = degrees
        self.locations = locations
        self.spherical_harmonics = SphericalHarmonics(dimension, degrees)


    def __len__(self):
        """Number of inducing variables"""
        return len(self.spherical_harmonics) + len(self.locations)


@cov.Kuu.register(MixedFeatures, gpflow.kernels.Kernel)
def Kuu_sphericalmatern_mixedfeatures(
        inducing_variable,
        kernel,
        jitter=None
):
    """Covariance matrix between spherical harmonic features."""
    eigenvalues_per_level = kernel.eigenvalues(inducing_variable.max_degree)
    num_harmonics_per_level = tf.convert_to_tensor([
        num_harmonics(inducing_variable.dimension, n)
        for n in range(inducing_variable.max_degree)
    ])
    eigenvalues = chain(eigenvalues_per_level, num_harmonics_per_level)
    # TODO: Implement so that schur complement can be used for inversion.
    # eigenfunction_block = tf.linalg.LinearOperatorDiag(1 / eigenvalues)
    eigenfunction_block = tf.linalg.diag(1 / eigenvalues)
    pseudo_input_block = kernel(inducing_variable.locations)
    # Cross covariance block
    ef_pi_block = tf.transpose(
        inducing_variable.spherical_harmonics(inducing_variable.locations)
    )
    top = tf.concat([eigenfunction_block, ef_pi_block], axis=1)
    bottom = tf.concat([tf.transpose(ef_pi_block), pseudo_input_block], axis=1)
    return tf.concat([top, bottom], axis=0)


@cov.Kuf.register(MixedFeatures, gpflow.kernels.Kernel, TensorLike)
def Kuf_sphericalmatern_mixedfeatures(
        inducing_variable,
        kernel,
        X
):
    """
    Covariance between spherical harmonic features and function values.

    """
    eigenfunction_block = tf.transpose(
        inducing_variable.spherical_harmonics(X)
    )
    pseudo_input_block = kernel(inducing_variable.locations, X)
    return tf.concat([eigenfunction_block, pseudo_input_block], axis=0)
