from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf
from scipy import integrate
from scipy.special import gegenbauer as scipy_gegenbauer

import gpflow
from gpflow.base import TensorType, Parameter

from ..utils import surface_area_sphere


class ChordMatern(gpflow.kernels.Kernel):
    def __init__(self, nu: float, dimension: int):
        # assert dimension == 3

        super().__init__()

        if nu == 1 / 2:
            self.base_kernel = gpflow.kernels.Matern12()
        elif nu == 3 / 2:
            self.base_kernel = gpflow.kernels.Matern32()
        elif nu == 5 / 2:
            self.base_kernel = gpflow.kernels.Matern52()

        self.dimension = dimension
        self._training = True
        self._eigenvalues = {}
        self._cache_parameters = {
            'variance': self.variance, 'lengthscales': self.lengthscales
        }

    @property
    def variance(self):
        return self.base_kernel.variance

    @property
    def lengthscales(self):
        return self.base_kernel.lengthscales

    def shape_function_cos_theta(
            self, t: TensorType, lengthscale: tf.Variable
    ) -> TensorType:
        r"""
        shape_function: [-1, 1] -> [-\infty, 1] with k(0) = 1
        """
        r2 = 2.0 * (1.0 - t) / tf.square(lengthscale)
        return self.base_kernel.K_r2(tf.cast(r2, tf.float64))

    def _compute_eigenvalues(self, max_degree: int) -> tf.Tensor:
        values = []
        for n in range(max_degree):
            v = _funk_hecke(
                self.shape_function_cos_theta,
                n,
                self.dimension,
                variance=self.variance,
                lengthscales=self.lengthscales
            )
            values.append(tf.reshape(v, shape=[-1]))
        return tf.concat(values, axis=0)

    def verify_eigenvalue_cache(self):
        """Check that lengthscale and variance have not been changed.

        If they have been changed, clear the cache.
        """
        variance = self.variance.numpy().item()
        lengthscale = self.lengthscales.numpy().item()
        if self._cache_parameters['variance'] != variance:
            for max_degree in self._eigenvalues:
                self._eigenvalues[max_degree] *= (
                        self.variance / self._cache_parameters['variance']
                )
        if self._cache_parameters['lengthscales'] != lengthscale:
            self._eigenvalues = {}
        self._cache_parameters['variance'] = variance
        self._cache_parameters['lengthscales'] = lengthscale

    def eigenvalues(self, max_degree: int) -> tf.Tensor:
        self.verify_eigenvalue_cache()
        if max_degree not in self._eigenvalues:
            self._eigenvalues[max_degree] = self._compute_eigenvalues(
                max_degree
            )
        return self._eigenvalues[max_degree]

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        # TODO: Refactor in terms of truncated Mercer decomposition.
        return self.base_kernel.K(X, X2)

    def K_diag(self, X: TensorType) -> tf.Tensor:
        """ Approximate the true kernel by an inner product between feature functions. """
        # TODO: Refactor in terms of truncated Mercer decomposition.
        return self.base_kernel.K_diag(X)


def _funk_hecke(
        shape_function: Callable[[float, tf.Variable], float],
        n: int,
        dim: int,
        variance: Parameter,
        lengthscales: Parameter
) -> Tuple[float, float]:
    r"""
    Implements Funk-Hecke [see 1] where we integrate over the sphere of dim-1
    living in \Re^dim.
    Using these coefficients we can approximate shape function s combined with
    the CollapsedEigenfunctions C as follows:
    For x, x' in \Re^dim
    s(x^T x') = \sum_n a_n C_n(x^T x')
    for which, when we uncollapse C_n, we get
    s(x^T x') = \sum_n \sum_k^N(dim, n) a_n \phi_{n, k}(x) \phi_{n, k}(x')

    [1] Variational Inducing Spherical Harmonics (appendix)

    :param shape_function: [-1, 1] -> \Re
    :param n: degree (level)
    :param dim: x, x' in \Re^dim
    """
    assert dim >= 3, "Sphere needs to be at least S^2."
    assert len(lengthscales.shape) == 0, "ChordMatern kernels can only have one lengthscale"
    omega_d = surface_area_sphere(dim - 1) / surface_area_sphere(dim)
    alpha = (dim - 2.0) / 2.0
    C = scipy_gegenbauer(n, alpha)
    C_1 = C(1.0)

    @tf.custom_gradient
    def integrate_tf(variance, lengthscales):
        lengthscale_variable = tf.constant(lengthscales.numpy())
        def integrand(t: float) -> float:
            return shape_function(t, lengthscale_variable) * C(t) * (1.0 - t ** 2) ** (alpha - 0.5)
        integral = integrate.quad(integrand, -1.0, 1.0)[0]
        def grad_fn(upstream, variables=None):
            def dl_integrand(t: float) -> float:
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(lengthscale_variable)
                    sf = shape_function(t, lengthscale_variable)
                sf_dl = tape.gradient(sf, lengthscale_variable)
                return sf_dl * C(t) * (1.0 - t ** 2) ** (alpha - 0.5)
            dint_dl = tf.convert_to_tensor(integrate.quad(dl_integrand, -1.0, 1.0)[0], dtype=tf.float64)
            dint_dv = tf.convert_to_tensor(integral, dtype=tf.float64) / variance
            return (
                (upstream * dint_dv, upstream * dint_dl),
                len(variables) * [None]
            )
        return tf.convert_to_tensor(integral, dtype=tf.float64), grad_fn

    integral = integrate_tf(variance, lengthscales)

    return integral * omega_d / C_1
