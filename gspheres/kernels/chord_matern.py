from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf
from scipy import integrate
from scipy.special import gegenbauer as scipy_gegenbauer

import gpflow
from gpflow.base import TensorType

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
        # self.base_kernel.lengthscales = 1.0

    def shape_function_cos_theta(self, t: TensorType) -> TensorType:
        r"""
        shape_function: [-1, 1] -> [-\infty, 1] with k(0) = 1
        """
        r2 = 2.0 * (1.0 - t) / tf.square(self.lengthscales)
        return self.base_kernel.K_r2(tf.cast(r2, tf.float64))

    def eigenvalues(self, max_degree: int) -> tf.Tensor:
        @tf.custom_gradient
        def eigenvalues_dlengthscale(lengthscale):
            """Enables automatic differentiation wrt lengthscale."""
            values = []
            derivatives = []
            for n in range(max_degree):
                v, dvdl = _funk_hecke(
                    self.shape_function_cos_theta,
                    n,
                    self.dimension,
                    lengthscale
                )
                values.append(v)
                derivatives.append(dvdl)
            def gradient(upstream_derivative):
                return upstream_derivative * tf.convert_to_tensor(dvdl)
            return tf.convert_to_tensor(values), gradient
        return eigenvalues_dlengthscale(self.lengthscales)


    @property
    def variance(self):
        return self.base_kernel.variance

    @property
    def lengthscales(self):
        return self.base_kernel.lengthscales

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        return self.base_kernel.K(X, X2)

    def K_diag(self, X: TensorType) -> tf.Tensor:
        """ Approximate the true kernel by an inner product between feature functions. """
        return self.base_kernel.K_diag(X)


def _funk_hecke(shape_function: Callable[[float], float], n: int, dim: int, lengthscale: tf.Variable) -> Tuple[float, float]:
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
    omega_d = surface_area_sphere(dim - 1) / surface_area_sphere(dim)
    alpha = (dim - 2.0) / 2.0
    C = scipy_gegenbauer(n, alpha)
    C_1 = C(1.0)

    def integrand(t: float) -> float:
        return shape_function(t) * C(t) * (1.0 - t ** 2) ** (alpha - 0.5)
    v = integrate.quad(integrand, -1.0, 1.0)[0]

    def dl_integrand(t: float) -> float:
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(lengthscale.variables)
            sf = shape_function(t)
        sf_dl = tape.gradient(sf, lengthscale.variables)
        return sf_dl * sf * C(t) * (1.0 - t ** 2) ** (alpha - 0.5)
    dv_dl = integrate.quad(dl_integrand, -1.0, 1.0)[0]

    return v * omega_d / C_1, dv_dl * omega_d / C_1
