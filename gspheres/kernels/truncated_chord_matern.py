from typing import Callable, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import integrate
from scipy.special import gegenbauer as scipy_gegenbauer

import gpflow
from gpflow.base import TensorType
from gpflow import default_float

from gspheres.spherical_harmonics import SphericalHarmonics
from ..fundamental_set import num_harmonics
from ..gegenbauer_polynomial import Gegenbauer
from ..utils import surface_area_sphere


class TruncatedChordMatern(gpflow.kernels.Kernel):
    def __init__(
        self,
        nu: float,
        dimension: int,
        degrees: int = 12,
        variance: float = 1.0,
        variance_constraint: tfp.bijectors.Bijector = gpflow.utilities.positive(),
        weight_variances: Union[float, np.ndarray] = 1.0,
        weight_variances_constraint: tfp.bijectors.Bijector = gpflow.utilities.positive(),
        bias_variance: float = 1.0,
        bias_variance_constraint: tfp.bijectors.Bijector = gpflow.utilities.positive(),
        *,
        name: Optional[str] = None,
    ):
        super().__init__(active_dims=None, name=name)

        if nu == 1 / 2:
            self.base_kernel = gpflow.kernels.Matern12()
        elif nu == 3 / 2:
            self.base_kernel = gpflow.kernels.Matern32()
        elif nu == 5 / 2:
            self.base_kernel = gpflow.kernels.Matern52()
        else:
            raise NotImplementedError("Unknown Matern kernel, use `nu` equal to 1/2, 3/2, 5/2.")

        self._eigenvalues = {}
        self.constants = {}
        self.diag = {}
        self.dimension = dimension
        self.alpha = (dimension - 2) / 2

        self.variance = gpflow.Parameter(
            variance, transform=variance_constraint
        )
        self.bias_variance = gpflow.Parameter(
            bias_variance, transform=bias_variance_constraint
        )
        self.weight_variances = gpflow.Parameter(
            weight_variances, transform=weight_variances_constraint
        )
        # un-parameterise the kernel's lengthscale
        self.base_kernel.lengthscales = 1.0
        self.base_kernel.variance = tf.cast(1.0, gpflow.config.default_float())

        # Truncation level is number of spherical harmonics.
        self.degrees = degrees
        self.truncation_level = np.sum(
            [num_harmonics(3, d) for d in range(degrees)]
        ).item()
        self.Cs = [Gegenbauer(n, self.alpha) for n in range(self.degrees)]
        self.eigenvalues(self.degrees)

    def shape_function_cos_theta(self, t: TensorType) -> TensorType:
        r"""
        shape_function: [-1, 1] -> [-\infty, 1] with k(0) = 1
        """
        r2 = 2.0 * (1.0 - t)
        return self.base_kernel.K_r2(tf.cast(r2, tf.float64))

    def eigenvalues(self, max_degree: int) -> tf.Tensor:
        if max_degree not in self._eigenvalues:
            values = []
            for n in range(max_degree):
                v = _funk_hecke(self.shape_function_cos_theta, n, self.dimension)
                values.append(v)
            self._eigenvalues[max_degree] = tf.convert_to_tensor(values)
            self.constants[max_degree] = tf.convert_to_tensor(
                [
                    eigenvalue * (n + self.alpha) / self.alpha
                    for eigenvalue, n in zip(
                        self._eigenvalues[max_degree], range(self.degrees)
                    )
                ]
            )  # [L]
            self.diag[max_degree] = tf.reduce_sum(
                [
                    num_harmonics(self.dimension, n) * eigenvalue
                    for eigenvalue, n in zip(
                        self._eigenvalues[max_degree], range(self.degrees)
                    )
                ]
            )  # scalar
        return self.variance * self._eigenvalues[max_degree]
        # return self._eigenvalues[max_degree]

    def evaluate_gegenbauers(self, X: TensorType):
        """ X: [...], return [L, ...] """
        values = map(lambda C: C(X)[None], self.Cs)  # List of length L with tensors of shape [...]
        return tf.concat(list(values), axis=0)  # [L, ...]

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        X = tf.ensure_shape(X, [None, self.dimension])
        if X2 is not None:
            X2 = tf.ensure_shape(X2, [None, self.dimension])
        else:
            X2 = X
        inner_product = tf.matmul(X, X2, transpose_b=True)  # [N1, N2]
        C = self.evaluate_gegenbauers(inner_product)  # [L, N1, N2]
        value = tf.reduce_sum(self.constants[self.degrees][:, None, None] * C, axis=0)  # [N1, N2]
        return self.variance * value

    def K_diag(self, X: TensorType) -> tf.Tensor:
        """ Approximate the true kernel by an inner product between feature functions. """
        X = tf.ensure_shape(X, [None, self.dimension])
        return self.variance * self.diag[self.degrees] * tf.ones_like(X[:, 0])

    def __call__(self, X, X2=None, *, full_cov=True, presliced=False):
        if (not full_cov) and (X2 is not None):
            raise ValueError("Ambiguous inputs: `not full_cov` and `X2` are not compatible.")

        if not full_cov:
            assert X2 is None
            return self.K_diag(X)
        else:
            return self.K(X, X2)


def _funk_hecke(shape_function: Callable[[float], float], n: int, dim: int) -> float:
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
    return v * omega_d / C_1
