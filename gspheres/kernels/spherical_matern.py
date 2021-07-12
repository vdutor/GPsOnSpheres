from typing import Optional, Union

import math
import numpy as np
import tensorflow as tf
from scipy.special import gamma

import gpflow
from gpflow.base import TensorType
from gpflow import covariances as cov

from gspheres.spherical_harmonics import SphericalHarmonics
from ..fundamental_set import num_harmonics
from ..gegenbauer_polynomial import Gegenbauer


class SphericalMatern(gpflow.kernels.Kernel):

    def __init__(
            self,
            nu: float,
            degrees: int,
            dimension: int,
            weight_variances: Union[float, np.ndarray] = 1.0,
            bias_variance: float = 1.0,
    ):
        """
        :param degrees: Max degree for spherical harmonics.
        :param dimension: S^{d-1}, R^d with d = dimension
        """
        assert nu in [1 / 2, 3 / 2, 5 / 2]
        assert dimension == 3

        super().__init__()

        self.normalisation_constant = 40.0
        self.dimension = dimension
        self.alpha = (dimension - 2) / 2
        self.nu = nu

        # Truncation level is number of spherical harmonics.
        self.degrees = degrees
        self.truncation_level = np.sum(
            [num_harmonics(3, d) for d in range(degrees)]
        ).item()
        self.Cs = [Gegenbauer(n, self.alpha) for n in range(self.degrees)]

        # self.lengthscales = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())

        _eigenvals = self.eigenvalues(self.degrees)
        self.constants = tf.convert_to_tensor(
            [
                eigenvalue * (n + self.alpha) / self.alpha
                for eigenvalue, n in zip(_eigenvals, range(self.degrees))
            ]
        )  # [L]
        self.diag = tf.reduce_sum(
            [
                num_harmonics(self.dimension, n) * eigenvalue
                for eigenvalue, n in zip(_eigenvals, range(self.degrees))
            ]
        )  # scalar
        self.variance = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())
        self.bias_variance = gpflow.Parameter(bias_variance, transform=gpflow.utilities.positive())
        self.weight_variances = gpflow.Parameter(weight_variances,
                                                 transform=gpflow.utilities.positive())

    def eigenvalues(self, max_degree: int):
        ns = tf.convert_to_tensor(np.arange(max_degree), dtype=gpflow.default_float())
        # Eigenvalues of the Laplace-Beltrami operator.
        eigenvalues_harmonics = tf.convert_to_tensor(ns * (ns + self.dimension - 2))  # [L]
        return (
            self.spectral_density(eigenvalues_harmonics ** 0.5) / self.normalisation_constant
        )  # [L]

    def evaluate_gegenbauers(self, X: TensorType):
        """ X: [...], return [L, ...] """
        values = map(lambda C: C(X)[None], self.Cs)  # List of length L with tensors of shape [...]
        return tf.concat(list(values), axis=0)  # [L, ...]

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        """ Approximate the true kernel by an inner product between feature functions. """
        if X2 is None:
            X2 = X
        inner_product = tf.matmul(X, X2, transpose_b=True)  # [N1, N2]
        C = self.evaluate_gegenbauers(inner_product)  # [L, N1, N2]
        value = tf.reduce_sum(self.constants[:, None, None] * C, axis=0)  # [N1, N2]
        return self.variance * value

    def K_diag(self, X: TensorType) -> tf.Tensor:
        """ Approximate the true kernel by an inner product between feature functions. """
        return self.variance * self.diag * tf.ones_like(X[:, 0])

    def spectral_density(self, s: TensorType) -> TensorType:
        """
        Evaluate the spectral density of the Matern kernel.
        Implementation follows [1].

        [1] GPML, Chapter 4 equation 4.15, Rasmussen and Williams, 2006

        :param s: frequency at which to evaluate the spectral density

        :return: Tensor [N, 1]
        """
        lengthscale = self.lengthscales
        D = self.dimension
        nu = self.nu

        def power(a, n):
            return a ** n

        if nu == np.inf:
            # Spectral density for SE kernel
            return (
                power(2.0 * np.pi, D / 2.0)
                * power(lengthscale, D)
                * tf.exp(-0.5 * power(s * lengthscale, 2.0))
            )
        elif nu > 0:
            # Spectral density for Matern-nu kernel
            # tmp = 2.0 * nu / power(lengthscale, 2.0) + power(2 * np.pi * s, 2.0)
            tmp = 2.0 * nu / power(lengthscale, 2.0) + power(s, 2.0)
            return (
                power(2.0, D)
                * power(np.pi, D / 2.0)
                * gamma(nu + D / 2.0)
                * power(2.0 * nu, nu)
                / gamma(nu)
                / power(lengthscale, 2.0 * nu)
                * power(tmp, -(nu + D / 2.0))
            )
        else:
            raise NotImplementedError
