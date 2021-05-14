from typing import Optional, Union

import numpy as np
import tensorflow as tf
from scipy.special import gamma

import gpflow
from gpflow.base import TensorType

from ..fundamental_set import num_harmonics
from ..gegenbauer_polynomial import Gegenbauer


class SphericalMatern(gpflow.kernels.Kernel):
    def __init__(self, nu: float, truncation_level: int, dimension: int):
        """
        :param dimension: S^{d-1}, R^d with d = dimension
        """
        assert nu in [1 / 2, 3 / 2, 5 / 2]
        assert dimension == 3

        super().__init__()

        self.normalisation_constant = 40.0
        self.dimension = dimension
        self.alpha = (dimension - 2) / 2
        self.nu = nu
        self.truncation_level = truncation_level  # = L
        self.Cs = [Gegenbauer(n, self.alpha) for n in range(self.truncation_level)]
        self.lengthscales = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())

        _eigenvals = self.eigenvalues(self.truncation_level)
        self.constants = tf.convert_to_tensor(
            [
                eigenvalue * (n + self.alpha) / self.alpha
                for eigenvalue, n in zip(_eigenvals, range(self.truncation_level))
            ]
        )  # [L]
        self.diag = tf.reduce_sum(
            [
                num_harmonics(self.dimension, n) * eigenvalue
                for eigenvalue, n in zip(_eigenvals, range(self.truncation_level))
            ]
        )  # scalar
        self.variance = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())
        self._training = True

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, flag: bool):
        self._training = flag

    def eigenvalues(self, max_degree: int):
        ns = tf.convert_to_tensor(np.arange(max_degree), dtype=gpflow.default_float())
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
