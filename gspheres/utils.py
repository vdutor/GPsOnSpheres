from typing import Any, List

import itertools
import numpy as np
import tensorflow as tf
from scipy.special import gamma

from gpflow.base import TensorType


def surface_area_sphere(d: int) -> float:
    r"""
    Surface area of sphere in R^d, S^{d-1}.

    \Omega_{d-1} following notation of [1].

    :param d: dimension
        S^{d-1} = { x ∈ R^d and ||x||_2 = 1 }
        For a ball d=3

    [1] Sparse Gaussian Processes with Spherical Harmonic Features,
        Vincent Dutordoir, Nicolas Durrande and James Hensman.
        Appendix, equation 2.
    """
    return 2 * (np.pi ** (d / 2)) / gamma(d / 2)


def spherical_to_cartesian_4d(thetas, r=1.0):
    """
    Converts points on the hypersphere S^3 in R^4 given in
    spherical coordinates to their cartesian value.

    :param thetas: [N, 3]
        thetas = [theta_1 | theta_2 | theta_3], with theta_i [N, 1]
        0 <= theta_1 < 2 * pi and
        0 <= theta_2 < pi
        0 <= theta_3 < pi
    :return: points in cartesian coordinate system [N, 4],
        norm of the points equals `r`.
    """
    assert thetas.shape[-1] == 3

    theta1, theta2, theta3 = [thetas[:, [i]] for i in range(3)]
    x1 = r * np.sin(theta3) * np.sin(theta2) * np.sin(theta1)
    x2 = r * np.sin(theta3) * np.sin(theta2) * np.cos(theta1)
    x3 = r * np.sin(theta3) * np.cos(theta2)
    x4 = r * np.cos(theta3)
    return np.c_[x1, x2, x3, x4]


def spherical_to_cartesian(thetas, r=1.0):
    """
    Converts points on the hypersphere S^3 in R^4 given in
    spherical coordinates to their cartesian value.

    :param thetas: [N, 2]
        thetas = [theta_1 | theta_2], with theta_i [N, 1]
        0 <= theta_1 < 2 * pi and
        0 <= theta_2 < pi
    :return: points in cartesian coordinate system [N, 3],
        norm of the points equals `r`.
    """
    assert thetas.shape[-1] == 2

    theta = thetas[:, [0]]
    phi = thetas[:, [1]]
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.c_[x, y, z]


def l2norm(X: TensorType) -> TensorType:
    """
    Returns the norm of the vectors in `X`. The vectors are
    D-dimensional and  stored in the last dimension of `X`.

    :param X: [..., D]
    :return: norm for each element in `X`, [N, 1]
    """
    return tf.reduce_sum(X ** 2, keepdims=True, axis=-1) ** 0.5
