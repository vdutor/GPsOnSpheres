from typing import Union
import numpy as np
import tensorflow as tf
from gpflow.models import SVGP
from gpflow.models.model import MeanAndVariance
from gpflow.models.training_mixins import InputData

from .eigenfunction_variables import SphericalHarmonicFeatures
from .mixed_variables import MixedFeatures


def map_to_sphere(X: np.ndarray, bias: Union[int, float]) -> np.ndarray:
    """
    Append bias to X and map to surface of the unit hypersphere.

    X is [N, D]
    """
    Xb = np.concatenate((X, bias * np.ones((*X.shape[:-1], 1))), axis=1)
    return Xb / np.linalg.norm(Xb, axis=1, keepdims=True)


class WsabiLSVGP(SVGP):
    """SVGP that should be fit to the log transform of the training
    targets. The unwarping is done by linearisation.
    """
    def __init__(self, *args, alpha=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def predict_unwarped_f(
            self, Xnew: InputData, full_cov=False, full_output_cov=False
    ) -> MeanAndVariance:
        mean, var = self.predict_f(Xnew, full_cov, full_output_cov)

        f_sq_mean = self.alpha + mean ** 2 / 2
        if full_cov:
            covar_factors = tf.linalg.matmul(mean, mean, transpose_a=True)
            f_sq_var = covar_factors * var
        else:
            f_sq_var = var * mean ** 2

        return f_sq_mean, f_sq_var


class LogMSVGP(SVGP):
    """SVGP that should be fit to the log transform of the training
    targets. The unwarping is done by moment matching.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict_unwarped_f(
            self, Xnew: InputData, full_cov=False, full_output_cov=False
    ) -> MeanAndVariance:
        mean, var = self.predict_f(Xnew, full_cov, full_output_cov)

        f_exp_mean = tf.exp(mean + 0.5 * var)
        exp_var = tf.exp(var) - 1
        if full_cov:
            covar_factors = tf.linalg.matmul(mean, mean, transpose_a=True)
            f_exp_var = covar_factors * exp_var
        else:
            f_exp_var = exp_var * mean ** 2

        return f_exp_mean, f_exp_var
