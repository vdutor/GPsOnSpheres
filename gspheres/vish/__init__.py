import math
from typing import Union, Generic, TypeVar
import numpy as np
import tensorflow as tf
from gpflow.models import SVGP, SGPR
from gpflow.models.model import MeanAndVariance
from gpflow.models.training_mixins import InputData

from gspheres.vish.eigenfunction_variables import SphericalHarmonicFeatures
from gspheres.vish.mixed_variables import MixedFeatures


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
            self,
            Xnew: InputData,
            full_cov=False,
            full_output_cov=False,
            log_offset=0
    ) -> MeanAndVariance:
        """Make unwarped prediction.

        log_offset is added to the warped mean before unwarping.
        """
        # Get warped prediction.
        mean, cov = self.predict_f(Xnew, full_cov, full_output_cov)
        shifted_mean = mean + log_offset
        var = tf.linalg.diag_part(cov) if full_cov else cov

        # Compute unwarping.
        f_exp_mean = tf.exp(shifted_mean + 0.5 * var)
        if full_cov:
            covar_factors = tf.linalg.matmul(
                f_exp_mean, f_exp_mean, transpose_a=True
            )
            f_exp_cov = covar_factors * (tf.exp(cov) - 1)
        else:
            f_exp_cov = (tf.exp(var) - 1) * f_exp_mean ** 2

        return f_exp_mean, f_exp_cov

    def predict_unwarped_log_variance(self, Xnew: InputData, log_offset=0):
        """Predictive variance of unwarped function.

        log_offset is added to the warped mean before unwarping.
        """
        # Get warped prediction.
        mean, var = self.predict_f(Xnew, full_cov=False, full_output_cov=False)
        shifted_mean = mean + log_offset
        # return log variance
        return (
            2 * shifted_mean + var + tf.math.log(tf.exp(var) - 1)
        )
