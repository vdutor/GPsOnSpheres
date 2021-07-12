from typing import Union
import numpy as np
import tensorflow as tf
from gpflow import default_float, default_jitter
from gpflow.utilities import to_default_float
from gpflow.utilities.utilities import parameter_dict
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.models import SGPR
from gpflow.models.model import MeanAndVariance
from gpflow.models.training_mixins import InputData


class VishGPR(SGPR):
    """VISH model for regression."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def map_to_sphere(
            self, X: Union[tf.Tensor, tf.Variable]
    ) -> Union[tf.Tensor, tf.Variable]:
        """Map data to the surface of a hypersphere.

        Append self.bias to locations so you have points in R^{d+1} and
        map to hypersphere S^d.

        Uses Tensorflow, so that it can be used for acquisition
        function optimisation.

        :param X: Points to map onto hypersphere [N, D].
        :return: Mapped points that lie on the hypersphere [N, D + 1].
        """
        Xb = tf.concat(
            [
                (self.kernel.weight_variances ** 0.5) * X,
                (self.kernel.bias_variance ** 0.5) * tf.ones_like(X[:, :1]),
            ],
            axis=1
        )
        return Xb / tf.norm(Xb, axis=1, keepdims=True)

    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.

        This is modified from the super class to work with Kuu of type
        tf.linalg.LinearOperator.
        """
        X_data, Y_data = self.data
        tf.ensure_shape(X_data, [None, self.kernel.dimension - 1])
        X_data = self.map_to_sphere(X_data)

        num_inducing = self.inducing_variable.num_inducing
        num_data = to_default_float(tf.shape(Y_data)[0])
        output_dim = to_default_float(tf.shape(Y_data)[1])

        err = Y_data - self.mean_function(X_data)
        Kdiag = self.kernel(X_data, full_cov=False)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        # Cholesky of diagonal matrix.
        L = tf.linalg.LinearOperatorDiag(kuu.diag_part() ** 0.5)
        sigma = tf.sqrt(self.likelihood.variance)

        # Compute intermediate matrices
        A = L.solve(kuf) / sigma
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma

        # compute log marginal bound
        bound = -0.5 * num_data * output_dim * np.log(2 * np.pi)
        bound += tf.negative(output_dim) * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        bound -= 0.5 * num_data * output_dim * tf.math.log(self.likelihood.variance)
        bound += -0.5 * tf.reduce_sum(tf.square(err)) / self.likelihood.variance
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * output_dim * tf.reduce_sum(Kdiag) / self.likelihood.variance
        bound += 0.5 * output_dim * tf.reduce_sum(tf.linalg.diag_part(AAT))

        return bound

    def upper_bound(self) -> tf.Tensor:
        raise NotImplementedError

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        X_data, Y_data = self.data
        tf.ensure_shape(X_data, [None, self.kernel.dimension - 1])
        tf.ensure_shape(Xnew, [None, self.kernel.dimension - 1])
        X_data = self.map_to_sphere(X_data)
        Xnew = self.map_to_sphere(Xnew)

        num_inducing = self.inducing_variable.num_inducing
        err = Y_data - self.mean_function(X_data)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        Kus = Kuf(self.inducing_variable, self.kernel, Xnew)
        sigma = tf.sqrt(self.likelihood.variance)
        # Cholesky of diagonal matrix.
        L = tf.linalg.LinearOperatorDiag(kuu.diag_part() ** 0.5)
        A = L.solve(kuf) / sigma
        B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma
        tmp1 = L.solve(Kus)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                self.kernel(Xnew)
                + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), 0)
                - tf.reduce_sum(tf.square(tmp1), 0)
            )
            var = tf.tile(var[:, None], [1, self.num_latent_gps])
        return mean + self.mean_function(Xnew), var
