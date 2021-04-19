import tensorflow as tf
import gpflow
from gpflow.inducing_variables import InducingVariables
from gpflow import covariances as cov
from gpflow import kullback_leiblers as kl
from gpflow.base import TensorLike
from gpflow.utilities import to_default_float

from gspheres.kernels.spherical_matern import SphericalMatern
from gspheres.spherical_harmonics import SphericalHarmonics


# Define InducingVariables
class SphericalHarmonicFeatures(InducingVariables):
    """Wraps SphericalHarmonics."""
    def __init__(self, dimension, degrees):
        self.spherical_harmonics = SphericalHarmonics(dimension, degrees)

    def __len__(self):
        """Number of inducing variables"""
        return len(self.spherical_harmonics)


@cov.Kuu.register(SphericalHarmonicFeatures, SphericalMatern)
def Kuu_sphericalmatern_sphericalharmonicfeatures(
        inducing_variable,
        kernel,
        jitter=None
):
    """Covariance matrix between spherical harmonic features."""
    eigenvalues = kernel.eigenvalues(len(inducing_variable))
    return tf.linalg.LinearOperatorDiag(1 / eigenvalues)


@cov.Kuf.register(SphericalHarmonicFeatures, SphericalMatern, TensorLike)
def Kuf_sphericalmatern_sphericalharmonicfeatures(
        inducing_variable,
        kernel,
        X
):
    """
    Covariance between spherical harmonic features and function values.

    """
    return tf.transpose(inducing_variable.spherical_harmonics(X))


@kl.prior_kl.register(SphericalHarmonicFeatures, SphericalMatern, TensorLike, TensorLike)
def prior_kl_vish(inducing_variable, kernel, q_mu, q_sqrt, whiten=False):
    if whiten:
        raise NotImplementedError
    K = cov.Kuu(inducing_variable, kernel)
    return gauss_kl_vish(q_mu, q_sqrt, K)


def gauss_kl_vish(q_mu, q_sqrt, K):
    """
    Compute the KL divergence from

          q(x) = N(q_mu, q_sqrt^2)
    to
          p(x) = N(0, K)

    q_mu is a vector [N, 1] that contains the mean.
    q_sqrt is a matrix that is the lower triangular square-root matrix of the covariance of q.

    K is a positive definite matrix: the covariance of p.
    NOTE: K is a LinearOperator that provides efficient methjods
        for solve(), log_abs_determinant(), and trace()
    """
    # KL(N₀ || N₁) = ½ [tr(Σ₁⁻¹ Σ₀) + (μ₁ - μ₀)ᵀ Σ₁⁻¹ (μ₁ - μ₀) - k + ln(det(Σ₁)/det(Σ₀))]
    # N₀ = q; μ₀ = q_mu, Σ₀ = q_sqrt q_sqrtᵀ
    # N₁ = p; μ₁ = 0, Σ₁ = K
    # KL(q || p) =
    #     ½ [tr(K⁻¹ q_sqrt q_sqrtᵀA + q_muᵀ K⁻¹ q_mu - k + logdet(K) - logdet(q_sqrt q_sqrtᵀ)]
    # k = number of dimensions, if q_sqrt is m x m this is m²
    Kinv_q_mu = K.solve(q_mu)

    mahalanobis_term = tf.squeeze(tf.matmul(q_mu, Kinv_q_mu, transpose_a=True))

    # GPflow: q_sqrt is num_latent_gps x N x N
    num_latent_gps = to_default_float(tf.shape(q_mu)[1])
    logdet_prior = num_latent_gps * K.log_abs_determinant()

    product_of_dimensions__int = tf.reduce_prod(
        tf.shape(q_sqrt)[:-1])  # dimensions are integers
    constant_term = to_default_float(product_of_dimensions__int)

    Lq = tf.linalg.band_part(q_sqrt, -1, 0)  # force lower triangle
    logdet_q = tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(Lq))))

    # S = tf.matmul(q_sqrt, q_sqrt, transpose_b=True)
    # trace_term = tf.trace(K.solve(S))
    trace_term = tf.squeeze(
        tf.reduce_sum(Lq * K.solve(Lq), axis=[-1, -2])
    )  # [O(N²) instead of O(N³)

    twoKL = trace_term + mahalanobis_term - constant_term + logdet_prior - logdet_q
    return 0.5 * twoKL


@gpflow.conditionals.conditional.register(
    TensorLike, SphericalHarmonicFeatures, gpflow.kernels.Kernel, TensorLike
)
def conditional_vish(
    Xnew,
    inducing_variable,
    kernel,
    f,
    *,
    full_cov=False,
    full_output_cov=False,
    q_sqrt=None,
    white=False,
):
    """
     - Xnew are the points of the data or minibatch, size N x D (tf.array, 2d)
     - feat is an instance of features.InducingFeature that provides `Kuu` and `Kuf` methods
       for Fourier features, this contains the limits of the bounding box and the frequencies
     - f is the value (or mean value) of the features (i.e. the weights)
     - q_sqrt (default None) is the Cholesky factor of the uncertainty about f
       (to be propagated through the conditional as per the GPflow inducing-point implementation)
     - white (defaults False) specifies whether the whitening has been applied

    Given the GP represented by the inducing points specified in `feat`, produce the mean and
    (co-)variance of the GP at the points Xnew.

       Xnew :: N x D
       Kuu :: M x M
       Kuf :: M x N
       f :: M x K, K = 1
       q_sqrt :: K x M x M, with K = 1
    """
    if full_output_cov:
        raise NotImplementedError

    # num_data = tf.shape(Xnew)[0]  # M
    num_func = tf.shape(f)[1]  # K

    Kuu = cov.Kuu(inducing_variable, kernel)  # this is now a LinearOperator
    Kuf = cov.Kuf(inducing_variable, kernel, Xnew)  # still a Tensor

    KuuInv_Kuf = Kuu.solve(Kuf)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = kernel(Xnew) - tf.matmul(Kuf, KuuInv_Kuf, transpose_a=True)
        shape = (num_func, 1, 1)
    else:
        KufT_KuuInv_Kuf_diag = tf.reduce_sum(Kuf * KuuInv_Kuf, axis=-2)
        fvar = kernel(Xnew, full_cov=False) - KufT_KuuInv_Kuf_diag
        shape = (num_func, 1)
    fvar = tf.expand_dims(fvar, 0) * tf.ones(
        shape, dtype=gpflow.default_float()
    )  # K x N x N or K x N

    # another backsubstitution in the unwhitened case
    if white:
        raise NotImplementedError

    A = KuuInv_Kuf

    # construct the conditional mean
    fmean = tf.matmul(A, f, transpose_a=True)

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            # LTA = A * tf.expand_dims(q_sqrt, 2)  # K x M x N
            # won't work  # make ticket for this?
            raise NotImplementedError
        elif q_sqrt.get_shape().ndims == 3:
            # L = tf.matrix_band_part(tf.transpose(q_sqrt, (2, 0, 1)), -1, 0)  # K x M x M

            # K x M x N
            # A_tiled = tf.expand_dims(A.get(), 0) * tf.ones((num_func, 1, 1), dtype=float_type)

            # LTA = tf.matmul(L, A_tiled, transpose_a=True)  # K x M x N
            # TODO the following won't work for K > 1
            assert q_sqrt.shape[0] == 1
            # LTA = (A.T @ DenseMatrix(q_sqrt[:,:,0])).T.get()[None, :, :]
            ATL = tf.matmul(A, q_sqrt, transpose_a=True)
        else:
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.get_shape().ndims))
        if full_cov:
            # fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # K x N x N
            fvar = fvar + tf.matmul(ATL, ATL, transpose_b=True)  # K x N x N
        else:
            # fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # K x N
            fvar = fvar + tf.reduce_sum(tf.square(ATL), 2)  # K x N
    fvar = tf.transpose(fvar)  # N x K or N x N x K

    return fmean, fvar