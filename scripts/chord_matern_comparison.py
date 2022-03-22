from typing import Union
import numpy as np
import scipy
import tensorflow as tf
from tensorflow.python.ops.variables import trainable_variables
import gpflow
from gpflow.covariances import Kuu, Kuf
import gspheres
from gspheres.fundamental_set import num_harmonics
from gspheres.kernels.chord_matern import ChordMatern
from gspheres.kernels.spherical_matern import SphericalMatern
from gspheres.kernels.truncated_chord_matern import TruncatedChordMatern
from gspheres.vish import VishGPR
from gspheres.vish.covariances import SphericalHarmonicFeatures
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def get_data(num_data=100):
    noise_variance = 0.01
    X = np.random.randn(num_data, 2)
    kernel = gpflow.kernels.ArcCosine(order=1, bias_variance=1, weight_variances=[1.0, 1.0])
    Kxx = kernel(X).numpy() + noise_variance ** 0.5 * np.eye(num_data)
    Y = np.linalg.cholesky(Kxx) @ np.random.randn(num_data, 1)
    return X, Y


def get_sgpr(data, kernel_type='chord_matern', max_degree=6, noise_variance=0.01, optimise_noise_variance=True):
    if kernel_type == 'chord_matern':
        kernel = ChordMatern(nu=0.5, dimension=3, bias_variance=1, weight_variances=[1.0, 1.0])
    elif kernel_type == 'truncated_chord_matern':
        kernel = TruncatedChordMatern(nu=0.5, dimension=3, degrees=max_degree, bias_variance=1, weight_variances=[1.0, 1.0])
    else:
        raise NotImplementedError
    _ = kernel.eigenvalues(max_degree)
    model = VishGPR(
        data=data,
        kernel=kernel,
        inducing_variable=SphericalHarmonicFeatures(dimension=3, degrees=max_degree),
        noise_variance=noise_variance,
    )
    gpflow.set_trainable(model.likelihood, optimise_noise_variance)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss_closure(), model.trainable_variables)
    return model


if __name__ == "__main__":
    max_degree = 6
    init_noise_variance = 0.0001
    optimise_noise_variance = False

    X, Y = get_data()
    full_model = get_sgpr((X, Y), kernel_type='chord_matern', max_degree=max_degree, noise_variance=init_noise_variance, optimise_noise_variance=optimise_noise_variance)
    trunc_model = get_sgpr((X, Y), kernel_type='truncated_chord_matern', max_degree=max_degree, noise_variance=init_noise_variance, optimise_noise_variance=optimise_noise_variance)

    full_variances = full_model.predict_f(tf.constant(X))[1].numpy().reshape(-1)
    trunc_variances = trunc_model.predict_f(tf.constant(X))[1].numpy().reshape(-1)

    fig = go.Figure(data=[
        go.Bar(name='Full', y=full_variances),
        go.Bar(name='Trunc', y=trunc_variances)
    ])
    if optimise_noise_variance:
        fig.add_hline(y=init_noise_variance, annotation_text='Init Noise Var')
        fig.add_hline(y=full_model.likelihood.variance.numpy(), annotation_text='Full Noise Var')
        fig.add_hline(y=trunc_model.likelihood.variance.numpy(), annotation_text='Trunc Noise Var')
    else:
        fig.add_hline(y=init_noise_variance, annotation_text='Noise Var')
    fig.show()
