import gspheres
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scipy
from gspheres.fundamental_set import num_harmonics
from gspheres.kernels.chord_matern import ChordMatern
from gspheres.kernels.spherical_matern import SphericalMatern
from gspheres.vish.covariances import SphericalHarmonicFeatures
from plotly.subplots import make_subplots
from tensorflow.python.ops.variables import trainable_variables

import gpflow


def get_data(num_data=100):
    noise_variance = 0.01
    X = np.random.randn(num_data, 2)
    kernel = gpflow.kernels.ArcCosine(order=1, bias_variance=1, weight_variances=1)
    Kxx = kernel(X).numpy() + noise_variance ** 0.5 * np.eye(num_data)
    Y = np.linalg.cholesky(Kxx) @ np.random.randn(num_data, 1)
    return X, Y


# def get_gpr(data):
#     kernel = gpflow.kernels.ArcCosine(order=1, bias_variance=1, weight_variances=1)
#     model = gpflow.models.gpr.GPR(data, kernel, noise_variance=0.01)
#     return model


def get_svgp(data):
    # degrees = 3
    # truncation_level = np.sum([num_harmonics(3, d) for d in range(degrees)])
    max_degree = 5
    kernel = ChordMatern(nu=0.5, dimension=3)
    _ = kernel.eigenvalues(max_degree)
    model = gpflow.models.SVGP(
        # kernel=SphericalMatern(nu=0.5, truncation_level=truncation_level, dimension=3),
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(variance=0.01),
        inducing_variable=SphericalHarmonicFeatures(dimension=3, degrees=max_degree),
        num_data=len(data[0]),
        whiten=False,
    )
    gpflow.utilities.set_trainable(model.likelihood, False)

    opt = gpflow.optimizers.Scipy()
    print(model.trainable_variables)
    opt.minimize(model.training_loss_closure(data), model.trainable_variables)
    return model


if __name__ == "__main__":
    X, Y = get_data()
    gpr_model = get_svgp((X, Y))

    N_test = 20
    X_test_1D = np.linspace(-3, 3, N_test)
    XX1, XX2 = np.meshgrid(X_test_1D, X_test_1D)
    X_test = np.vstack([XX1.ravel(), XX2.ravel()]).T  # [N**2, 2]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[
            [{"type": "surface"}, {"type": "surface"}],
        ],
        subplot_titles=["Data", "SVGP with Spherical Harmonic"],
    )

    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=Y[:, 0],
            mode="markers",
            marker=dict(
                size=12,
                color=Y[:, 0],
                colorscale="Viridis",
                opacity=0.8,
            ),
        ),
        row=1,
        col=1,
    )

    mean = gpr_model.predict_f(X_test)[0].numpy().reshape(N_test, N_test)
    var = gpr_model.predict_f(X_test)[1].numpy().reshape(N_test, N_test)
    up, lo = [mean + c * var ** 0.5 for c in [1, -1]]

    fig.add_trace(go.Surface(x=X_test_1D, y=X_test_1D, z=mean, colorscale="Viridis"), row=1, col=2)
    fig.add_trace(
        go.Surface(x=X_test_1D, y=X_test_1D, z=lo, colorscale="Viridis", opacity=0.25), row=1, col=2
    )
    fig.add_trace(
        go.Surface(x=X_test_1D, y=X_test_1D, z=up, colorscale="Viridis", opacity=0.25), row=1, col=2
    )

    fig.show()
