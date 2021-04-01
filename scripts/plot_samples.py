import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from gspheres.fundamental_set import num_harmonics
from gspheres.kernels import ChordMatern, SphericalMatern
from gspheres.plotting import get_mesh_on_sphere
from gspheres.spherical_harmonics import SphericalHarmonics
from gspheres.utils import chain
from plotly.subplots import make_subplots

import gpflow


def f_sample(*, nu=0.5, kernel_class="chord", eps=None):
    dimension = 3
    max_degree = 20  # = L

    if kernel_class == "spherical":
        kernel = SphericalMatern(nu, max_degree, dimension)
    elif kernel_class == "chord":
        kernel = ChordMatern(nu, dimension)

    eigenfeatures = SphericalHarmonics(dimension, max_degree)
    eigenvalues_per_level = kernel.eigenvalues(max_degree)
    num_harmonics_per_level = tf.convert_to_tensor(
        [num_harmonics(dimension, n) for n in range(max_degree)]
    )
    eigenvalues = chain(eigenvalues_per_level, num_harmonics_per_level)

    if eps is None:
        n = tf.shape(eigenvalues)[0]
        eps = tf.random.normal((n,), dtype=gpflow.default_float())

    def func(X):
        weights = eps * eigenvalues ** 0.5  # [L]
        phi_X = eigenfeatures(X)  # [N, L]
        return tf.reduce_sum(weights[None, :] * phi_X, axis=1).numpy()

    return func


EPS = tf.random.normal((400,), dtype=gpflow.default_float(), seed=0)
GRID, CELLS = get_mesh_on_sphere(10)
FUNCTIONS = [
    f_sample(nu=0.5, eps=EPS, kernel_class="spherical"),
    f_sample(nu=1.5, eps=EPS, kernel_class="spherical"),
    f_sample(nu=2.5, eps=EPS, kernel_class="spherical"),
    f_sample(nu=2.5, eps=EPS, kernel_class="chord"),
    f_sample(nu=2.5, eps=EPS, kernel_class="chord"),
    f_sample(nu=2.5, eps=EPS, kernel_class="chord"),
]

CMIN = np.min(np.concatenate([f(GRID)[None] for f in FUNCTIONS], axis=0))
CMAX = np.max(np.concatenate([f(GRID)[None] for f in FUNCTIONS], axis=0))
print(CMIN, CMAX)

fig = make_subplots(
    rows=2,
    cols=3,
    specs=[
        [{"type": "surface"}, {"type": "surface"}, {"type": "surface"}],
        [{"type": "surface"}, {"type": "surface"}, {"type": "surface"}],
    ],
    subplot_titles=[
        r"$\text{Spherical}\ \nu = 1/2$",
        r"$\text{Spherical}\ \nu = 3/2$",
        r"$\text{Spherical}\ \nu = 5/2$",
        r"$\text{Chord}\ \nu = 1/2$",
        r"$\text{Chord}\ \nu = 3/2$",
        r"$\text{Chord}\ \nu = 5/2$",
    ],
)

i = 0
for r in range(2):
    for c in range(3):
        fgrid = FUNCTIONS[i](GRID)
        p = go.Mesh3d(
            x=GRID[:, 0],
            y=GRID[:, 1],
            z=GRID[:, 2],
            i=CELLS[:, 0],
            j=CELLS[:, 1],
            k=CELLS[:, 2],
            colorbar_title="z",
            intensity=fgrid,
            showscale=(i == 0),
            cmax=CMAX,
            cmin=CMIN,
        )
        fig.add_trace(p, row=r + 1, col=c + 1)
        i += 1

fig.show()
