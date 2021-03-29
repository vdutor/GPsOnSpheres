import numpy as np
import matplotlib.pyplot as plt

from gspheres.plotting import plotly_plot_spherical_function
from gspheres.spherical_harmonics import SphericalHarmonics


phi = SphericalHarmonics(3, 20)
f = lambda x: phi(x)[-1].numpy()
# f = lambda x: np.ones_like(x[:,0])
fig = plotly_plot_spherical_function(f)
# fig.show()
fig.write_html("spherical_harmonic.html", auto_play=True)
