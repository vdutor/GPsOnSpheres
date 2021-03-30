import numpy as np
import matplotlib.pyplot as plt

from gspheres.plotting import plotly_plot_spherical_function
from gspheres.spherical_harmonics import SphericalHarmonics


phi = SphericalHarmonics(3, 20)
f = lambda x: phi(x)[-1].numpy()
# f = lambda x: np.ones_like(x[:,0])
fig = plotly_plot_spherical_function(f, resolution=250)
# fig.show()
animation_options = dict(
    frame=dict(redraw=True, duration=50),
    transition=dict(easing="linear", duration=0)
)

config = dict(
    displayModeBar=False
)

fig.write_html(
    "spherical_harmonic.html",
    config=config,
    auto_play=True,
    animation_opts=animation_options,
    include_plotlyjs="cdn"
)
