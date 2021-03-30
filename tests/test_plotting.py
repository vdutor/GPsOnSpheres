import numpy as np
import pytest
from gspheres.plotting import plotly_plot_spherical_function
from gspheres.spherical_harmonics import SphericalHarmonics


@pytest.fixture
def spherical_function_to_plot():
    phi = SphericalHarmonics(3, 20)
    func = lambda x: phi(x)[-1].numpy()
    _ = func(np.random.randn(1, 3))
    return func


@pytest.mark.parametrize("use_mesh", [True, False])
@pytest.mark.parametrize("animate_steps", [0, 11])
def test_plotting(spherical_function_to_plot, use_mesh, animate_steps):
    fig = plotly_plot_spherical_function(
        spherical_function_to_plot, resolution=10, animate_steps=animate_steps, use_mesh=use_mesh
    )


# animation_options = dict(
#     frame=dict(redraw=True, duration=50),
#     transition=dict(easing="linear", duration=0)
# )

# config = dict(
#     displayModeBar=False
# )

# fig.write_html(
#     "spherical_harmonic.html",
#     config=config,
#     auto_play=True,
#     animation_opts=animation_options,
#     include_plotlyjs="cdn"
# )
