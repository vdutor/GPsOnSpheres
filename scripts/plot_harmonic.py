from gspheres.plotting import plotly_plot_spherical_function
from gspheres.spherical_harmonics import SphericalHarmonics


def main():
    phi = SphericalHarmonics(3, 20)

    def spherical_function_to_plot(x):
        return phi(x)[-1].numpy()

    fig = plotly_plot_spherical_function(
        spherical_function_to_plot, resolution=30, animate_steps=0, use_mesh=True
    )

    animation_options = dict(
        frame=dict(redraw=True, duration=50), transition=dict(easing="linear", duration=0)
    )

    config = dict(displayModeBar=False)

    fig.write_html(
        "spherical_harmonic.html",
        config=config,
        auto_play=True,
        animation_opts=animation_options,
        include_plotlyjs="cdn",
    )


if __name__ == "__main__":
    main()
