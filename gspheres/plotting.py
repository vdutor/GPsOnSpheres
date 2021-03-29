import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def spherical_grid(resolution=100):
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.vstack([x.flatten(), y.flatten(), z.flatten()]).T


def plot_spherical_function(f, resolution=100, rescale_radius=False, ax=None):
    """
    f is a function which takes a N x 3 matrix of points on the sphere in
    cartesian coordinates, and returns a N, vector.
    Here we construc the cartesian coordinates in a big grid and then plot
    """

    grid = spherical_grid(resolution)
    fgrid = f(grid).reshape(resolution, resolution)

    # Set the aspect ratio to 1 so our sphere looks spherical
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    if rescale_radius:
        scale = np.abs(fgrid)
    else:
        scale = 1.0

    # scale the colors
    fmax, fmin = fgrid.max(), fgrid.min()
    fcolors = (fgrid - fmin) / (fmax - fmin)

    ax.plot_surface(
        grid[:, 0].reshape(resolution, resolution) * scale,
        grid[:, 1].reshape(resolution, resolution) * scale,
        grid[:, 2].reshape(resolution, resolution) * scale,
        rstride=1,
        cstride=1,
        facecolors=cm.viridis(fcolors),
    )

    # Turn off the axis planes
    ax.set_axis_off()
    return ax


import plotly.graph_objects as go


def plotly_plot_spherical_function(f, resolution=500, rescale_radius=False):
    """
    f is a function which takes a N x 3 matrix of points on the sphere in
    cartesian coordinates, and returns a N, vector.
    Here we construc the cartesian coordinates in a big grid and then plot
    """

    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    grid = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
    fgrid = f(grid).reshape(resolution, resolution)

    if rescale_radius:
        # scale = np.abs(fgrid)
        scale = fgrid
        x, y, z = [scale * 0.05 + t for t in [x, y, z]]
    else:
        scale = 1.0

    # scale the colors
    fmax, fmin = fgrid.max(), fgrid.min()
    fcolors = (fgrid - fmin) / (fmax - fmin)

    surf = go.Surface(x=x, y=y, z=z, surfacecolor=fcolors)  # , colorscale='Viridis')
    fig = go.Figure(surf)
    fig.update_layout(scene_aspectmode="cube")
    fig.update_traces(showscale=False)
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, visible=False),
            yaxis=dict(showbackground=False, showticklabels=False, visible=False),
            zaxis=dict(showbackground=False, showticklabels=False, visible=False),
        )
    )

    x_eye = -1.25
    y_eye = 2
    z_eye = 0.5

    fig.update_layout(
        width=600,
        height=600,
        scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])],
        # updatemenus=[
        #     dict(
        #         type="buttons",
        #         showactive=False,
        #         y=1,
        #         x=0.8,
        #         xanchor="left",
        #         yanchor="bottom",
        #         pad=dict(t=45, r=10),
        #         buttons=[
        #             dict(
        #                 label="Play",
        #                 method="animate",
        #                 args=[
        #                     None,
        #                     dict(
        #                         frame=dict(duration=5, redraw=True),
        #                         transition=dict(duration=0),
        #                         fromcurrent=True,
        #                         mode="immediate",
        #                     ),
        #                 ],
        #             )
        #         ],
        #     )
        # ],
    )

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    frames = []
    for t in np.arange(0, 6.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))

    fig.frames = frames

    return fig
