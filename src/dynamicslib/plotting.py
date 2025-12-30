from typing import List, Callable
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pandas as pd
from IPython.display import display, HTML
import plotly
from numba import njit
from scipy.signal import find_peaks
from base64 import b64encode
from dash import Dash, dcc, html, Input, Output, callback
from dynamicslib.consts import muEM, LU, TU
from dynamicslib.common import get_Lpts
from plotly.subplots import make_subplots
import os

plotly.offline.init_notebook_mode()
display(
    HTML(
        '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
    )
)


@njit(cache=True)
def get_rotm(theta: float, inc: float = 0) -> NDArray[np.float64]:
    """Support function for to_eci, gets rotation matrix for 13 rotation (to model an inclination + truan rotation)

    Args:
        theta (float): True anomaly (angle 2, about axis 3)
        inc (float, optional): Inclination (angle 1, about axis 1). Defaults to 0.

    Returns:
        NDArray[np.float64]: rotation matrix
    """
    r1 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(inc), -np.sin(inc)],
            [0.0, np.sin(inc), np.cos(inc)],
        ]
    )
    r3 = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return r1 @ r3


@njit(cache=True)
def to_eci(
    xs: NDArray[np.float64],
    ts: NDArray[np.float64],
    theta0: float = 0,
    inc_rad: float = 0,
    LU=LU,
    TU=TU,
    mu: float = muEM,
) -> NDArray[np.float64]:
    """Converts CR3BP coordinates to ECI coordinates

    Args:
        xs (NDArray[np.float64]): state vectors (N, 6)
        ts (NDArray[np.float64]): times (6,)
        t0 (float, optional): initial time. Defaults to 0.
        revs (int, optional): number or revs of the trajectory (if greater than 1, will repeat). Defaults to 1.
        inc_rad (float, optional): inclination. Can be ignored unless you are specifying a specific ECI frame. Defaults to 0.
        LU (_type_, optional): Length unit [km]. Defaults to package-defined LU.
        TU (_type_, optional): Time Unit [sec]. Defaults to package-defined TU.
        mu (float, optional): mass ratio. Defaults to muEM.

    Returns:
        NDArray[np.float64]: ECI outputs
    """
    # assert np.size(xs, 1) == 6  # make sure 6 cols
    omega = 1
    rot_angs = omega * ts + theta0
    pos_CR3BP = xs[:, :3]
    vel_CR3BP = xs[:, 3:]
    # Earth-Centered ND Rotating
    pos_ECNDR = pos_CR3BP + np.array([mu, 0, 0])
    vel_ECNDR = vel_CR3BP

    # Earth-Centered ND Inertial
    # rot_mtxs = np.empty((len(ts), 3, 3))
    vel_ECNDI = np.empty((len(ts), 3))
    pos_ECNDI = np.empty((len(ts), 3))
    for i, ang in enumerate(rot_angs):
        rotm = get_rotm(ang, inc_rad)
        # earth-centered nondim rotating plus omega x r
        vel_localframe = vel_ECNDR[i] + np.array([-pos_ECNDR[i, 1], pos_ECNDR[i, 0], 0])
        vel_ECNDI[i] = rotm @ vel_localframe
        pos_ECNDI[i] = rotm @ pos_ECNDR[i]

    pos_ECI = pos_ECNDI * LU
    vel_ECI = vel_ECNDI * LU / TU

    xs_ECI = np.hstack((pos_ECI, vel_ECI))
    return xs_ECI


@njit(cache=True)
def to_lci(
    xs: NDArray[np.float64],
    ts: NDArray[np.float64],
    theta0: float = 0,
    inc_rad: float = 0,
    LU=LU,
    TU=TU,
    mu: float = muEM,
):
    """Converts CR3BP coordinates to LCI coordinates

    Args:
        xs (NDArray[np.float64]): state vectors (N, 6)
        ts (NDArray[np.float64]): times (6,)
        t0 (float, optional): initial time. Defaults to 0.
        inc_rad (float, optional): inclination. Can be ignored unless you are specifying a specific LCI frame. Defaults to 0.
        LU (_type_, optional): Length unit [km]. Defaults to package-defined LU.
        TU (_type_, optional): Time Unit [sec]. Defaults to package-defined TU.
        mu (float, optional): mass ratio. Defaults to muEM.

    Returns:
        NDArray[np.float64]: ECI outputs
    """
    return to_eci(xs, ts, theta0, inc_rad, LU, TU, mu - 1)


# def matplotlib_family(
#     Xs_all: List,
#     xyzs_all: list,
#     names: list,
#     eig_vals_all: list,
#     X2xtf_func: Callable,
#     colormap: str = "rainbow",
#     mu: float = muEM,
# ):
#     # %config InlineBackend.print_figure_kwargs = {'bbox_inches':None}

#     assert len(Xs_all) == len(xyzs_all) == len(names) == len(eig_vals_all)
#     Lpoints = get_Lpts(mu)
#     num_fams = len(Xs_all)
#     cm = plt.get_cmap(colormap)

#     for ifam in range(num_fams):
#         fig = plt.figure(figsize=(14, 6))
#         ax = fig.add_subplot(121, projection="3d")
#         Xs = Xs_all[ifam]
#         eig_vals = eig_vals_all[ifam]
#         n = len(Xs)
#         # plot the 3D
#         xyzs_fam = xyzs_all[ifam]
#         for i, xyzs in enumerate(xyzs_fam):
#             ax.plot(*xyzs, "-", lw=1, color=cm(i / n))

#         # find the limits so we can plot and preserve them
#         ax.axis("equal")
#         lims = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
#         xl, yl, zl = (
#             1.2 * (lims - np.mean(lims, axis=1)[:, None])
#             + np.mean(lims, axis=1)[:, None]
#         )
#         ax.set(xlim=xl, ylim=yl, zlim=zl)

#         # plot the projections
#         for i, xyzs in enumerate(xyzs_fam):
#             x, y, z = xyzs
#             ax.plot(xl[0], y, z, "-", lw=0.3, color=cm(i / n), alpha=0.5)
#             ax.plot(x, yl[1], z, "-", lw=0.3, color=cm(i / n), alpha=0.5)
#             ax.plot(x, y, zl[0], "-", lw=0.3, color=cm(i / n), alpha=0.5)

#         # Lpoints and bodies
#         ax.scatter(Lpoints[0], Lpoints[1], c="c", s=6, alpha=1, axlim_clip=True)
#         ax.scatter([-mu, 1 - mu], [0, 0], c="w", s=25, alpha=1, axlim_clip=True)

#         for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
#             axis.pane.fill = False
#             axis._axinfo["grid"]["linewidth"] = 0.3
#             axis._axinfo["grid"]["color"] = "darkgrey"
#             axis.set_label_coords(0, 0)

#         ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$", title="Position Space")
#         ax.tick_params(axis="both", which="major", labelsize=7)
#         fig.suptitle(names[ifam])

#         # stability, period, energy
#         ax = fig.add_subplot(322)
#         ax2 = fig.add_subplot(324, sharex=ax)
#         ax3 = fig.add_subplot(326, sharex=ax)
#         x_axis = list(range(n))
#         periods = []
#         jacobi_consts = []
#         stabs1 = []
#         stabs2 = []
#         for i, X in enumerate(Xs):
#             jc, tf = get_JC_tf(X, X2xtf_func)
#             periods.append(tf)
#             jacobi_consts.append(jc)
#             eigvals = eig_vals[i]
#             eigvals = eigvals[np.argsort(np.abs(eigvals))]
#             stabs1.append(np.abs(eigvals[0] + 1 / eigvals[0]) / 2)
#             stabs2.append(np.abs(eigvals[1] + 1 / eigvals[1]) / 2)

#         ax.scatter(x_axis, jacobi_consts, c=x_axis, alpha=1, cmap=cm)
#         ax2.scatter(x_axis, periods, c=x_axis, alpha=1, cmap=cm)
#         ax3.scatter(x_axis, stabs1, c=x_axis, alpha=1, cmap=cm)
#         ax3.scatter(x_axis, stabs2, c=x_axis, alpha=1, cmap=cm)
#         ax.set(ylabel="Jacobi Constant", title="Family Evolution")
#         ax2.set(ylabel="Peroid")
#         ax3.set(xlabel="Index Along Family", ylabel="Stability Index")
#         ax3.set_yscale("log")
#         ax.grid(True, lw=0.25)
#         ax2.grid(True, lw=0.25)
#         ax3.grid(True, lw=0.25)
#         ax.set_xticklabels([])
#         ax2.set_xticklabels([])
#         fig.tight_layout(h_pad=0, w_pad=0)
#     plt.show()


# PLOTLY
def plotly_curve(x, y, z, name="", opacity=1.0, uid="", **kwargs):
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=kwargs,
        uid=uid,
        name=name,
        hoverlabel=dict(
            font=dict(size=9), namelength=-1, bgcolor="black", font_color="white"
        ),
        opacity=opacity,
    )


def plotly_curve_2d(x, y, name="", opacity=1.0, uid="", **kwargs):
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=kwargs,
        name=name,
        uid=uid,
        hoverlabel=dict(
            font=dict(size=9), namelength=-1, bgcolor="black", font_color="white"
        ),
        opacity=opacity,
    )


def make_label(data: List | NDArray, param_names: List[str]):
    terms = []
    for val, param in zip(data, param_names):
        valtxt = f"{val}" if param.lower() == "index" else f"{val:.5f}"
        terms.append(f"{param}: {valtxt}")
    return "<br>".join(terms)


def darken_color(color_str, scale=0.75):
    vals = [float(st) * scale for st in color_str.rstrip(")").lstrip("rgb(").split(",")]
    return f"rgb({vals[0]}, {vals[1]}, {vals[2]})"


def lighten_color(color_str, scale=0.75):
    vals = [
        (float(st) + scale * 255) / ((1 + scale))
        for st in color_str.rstrip(")").lstrip("rgb(").split(",")
    ]
    return f"rgb({vals[0]}, {vals[1]}, {vals[2]})"


def darken_colors(color_list, scale=0.75):
    out = tuple([darken_color(string, scale) for string in color_list])
    return out


def plotly_display(
    xyzs: List,
    full_dataframe: pd.DataFrame,
    colormap: str = "rainbow",
    mu: float = muEM,
    figsize: tuple[float, float] = (900, 600),
    port: int = 8050,
    flip_vert: bool = False,
    flip_horiz: bool = False,
    arrow_inds: List = [],
    n_arrow: int = 2,
    color_by: str = "Index",
):
    """Displays a family using a Plotly figure

    Args:
        xyzs (List): prepropagated trajectory ((3 x Ni)_i for i in num trajectories)
        full_dataframe (pd.DataFrame): Dataframe of values. Must include indices at least.
        colormap (str, optional): What color map to use. Defaults to "rainbow".
        mu (float, optional): Mu value. Used to show the Lagrange points and primaries. Defaults to muEM.
        figsize (tuple[float, float], optional): Figure size (in pix? I think?). Defaults to (900, 600).
        port (int, optional): What port number to use. Don't reuse the same port for multiple plots. Defaults to 8050.
        flip_vert (bool, optional): Flip horizontally. Defaults to False.
        flip_horiz (bool, optional): Flip vertically. Defaults to False.
        arrow_inds (list, optional): Indices (of the passed list) to add arrows to the curve
        n_arrow (int, optional): How many arrows per curve

    Returns:
        None
    """
    full_dataframe = full_dataframe.reset_index()

    # do the flips
    if flip_vert:
        full_dataframe["Initial z"] *= -1
        if "Initial vz" in full_dataframe.columns:
            full_dataframe["Initial vy"] *= -1
        xyzs_temp = xyzs.copy()
        for ii in range(len(xyzs_temp)):
            trj = xyzs_temp[ii]
            xyzs_temp[ii] = (trj[0], trj[1], -trj[2])
            if len(trj) > 3:
                xyzs_temp[ii] += (trj[3], trj[4], -trj[5])
        xyzs = xyzs_temp
    if flip_horiz:
        print(
            Warning(
                "Horizontal flip does not update vy properly; do not trust vy information"
            )
        )
        # if "Initial y" in full_dataframe.columns:
        #     full_dataframe["Initial vy"] *= -1
        xyzs_temp = xyzs.copy()
        for ii in range(len(xyzs_temp)):
            trj = xyzs_temp[ii]
            xyzs_temp[ii] = (trj[0], -trj[1], trj[2])
            if len(trj) > 3:
                xyzs_temp[ii] += (trj[3], -trj[4], trj[5])
        xyzs = xyzs_temp

    # build dataframe
    df = full_dataframe[[col for col in full_dataframe.columns if "Eig" not in col]]
    data = df.values.astype(np.float32)
    param_names = list(df.columns)

    # check whether it's planar (will use 2d plot if so)
    is2d = ("Initial z" not in param_names and "Initial vz" not in param_names) or max(
        [np.max(np.abs(xyz[-1])) for xyz in xyzs]
    ) < 1e-10

    datatr = data.T

    # make sure the lengths match
    assert len(xyzs) == len(data)

    # convert to float 32 to save storage space. Float 16 may even suffice
    xyzs = [np.float32(xyz) for xyz in xyzs]
    n = len(xyzs)  # number of trajectories

    cdata = datatr[param_names.index(color_by)]  # color data

    # get bounding box
    xs, ys, zs = np.hstack(xyzs)[:3]
    minx, miny, minz = (min(xs), min(ys), min(zs))
    maxx, maxy, maxz = (max(xs), max(ys), max(zs))
    rng = 1.25 * max([maxx - minx, maxy - miny, maxz - minz])

    ctrX = (maxx + minx) / 2
    ctrY = (maxy + miny) / 2
    ctrZ = (maxz + minz) / 2
    projX = ctrX - rng / 2
    projY = ctrY - rng / 2
    projZ = ctrZ - rng / 2

    # get curves and their projections

    projs = []
    curves = []

    colornums = cdata - min(cdata)
    colornums /= max(colornums)
    colors = px.colors.sample_colorscale(colormap, colornums)
    colors_dark = darken_colors(colors, 0.5)
    for i, xyzsi in enumerate(xyzs):
        x, y, z = xyzsi[:3]
        lbl = make_label(data[i], param_names)

        c = colors[i]
        cd = colors_dark[i]
        if not is2d:
            curves.append(plotly_curve(x, y, z, lbl, color=c, width=5, uid=f"traj{i}"))
            projs.append(
                plotly_curve(x * 0 + projX, y, z, color=cd, width=2, uid=f"proj{i}")
            )
            projs.append(
                plotly_curve(x, 0 * y + projY, z, color=cd, width=2, uid=f"proj{i}")
            )
            projs.append(
                plotly_curve(x, y, 0 * z + projZ, color=cd, width=2, uid=f"proj{i}")
            )
        else:
            curves.append(
                plotly_curve_2d(x, y, lbl, color=c, width=1.5, uid=f"traj{i}")
            )

    # make a colorbar with nan data
    if is2d:
        cbar_dummy = go.Scatter(
            x=[np.nan] * n,
            y=[np.nan] * n,
            mode="markers",
            name="dummy",
            marker=dict(
                color=cdata,
                colorscale=colormap,
                colorbar=dict(title="Index", thickness=12, exponentformat="power"),
            ),
        )
    else:
        cbar_dummy = go.Scatter3d(
            x=[np.nan] * n,
            y=[np.nan] * n,
            z=[np.nan] * n,
            mode="markers",
            name="dummy",
            marker=dict(
                color=cdata,
                colorscale=colormap,
                colorbar=dict(title="Index", thickness=12, exponentformat="power"),
            ),
        )

    fig = go.Figure(data=[cbar_dummy, *curves, *projs])

    # draw primaries and Lagrange points
    Lpoints = get_Lpts(mu=mu)
    if is2d:
        fig.add_trace(
            go.Scatter(
                x=Lpoints[0],
                y=Lpoints[1],
                text=[f"L{lp+1}" for lp in range(5)],
                hoverinfo="x+y+text",
                mode="markers",
                marker=dict(color="magenta", size=4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[-mu, 1 - mu],
                y=[0, 0],
                mode="markers",
                text=["P1", "P2"],
                hoverinfo="x+y+text",
                marker=dict(color="cyan"),
            )
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=Lpoints[0],
                y=Lpoints[1],
                z=0 * Lpoints[0],
                text=[f"L{lp+1}" for lp in range(5)],
                hoverinfo="x+y+text",
                mode="markers",
                marker=dict(color="magenta", size=4),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[-mu, 1 - mu],
                y=[0, 0],
                z=[0, 0],
                mode="markers",
                text=["P1", "P2"],
                hoverinfo="x+y+text",
                marker=dict(color="cyan"),
            )
        )

    if n_arrow > 0 and len(arrow_inds):
        for i_arrow in arrow_inds:  # for each arrows
            if i_arrow >= len(xyzs):
                continue

            cords = xyzs[i_arrow]
            N = cords.shape[1]  # num points
            # arclength for uniformly-ish spaced dots
            arc = np.append(0, np.cumsum(np.linalg.norm(np.diff(cords[:3], 1), axis=0)))
            arc /= np.max(arc)
            # print(arc)

            where = np.interp(np.arange(n_arrow) / n_arrow, arc, np.arange(N))

            # indices where to draw arrow/cone
            inds = np.int32(np.round(where))

            vels = (
                cords[3:, inds]
                if len(cords) > 3
                else cords[:, inds + 1] - cords[:, inds]
            )  # velocities

            # normalize
            vels = np.array([vel / np.linalg.norm(vel) for vel in vels.T]).T
            if not is2d:

                # extract components
                xc, yc, zc = cords[:3, inds]
                uc, vc, wc = vels
                # draw arrows
                arrows = go.Cone(
                    x=xc,
                    y=yc,
                    z=zc,
                    u=uc,
                    v=vc,
                    w=wc,
                    anchor="center",
                    sizemode="raw",
                    sizeref=rng / 20,
                    hoverinfo="none",
                    colorscale=["rgb(60, 15, 0)", "rgb(80, 20, 0)"],
                    # colorscale=[darken_color(colors[i_arrow], 0.5)] * 2,
                    showscale=False,
                )
            else:
                xc, yc = cords[:2, inds]
                uc, vc = vels[:2]
                angs = 90 - np.rad2deg(np.atan2(vc, uc))

                arrows = go.Scatter(
                    x=xc,
                    y=yc,
                    mode="markers",
                    hoverinfo="none",
                    marker=dict(
                        color=darken_color(colors[i_arrow], 1),
                        symbol="triangle-up",  # "arrow",
                        size=15,
                        angle=angs,
                        standoff=7.5,
                    ),
                )
            fig.add_trace(arrows)

    # set layout
    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=0, r=30, b=0, t=0),
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
    )
    # set axes
    if is2d:
        fig.update_layout(
            xaxis=dict(
                title="x [nd]",
                range=[ctrX - rng / 2, ctrX + rng / 2],
            ),
            yaxis=dict(
                title="y [nd]",
                range=[ctrY - rng / 2, ctrY + rng / 2],
            ),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1, exponentformat="power")
        fig.update_xaxes(exponentformat="power")
    else:
        fig.update_scenes(
            xaxis=dict(
                title="x [nd]",
                showbackground=False,
                showgrid=True,
                zeroline=False,
                range=[ctrX - rng / 2, ctrX + rng / 2],
                exponentformat="power",
            ),
            yaxis=dict(
                title="y [nd]",
                showbackground=False,
                showgrid=True,
                zeroline=False,
                range=[ctrY - rng / 2, ctrY + rng / 2],
                exponentformat="power",
            ),
            zaxis=dict(
                title="z [nd]",
                showbackground=False,
                showgrid=True,
                zeroline=False,
                range=[ctrZ - rng / 2, ctrZ + rng / 2],
                exponentformat="power",
            ),
            aspectmode="cube",
        )

    # INTERACTIVITY IN HTML

    # show/hide buttons. We only have projections in 3d, so only show these in 3d
    if not is2d:
        argshide = {"visible": [True, *[True] * n, *[False] * (3 * n), True, True]}
        argsshow = {"visible": [True, *[True] * (4 * n + 2)]}
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.1,
                    y=1,
                    xanchor="left",
                    yanchor="top",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Show<br>Projections",
                            method="restyle",
                            args=[argsshow],
                        ),
                        dict(
                            label="Hide<br>Projections",
                            method="restyle",
                            args=[argshide],
                        ),
                    ],
                ),
            ]
        )

    # Dash dropdowns

    app = Dash()
    app.layout = html.Div(
        [
            dcc.Graph(figure=fig, id="display"),
            dcc.Dropdown(
                ["Index", "Period", "Jacobi Constant", "Stability Index"],
                "Index",
                id="param-dropdown",
            ),
            html.Button("Download HTML", id="btn_download_html"),
            dcc.Download(id="download-html-file"),
        ]
    )

    @callback(
        Output("display", "figure", allow_duplicate=False),
        Input("param-dropdown", "value"),
        prevent_initial_call=True,
    )
    def update_colorby(value):
        updated_fig = go.Figure(fig)
        cdata = datatr[param_names.index(value)]  # color data
        colornums = cdata - min(cdata)
        colornums /= max(colornums)
        colors = px.colors.sample_colorscale(colormap, colornums)
        colors_dark = darken_colors(colors, 0.5)
        for obj in updated_fig.data:
            if obj.uid is not None and (("traj" in obj.uid) or ("proj" in obj.uid)):
                num = int(obj.uid[4:])
                obj.line.color = colors[num] if "traj" in obj.uid else colors_dark[num]
        updated_fig.update_traces(
            marker=dict(color=cdata, colorbar=dict(title=value.replace(" ", "<br>"))),
            selector=dict(name="dummy"),
        )
        return updated_fig

    @callback(
        Output("download-html-file", "data"),
        Input("btn_download_html", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_html(n_clicks):
        if n_clicks:
            html_content = fig.to_html(full_html=False, include_plotlyjs="cdn")
            return dcc.send_string(html_content, filename="enter_name.html")

    app.run(debug=False, use_reloader=False, port=port)


def hodographs(
    df: pd.DataFrame,
    params=[
        "Initial x",
        "Initial y",
        "Initial z",
        "Initial vx",
        "Initial vy",
        "Initial vz",
        "Period",
        "Jacobi Constant",
    ],
    colormap="rainbow",
):
    n = len(df)
    params = [
        param
        for param in params
        if param in df.columns and not np.all(np.abs(df[param]) < 1e-10)
    ]
    nparam = len(params)

    fig = make_subplots(
        rows=nparam - 1,
        cols=nparam - 1,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.02,
        horizontal_spacing=0.01,
    )

    c = px.colors.sample_colorscale(colormap, n)
    for ix, param_x in enumerate(params[:-1]):
        for iy, param_y in enumerate(params[1:]):
            data_x = df[param_x].values
            data_y = df[param_y].values

            if iy == 0:
                fig.update_xaxes(title=param_x, row=nparam - 1, col=ix + 1)
            if ix == 0:
                fig.update_yaxes(title=param_y, row=iy + 1, col=1)

            if ix > iy:
                continue

            curve = go.Scatter(
                x=data_x,
                y=data_y,
                name=f"{ix}, {iy}",
                hoverinfo="x+y",
                mode="lines",
                hoverlabel=dict(namelength=-1, bgcolor="black", font_color="white"),
                # marker=dict(color=c, size=1),
                line=dict(color="white", width=0.5),
            )
            fig.add_trace(curve, row=iy + 1, col=ix + 1)

    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=0),
        # plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        modebar_remove=["autoScale", "lasso2d", "select2d", "toImage"],
    )

    fig.show()


def broucke_diagram(df: pd.DataFrame, html_save: str | None = None, show: bool = True):
    n = len(df)
    colormap = "rainbow"
    eig_df = df[[col for col in df.columns if "Eig" in col]]
    jcs = df["Jacobi Constant"]
    eigs = eig_df.values.astype(np.complex128)
    alpha = 2 - np.sum(eigs, axis=1).real
    beta = (alpha**2 - (np.sum(eigs**2, axis=1).real - 2)) / 2
    alphrange = np.max(np.abs(alpha))
    x = np.linspace(-((alphrange) ** (1 / 3)), (alphrange) ** (1 / 3), 1000) ** 3
    x = np.unique([*x, *np.linspace(-3, 3, 50)])

    def get_per_mult(x, n, q=1):
        a = -2 * np.cos(2 * np.pi * q / n)
        b = 2 - 4 * (np.cos(2 * np.pi * q / n)) ** 2
        return a * x + b

    lines_cross = np.array(
        [
            get_per_mult(x, 2),
            get_per_mult(x, 3),
            get_per_mult(x, 4),
            get_per_mult(x, 5, 1),
            get_per_mult(x, 5, 2),
            get_per_mult(x, 6),
            get_per_mult(x, 7, 1),
            get_per_mult(x, 7, 2),
            get_per_mult(x, 7, 3),
            get_per_mult(x, 8, 1),
            get_per_mult(x, 8, 3),
            get_per_mult(x, 9, 1),
            get_per_mult(x, 9, 2),
            get_per_mult(x, 9, 4),
            -2 * x - 2,
            x**2 / 4 + 2,
        ]
    )
    lines_names = [
        "Period-Double",
        "Period-Triple",
        "Period-Quadrouple",
        "Period-Quintuple (1)",
        "Period-Quintuple (2)",
        "Period-Sextuple",
        "Period-Septuple (1)",
        "Period-Septuple (2)",
        "Period-Septuple (3)",
        "Period-Octuple (1)",
        "Period-Octuple (3)",
        "Period-Nonuple (1)",
        "Period-Nonuple (2)",
        "Period-Nonuple (4)",
        "Tangent",
        "Hopf",
    ]
    eqns = [
        r"2\alpha - 2",
        r"\alpha + 1",
        r"2",
        r"\frac{1-\sqrt{5}}{2}\alpha+\frac{1+\sqrt{5}}{2}\alpha",
        r"\frac{1+\sqrt{5}}{2}\alpha+\frac{1-\sqrt{5}}{2}\alpha",
        r"-\alpha+1",
        r"-2\cos(\frac{2\pi}{7})\alpha+2-4\cos^2(\frac{2\pi}{7})",
        r"-2\cos(\frac{4\pi}{7})\alpha+2-4\cos^2(\frac{4\pi}{7})",
        r"-2\cos(\frac{6\pi}{7})\alpha+2-4\cos^2(\frac{6\pi}{7})",
        r"-\sqrt{2}\alpha",
        r"\sqrt{2}\alpha",
        r"-2\cos(\frac{2\pi}{9})\alpha+2-4\cos^2(\frac{2\pi}{9})",
        r"-2\cos(\frac{4\pi}{9})\alpha+2-4\cos^2(\frac{4\pi}{9})",
        r"-2\cos(\frac{8\pi}{9})\alpha+2-4\cos^2(\frac{8\pi}{9})",
        r"-2\alpha - 2",
        r"\frac{\alpha^2}{4}+2",
    ]
    # generally, beta = a*alpha+b where a = -2cos(q2pi/n), 2-4cos^2(q2pi/n) for n-periodic and q\in 1..n/2

    c = px.colors.sample_colorscale(colormap, n)
    curve = go.Scatter(
        x=alpha,
        y=beta,
        name="Family",
        text=[f"Index: {ind}<br>JC: {jc:.6f}" for ind, jc in zip(df.index, jcs)],
        hoverinfo="text",
        mode="lines+markers",
        hoverlabel=dict(namelength=-1, bgcolor="black", font_color="white"),
        marker=dict(color=c, size=4),
        line=dict(color="white", width=0.75),
    )

    hide = ["sextuple", "septuple", "octuple", "nonuple"]
    guides = [
        go.Scatter(
            x=x,
            y=y,
            name=rf"{name}",  #: $\beta={eqn}$",
            hoverinfo="text",
            text=name,
            mode="lines",
            hoverlabel=dict(namelength=-1, bgcolor="black", font_color="white"),
            line=dict(width=1),
            visible=(
                "legendonly" if any(term in name.lower() for term in hide) else True
            ),
        )
        for y, name, eqn in zip(lines_cross, lines_names, eqns)
    ]

    fig = go.Figure(data=[*guides, curve])

    fig.update_layout(
        title=dict(
            text="Broucke Diagram", x=0.5, xanchor="center", yanchor="bottom", y=0.95
        ),
        xaxis_range=[-5, 5],
        yaxis_range=[-5, 5],
        width=1000,
        height=800,
        template="plotly_dark",
        showlegend=True,
        margin=dict(l=10, r=30, b=10, t=50),
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        xaxis=dict(title=r"$\alpha$"),
        yaxis=dict(title=r"$\beta$"),
    )
    fig.update_xaxes(exponentformat="power")
    fig.update_yaxes(exponentformat="power")
    if show:
        fig.show()

    if html_save is not None:
        fig.write_html(html_save, include_plotlyjs="cdn")


def bifurcation_search(
    df: pd.DataFrame,
    html_save: str | None = None,
    show: bool = True,
    line_color_by: str | None = None,
):
    # assert line_color_by in [None, "Period", "Jacobi Constant"]
    n = len(df)
    jacobis = df["Jacobi Constant"].values
    periods = df["Period"].values
    jc_deriv = np.gradient(jacobis)
    per_deriv = np.gradient(periods)

    # DETECT EXTREMA
    dif_period_p = periods[2:] - periods[1:-1]
    dif_period_m = periods[1:-1] - periods[:-2]
    per_sign_change = np.sign(dif_period_p) != np.sign(dif_period_m)
    per_ineq = (periods[2:] != periods[1:-1]) * (periods[1:-1] != periods[:-2])
    period_indices = np.where(per_sign_change * per_ineq)[0] + 1

    dif_jacobi_p = jacobis[2:] - jacobis[1:-1]
    dif_jacobi_m = jacobis[1:-1] - jacobis[:-2]
    jcb_sign_change = np.sign(dif_jacobi_p) != np.sign(dif_jacobi_m)
    jcb_ineq = (jacobis[2:] != jacobis[1:-1]) * (jacobis[1:-1] != jacobis[:-2])
    jacobi_indices = np.where(jcb_sign_change * jcb_ineq)[0] + 1

    eig_df = df[[col for col in df.columns if "Eig" in col]]
    eigs = eig_df.values.astype(np.complex128)
    alpha = 2 - np.sum(eigs, axis=1).real
    beta = (alpha**2 - (np.sum(eigs**2, axis=1).real - 2)) / 2

    def get_per_mult(x, n, q=1):
        a = -2 * np.cos(2 * np.pi * q / n)
        b = 2 - 4 * (np.cos(2 * np.pi * q / n)) ** 2
        return a * x + b

    lines_cross = np.array(
        [
            get_per_mult(alpha, 2),
            get_per_mult(alpha, 3),
            get_per_mult(alpha, 4),
            get_per_mult(alpha, 5, 1),
            get_per_mult(alpha, 5, 2),
            get_per_mult(alpha, 6),
            get_per_mult(alpha, 7, 1),
            get_per_mult(alpha, 7, 2),
            get_per_mult(alpha, 7, 3),
            get_per_mult(alpha, 8, 1),
            get_per_mult(alpha, 8, 3),
            get_per_mult(alpha, 9, 1),
            get_per_mult(alpha, 9, 2),
            get_per_mult(alpha, 9, 4),
            -2 * alpha - 2,
            alpha**2 / 4 + 2,
        ]
    )
    lines_names = [
        "Double",
        "Triple",
        "Quadrouple",
        "Quintuple (1)",
        "Quintuple (2)",
        "Sextuple",
        "Septuple (1)",
        "Septuple (2)",
        "Septuple (3)",
        "Octuple (1)",
        "Octuple (3)",
        "Nonuple (1)",
        "Nonuple (2)",
        "Nonuple (4)",
        "Tangent",
        "Hopf",
    ]

    red = "rgb(100, 0, 0)"
    grn = "rgb(0, 100, 0)"
    per_colors = [red if prd < 0 else grn for prd in per_deriv]
    jc_colors = [red if jcd < 0 else grn for jcd in jc_deriv]

    if line_color_by == "Jacobi Constant":
        line_col = jc_colors
    elif line_color_by == "Period":
        line_col = per_colors
    else:
        line_col = None

    xs = list(range(n))
    zero = go.Scatter(
        x=xs,
        y=np.zeros_like(xs),
        text=[
            f"Index: {ind}<br>JC: {jc:.6f}<br>Period: {period:.6f}"
            for ind, jc, period in zip(df.index, jacobis, periods)
        ],
        hoverinfo="text",
        mode="lines+markers" if line_col else "lines",
        hoverlabel=dict(namelength=-1, bgcolor="black", font_color="white"),
        line=dict(width=0.75, color="white"),
        marker=dict(size=3, color=line_col) if line_col else None,
        showlegend=False,
        visible=True,
    )
    curves = [zero]
    for name, line in zip(lines_names, lines_cross):
        curve = go.Scatter(
            x=xs,
            y=(line - beta),
            name=name,
            # text=[
            #     f"{name}<br>Index: {ind}<br>JC: {jc:.6f}<br>Period: {per:.6f}"
            #     for ind, jc, per in zip(df.index, jacobis, periods)
            # ],
            hoverinfo="name",
            mode="lines+markers",
            hoverlabel=dict(namelength=-1, bgcolor="black", font_color="white"),
            marker=dict(size=5),
            line=dict(width=1),
            visible=(True if name in ["Hopf", "Tangent"] else "legendonly"),
        )
        curves.append(curve)
    fig = go.Figure(data=curves)

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="red", dash="dot"),
            name="Fold Candidate",
            showlegend=True,
            opacity=0.5,
        )
    )
    for ind in set(list(period_indices) + list(jacobi_indices)):
        fig.add_vline(
            x=ind, line=dict(dash="dot", color="red"), layer="below", opacity=0.3
        )
    fig.update_layout(
        template="plotly_dark",
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=0),
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        xaxis=dict(title=r"Index", showgrid=False, zeroline=False),
        hovermode="x",
        modebar_remove=["autoScale", "lasso2d", "select2d", "toImage"],
        legend=dict(y=0, yanchor="bottom"),
    )
    fig.update_yaxes(exponentformat="power")

    fig.update_layout(xaxis=dict(modebardisable="zoominout+autoscale"))
    if show:
        config = dict(displaylogo=False, displayModeBar=True)
        fig.show(config=config)

    if html_save is not None:
        fig.write_html(html_save, include_plotlyjs="cdn")


def test_points(df_compare: pd.DataFrame, colormap="rainbow"):
    for col in ["Initial " + state for state in ["x", "y", "z", "vx", "vy", "vz"]]:
        if col in df_compare.columns and np.all(np.abs(df_compare[col])) < 1e-9:
            df_compare = df_compare.drop(columns=col)
    cols = df_compare.columns
    newtype = (
        "Spatial" if (("Initial z" in cols) or ("Initial vz" in cols)) else "Planar"
    )

    root = f"{os.getcwd()}/database/"
    filenames = [
        file.removesuffix(".csv") for file in os.listdir(root) if file.endswith("csv")
    ]
    vals_dict = {}
    for fname in filenames:
        path = root + fname + ".csv"
        db = pd.read_csv(path).set_index("Index")
        for col in ["Initial " + state for state in ["x", "y", "z", "vx", "vy", "vz"]]:
            if col in db.columns and np.all(np.abs(db[col])) < 1e-9:
                db = db.drop(columns=col)
        cols = db.columns
        periods = db["Period"]
        jcs = db["Jacobi Constant"]
        orbtype = "Spatial" if "Initial z" in cols or "Initial vz" in cols else "Planar"
        if orbtype == newtype:
            vals_dict[fname] = {
                "Period": periods.values,
                "JC": jcs.values,
                "Type": orbtype,
            }

    periods = df_compare["Period"].values
    jcs = df_compare["Jacobi Constant"].values

    per_range = max(periods) - min(periods)
    jc_range = max(jcs) - min(jcs)
    xlim = (max(periods) + min(periods)) / 2 + np.array([-per_range / 2, per_range / 2])
    ylim = (max(jcs) + min(jcs)) / 2 + np.array([-jc_range / 2, jc_range / 2])

    curve1 = go.Scatter(
        x=periods,
        y=jcs,
        name="Compare",
        hoverinfo="name",
        mode="lines",
        line=dict(width=1.5, color="white"),
        showlegend=False,
    )

    n = len(vals_dict)
    colors = px.colors.sample_colorscale(colormap, n)
    curves = [curve1]
    j = 0
    for name, dct in vals_dict.items():
        curve = go.Scatter(
            x=dct["Period"],
            y=dct["JC"],
            name=name,
            hoverinfo="name",
            mode="lines",
            line=dict(width=1, color=colors[j]),
        )
        j += 1
        curves.append(curve)

    curveend = go.Scatter(
        x=periods,
        y=jcs,
        name="Compare",
        hoverinfo="name",
        mode="markers",
        marker=dict(color="white", size=4),
    )

    curves.append(curveend)
    fig = go.Figure(data=curves)
    fig.update_layout(
        template="plotly_dark",
        showlegend=True,
        xaxis_range=list(xlim),
        yaxis_range=list(ylim),
        margin=dict(l=0, r=0, b=0, t=0),
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        xaxis=dict(title="Period"),
        yaxis=dict(title="Jacobi Constant"),
        modebar_remove=["autoScale", "lasso2d", "select2d", "toImage"],
    )

    config = dict(displaylogo=False, displayModeBar=True)
    fig.show(config=config)
