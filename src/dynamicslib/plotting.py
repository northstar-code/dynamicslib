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
from base64 import b64encode
from dash import Dash, dcc, html, Input, Output, callback

from dynamicslib.consts import muEM
from dynamicslib.common import get_Lpts

plotly.offline.init_notebook_mode()
display(
    HTML(
        '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
    )
)


def matplotlib_family(
    Xs_all: List,
    xyzs_all: list,
    names: list,
    eig_vals_all: list,
    X2xtf_func: Callable,
    colormap: str = "rainbow",
    mu: float = muEM,
):
    # %config InlineBackend.print_figure_kwargs = {'bbox_inches':None}

    assert len(Xs_all) == len(xyzs_all) == len(names) == len(eig_vals_all)
    Lpoints = get_Lpts(mu)
    num_fams = len(Xs_all)
    cm = plt.get_cmap(colormap)

    for ifam in range(num_fams):
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(121, projection="3d")
        Xs = Xs_all[ifam]
        eig_vals = eig_vals_all[ifam]
        n = len(Xs)
        # plot the 3D
        xyzs_fam = xyzs_all[ifam]
        for i, xyzs in enumerate(xyzs_fam):
            ax.plot(*xyzs, "-", lw=1, color=cm(i / n))

        # find the limits so we can plot and preserve them
        ax.axis("equal")
        lims = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
        xl, yl, zl = (
            1.2 * (lims - np.mean(lims, axis=1)[:, None])
            + np.mean(lims, axis=1)[:, None]
        )
        ax.set(xlim=xl, ylim=yl, zlim=zl)

        # plot the projections
        for i, xyzs in enumerate(xyzs_fam):
            x, y, z = xyzs
            ax.plot(xl[0], y, z, "-", lw=0.3, color=cm(i / n), alpha=0.5)
            ax.plot(x, yl[1], z, "-", lw=0.3, color=cm(i / n), alpha=0.5)
            ax.plot(x, y, zl[0], "-", lw=0.3, color=cm(i / n), alpha=0.5)

        # Lpoints and bodies
        ax.scatter(Lpoints[0], Lpoints[1], c="c", s=6, alpha=1, axlim_clip=True)
        ax.scatter([-mu, 1 - mu], [0, 0], c="w", s=25, alpha=1, axlim_clip=True)

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis._axinfo["grid"]["linewidth"] = 0.3
            axis._axinfo["grid"]["color"] = "darkgrey"
            axis.set_label_coords(0, 0)

        ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$", title="Position Space")
        ax.tick_params(axis="both", which="major", labelsize=7)
        fig.suptitle(names[ifam])

        # stability, period, energy
        ax = fig.add_subplot(322)
        ax2 = fig.add_subplot(324, sharex=ax)
        ax3 = fig.add_subplot(326, sharex=ax)
        x_axis = list(range(n))
        periods = []
        jacobi_consts = []
        stabs1 = []
        stabs2 = []
        for i, X in enumerate(Xs):
            jc, tf = get_JC_tf(X, X2xtf_func)
            periods.append(tf)
            jacobi_consts.append(jc)
            eigvals = eig_vals[i]
            eigvals = eigvals[np.argsort(np.abs(eigvals))]
            stabs1.append(np.abs(eigvals[0] + 1 / eigvals[0]) / 2)
            stabs2.append(np.abs(eigvals[1] + 1 / eigvals[1]) / 2)

        ax.scatter(x_axis, jacobi_consts, c=x_axis, alpha=1, cmap=cm)
        ax2.scatter(x_axis, periods, c=x_axis, alpha=1, cmap=cm)
        ax3.scatter(x_axis, stabs1, c=x_axis, alpha=1, cmap=cm)
        ax3.scatter(x_axis, stabs2, c=x_axis, alpha=1, cmap=cm)
        ax.set(ylabel="Jacobi Constant", title="Family Evolution")
        ax2.set(ylabel="Peroid")
        ax3.set(xlabel="Index Along Family", ylabel="Stability Index")
        ax3.set_yscale("log")
        ax.grid(True, lw=0.25)
        ax2.grid(True, lw=0.25)
        ax3.grid(True, lw=0.25)
        ax.set_xticklabels([])
        ax2.set_xticklabels([])
        fig.tight_layout(h_pad=0, w_pad=0)
    plt.show()


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


def darken_colors(color_list, scale=0.75):
    out = tuple([darken_color(string, scale) for string in color_list])
    return out


def plotly_family(
    xyzs: List,
    name: str,
    data: List | NDArray,
    param_names: list,
    colormap: str = "rainbow",
    mu: float = muEM,
    renderer: str | None = "browser",
    color_by: str = "Index",
    html_save: str | None = None,
    alpha: float = 1,
    figsize: tuple[float, float] = (700, 500),
):
    data = np.array(data)
    data = data.astype(np.float32)
    datatr = data.T

    assert len(xyzs) == len(data)
    assert len(data[0]) == len(param_names)

    xyzs = [np.float32(xyz) for xyz in xyzs]
    if html_save is not None and html_save.endswith(".html"):
        html_save = html_save.rstrip(".html")

    if renderer is not None:
        pio.renderers.default = renderer

    n = len(xyzs)

    cdata = datatr[param_names.index(color_by)]  # color data

    xs, ys, zs = np.hstack(xyzs)
    minx, miny, minz = (min(xs), min(ys), min(zs))
    maxx, maxy, maxz = (max(xs), max(ys), max(zs))
    rng = 1.25 * max([maxx - minx, maxy - miny, maxz - minz])

    ctrX = (maxx + minx) / 2
    ctrY = (maxy + miny) / 2
    ctrZ = (maxz + minz) / 2
    projX = ctrX - rng / 2
    projY = ctrY - rng / 2
    projZ = ctrZ - rng / 2
    projs = []
    curves3d = []

    colornums = cdata - min(cdata)
    colornums /= max(colornums)
    colors = px.colors.sample_colorscale(colormap, colornums)
    for i, xyzs in enumerate(xyzs):
        x, y, z = xyzs
        lbl = make_label(data[i], param_names)
        c = colors[i]
        curves3d.append(plotly_curve(x, y, z, lbl, color=c, width=5, opacity=alpha))
        projs.append(
            plotly_curve(x * 0 + projX, y, z, color=c, width=2, opacity=0.75 * alpha)
        )
        projs.append(
            plotly_curve(x, 0 * y + projY, z, color=c, width=2, opacity=0.75 * alpha)
        )
        projs.append(
            plotly_curve(x, y, 0 * z + projZ, color=c, width=2, opacity=0.75 * alpha)
        )

    cbar_dummy = go.Scatter3d(
        x=[np.nan] * n,
        y=[np.nan] * n,
        z=[np.nan] * n,
        mode="markers",
        marker=dict(
            color=cdata,
            colorscale=colormap,
            colorbar=dict(title=color_by.replace(" ", "<br>"), thickness=12),
        ),
    )

    fig = go.Figure(data=[cbar_dummy, *curves3d, *projs])

    Lpoints = get_Lpts(mu=mu)
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

    fig.update_layout(
        title=dict(text=name, x=0.5, xanchor="center", yanchor="bottom", y=0.95),
        width=figsize[0],
        height=figsize[1],
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=0, r=30, b=0, t=50),
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
    )
    fig.update_scenes(
        xaxis=dict(
            title="x [nd]",
            showbackground=False,
            showgrid=True,
            zeroline=False,
            range=[ctrX - rng / 2, ctrX + rng / 2],
        ),
        yaxis=dict(
            title="y [nd]",
            showbackground=False,
            showgrid=True,
            zeroline=False,
            range=[ctrY - rng / 2, ctrY + rng / 2],
        ),
        zaxis=dict(
            title="z [nd]",
            showbackground=False,
            showgrid=True,
            zeroline=False,
            range=[ctrZ - rng / 2, ctrZ + rng / 2],
        ),
        aspectmode="cube",
    )

    argshide = {"visible": [True, *[True] * n, *[False] * (3 * n), True, True]}
    argsshow = {"visible": [True, *[True] * (4 * n + 2)]}
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.2,
                y=1,
                xanchor="left",
                yanchor="bottom",
                showactive=False,
                buttons=[
                    dict(
                        label="Show<br>Projections", method="restyle", args=[argsshow]
                    ),
                    dict(
                        label="Hide<br>Projections", method="restyle", args=[argshide]
                    ),
                ],
            ),
        ]
    )

    trace = go.Scatter(
        x=list(range(n)),
        y=datatr[param_names.index(color_by)],
        mode="markers",
        name="Parameter Sweep",
        marker=dict(color=px.colors.sample_colorscale(colormap, np.arange(n) / n)),
    )
    fig2 = go.Figure(data=trace)
    fig2.update_layout(
        title=dict(
            text=name + " Parameter Sweep",
            x=0.5,
            xanchor="center",
            yanchor="bottom",
            y=0.95,
        ),
        # xaxis=dict(title="Index Along Family"),
        width=figsize[0],
        height=figsize[1],
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=0, r=0, b=50, t=50),
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
    )

    fig2.update_yaxes(exponentformat="power")

    # DROPDOWNS
    fig2.update_layout(
        updatemenus=[
            dict(
                buttons=list(
                    [
                        dict(
                            label="x: " + param,
                            method="update",
                            args=[{"x": [datatr[i]]}, {"xaxis.title.text": param}],
                        )
                        for i, param in enumerate(param_names)
                    ]
                ),
                direction="down",
                showactive=True,
                x=0,
                xanchor="left",
                y=1,
                yanchor="bottom",
            ),
            dict(
                buttons=list(
                    [
                        dict(
                            label="y: " + param,
                            method="update",
                            args=[{"y": [datatr[i]]}, {"yaxis.title.text": param}],
                        )
                        for i, param in enumerate(param_names)
                    ]
                ),
                direction="down",
                showactive=True,
                x=1,
                xanchor="right",
                y=1,
                yanchor="bottom",
            ),
        ]
    )

    if html_save is not None:
        fig.write_html(html_save + "_3d.html", include_plotlyjs="cdn")
        fig2.write_html(html_save + "_sweep.html", include_plotlyjs="cdn")
    if renderer is not None:
        fig.show()
        fig2.show()


def plotly_display(
    xyzs: List,
    full_dataframe: pd.DataFrame,
    colormap: str = "rainbow",
    mu: float = muEM,
    figsize: tuple[float, float] = (900, 600),
    port=8050,
    flip_vert=False,
    flip_horiz=False,
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

    cdata = datatr[param_names.index("Index")]  # color data

    # get bounding box
    xs, ys, zs = np.hstack(xyzs)
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
    for i, xyzs in enumerate(xyzs):
        x, y, z = xyzs
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
        Output("display", "figure", allow_duplicate=True),
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
