from numba import njit
from numpy.typing import NDArray
import numpy as np


@njit(cache=True)
def Hermite_interp_interval(
    t: float | NDArray,
    t0: float,
    t1: float,
    x0: float | NDArray,
    x1: float | NDArray,
    dxdt0: float | NDArray,
    dxdt1: float | NDArray,
) -> float | NDArray:
    """Cubic Hermite spline interpolation. Source: Wikipedia

    Args:
        t (float): time to evaluate at
        t0 (float): Beginning time of two points
        t1 (float): End time of two points
        x0 (float | NDArray): First state
        x1 (float | NDArray): Second state
        dxdt0 (float | NDArray): First tangent
        dxdt1 (float | NDArray): Second tangent

    Returns:
        NDArray|Float: interpolated value
    """
    dt = t1 - t0  # time between samples
    tau = (t - t0) / (t1 - t0)  # fraction of time interval
    # polynomial bases
    h00 = 2 * tau**3 - 3 * tau**2 + 1
    h10 = tau**3 - 2 * tau**2 + tau
    h01 = -2 * tau**3 + 3 * tau**2
    h11 = tau**3 - tau**2

    x0e = np.expand_dims(x0, axis=1)
    x1e = np.expand_dims(x1, axis=1)
    dxdt0e = np.expand_dims(dxdt0, axis=1)
    dxdt1e = np.expand_dims(dxdt1, axis=1)

    return (h00 * x0e + h10 * dt * dxdt0e + h01 * x1e + h11 * dt * dxdt1e).T


@njit(cache=True)
def interp_hermite(
    t: NDArray[np.floating],
    x: NDArray[np.floating],
    dxdt: NDArray[np.floating],
    t_eval: NDArray[np.floating] | None = None,
    n_mult: int | None = None,
):
    """Hermite spline interpolation. Provide either times to evaluate at or a density multiplier

    Args:
        t (NDArray): Originally evaluated times (N, )
        x (NDArray): Originally evaluated values (N, nx)
        dxdt (NDArray): Derivatives at evaluated times (N, nx)
        t_eval (NDArray | None, optional): Times to evaluate at (N2, ). Defaults to None.
        n_mult (int | None, optional): density multiplier. Defaults to None.

    Returns:
        _type_: new ts (N2, ), new xs (N2, nx)
    """

    # flip = t[-1] < t[0]  # whether direction is flipped
    if len(np.shape(x)) != 2:
        raise NotImplementedError(
            "can only interpolate vectors for now, so x and dxdt must be 2D"
        )

    if t_eval is None:
        if n_mult is not None:
            t_eval = np.empty((0,), np.float64)
            for a in range(len(t) - 1):
                toadd = np.linspace(t[a], t[a + 1], n_mult + 1)[:-1]
                t_eval = np.append(t_eval, toadd)
            t_eval = np.append(t_eval, np.array([t[-1]]))
        else:
            raise ValueError("Must provide value for t_eval or n_mult")
    else:
        assert t_eval[0] >= t[0] and t_eval[-1] <= t[-1]

    x_eval = np.empty((0, len(x[0])), dtype=np.float64)
    for j in range(len(t) - 1):
        t0 = t[j]
        t1 = t[j + 1]

        if j == 0:
            ts = t_eval[np.logical_and(t0 <= t_eval, t_eval <= t1)].copy()
        else:
            ts = t_eval[np.logical_and(t0 < t_eval, t_eval <= t1)].copy()

        if len(ts):
            x0 = x[j]
            x1 = x[j + 1]
            dxdt0 = dxdt[j]
            dxdt1 = dxdt[j + 1]
            newterms = Hermite_interp_interval(ts, t0, t1, x0, x1, dxdt0, dxdt1)
            x_eval = np.concatenate((x_eval, newterms))

    return t_eval, x_eval


@njit(cache=True)
def dop_interp_step(
    t_eval: NDArray[np.floating],
    x0: NDArray[np.floating],
    t0: float,
    tf: float,
    F: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Interpolate one timestep of DOP853 output. Probably slightly suboptimal, based on Scipy implementation of dense output

    Args:
        t_eval (NDArray[np.floating]): Times at which to evaluate, must be in [t0, tf]
        x0 (NDArray[np.floating]): State at the beginning of the timestep
        t0 (float): Time at the start of the timestep
        tf (float): Time at the end of the timestep
        F (NDArray[np.floating]): Intermediate function evaluations matrix from DOP output (7xNstates)

    Returns:
        NDArray[np.floating]: States evaluated at given times
    """
    delta_t = tf - t0

    te = (t_eval - t0) / delta_t

    te = te[:, None]
    xs = np.zeros((len(te), len(x0)), dtype=np.float64)

    for i, f in enumerate(F[::-1]):
        xs += f
        if i % 2 == 0:
            xs *= te
        else:
            xs *= 1 - te
    xs += x0
    return xs


# @njit(cache=True)
def dop_interpolate(
    ts: NDArray,
    xs: NDArray,
    Fs: NDArray,
    t_eval: NDArray | None = None,
    n_mult: int | None = None,
) -> NDArray[np.floating]:
    """Interpolate DOP853 results. For now only works with forward-integrated results, though I intend to change this. Must provide either n_mult or t_eval

    Args:
        ts (NDArray): Original evaluation times (Nt, )
        xs (NDArray): Original evaluation states (Nt, Nx)
        Fs (NDArray): Intermediate function evaluations (Nt-1, 7, Nx)
        t_eval (NDArray | None, optional): Times to evaluate at (Nte, ). Defaults to None.
        n_mult (int | None, optional): Multiply temporal density by this much. Defaults to None.

    Raises:
        ValueError: If you fail to give anything for t_eval or n_mult

    Returns:
        NDArray[np.floating]: States evaluated at requested times (Nx, Nte)
    """

    if t_eval is None:
        if n_mult is not None:
            t_eval = np.empty((0,), np.float64)
            for a in range(len(ts) - 1):
                toadd = np.linspace(ts[a], ts[a + 1], n_mult + 1)[:-1]
                t_eval = np.append(t_eval, toadd)
            t_eval = np.append(t_eval, np.array([ts[-1]]))
        else:
            raise ValueError("Must provide value for t_eval or n_mult")
    else:
        assert t_eval[0] >= ts[0] and t_eval[-1] <= ts[-1]

    x_eval = np.empty((0, len(xs[0])), dtype=np.float64)
    # tlst = []
    for j in range(len(ts) - 1):
        t0 = ts[j]
        t1 = ts[j + 1]

        if j == len(ts) - 2:
            ts_interval = t_eval[np.logical_and(t0 <= t_eval, t_eval <= t1)].copy()
        else:
            ts_interval = t_eval[np.logical_and(t0 <= t_eval, t_eval < t1)].copy()
        # if j == len(ts) - 2:
        #     ts_interval = np.append(ts_interval, t1)
        # # for tt in ts_interval:
        #     tlst.append(tt)

        if len(ts_interval):
            x0 = xs[j]
            Fs_interval = Fs[j]
            newterms = dop_interp_step(ts_interval, x0, t0, t1, Fs_interval)
            x_eval = np.concatenate((x_eval, newterms))

    return t_eval, x_eval.T  # , tlst
