from numba import njit
from numpy.typing import NDArray
import numpy as np
from numba import float64 as nbfloat64
from numba.typed import List as nbList
from typing import Tuple, Callable, List
import dynamicslib.DOP853_coefs as coefs
from dynamicslib.interpolate import dop_interpolate, interpolate_event


# Keeping for later, probably wont use
@njit(cache=True)
def rkf45(
    func: Callable,
    t_span: Tuple[float, float],
    x0: NDArray,
    atol: float = 1e-10,
    rtol: float = 1e-10,
    init_step: float = 1e-6,
) -> Tuple[NDArray, NDArray]:
    """Runge Kutta 45

    Args:
        func (Callable): _description_
        t_span (Tuple[float, float]): _description_
        x0 (NDArray): _description_
        atol (float, optional): _description_. Defaults to 1e-10.
        rtol (float, optional): _description_. Defaults to 1e-10.
        init_step (float, optional): _description_. Defaults to 1e-6.

    Returns:
        Tuple[NDArray, NDArray]: _description_
    """

    t0, tf = t_span
    t = t0
    xs = np.expand_dims(x0, axis=0)
    ts = np.array([t0], dtype=np.float64)
    x = x0
    h = init_step

    while t < tf:
        if t + h > tf:
            h = tf - t
        tol = atol + np.linalg.norm(x, np.inf) * rtol
        # fmt: off
        k1 = h * func(t, x)
        k2 = h * func(t + (1/4)*h, x + (1/4)*k1)
        k3 = h * func(t + (3/8)*h, x + (3/32)*k1 + (9/32)*k2)
        k4 = h * func(t + (12/13)*h, x + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3)
        k5 = h * func(t + h, x + (439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4)
        k6 = h * func(t + (1/2)*h, x - (8/27)*k1 + 2*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5)
        x4 = x + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4104) * k4 - (1 / 5) * k5
        x5 = x + (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9 / 50)*k5 + (2 / 55)*k6
        # fmt: on

        error = np.linalg.norm(x5 - x4)

        # no div0
        s = (tol / error) ** 0.25 if error != 0 else 2

        if s > 2:
            s = 2
        if s < 0.1:
            s = 0.1

        if error <= tol:
            t += h
            x = x5
            ts = np.concatenate((ts, np.array([t], dtype=np.float64)))
            xs = np.concatenate((xs, np.expand_dims(x, axis=0)))

        h *= s

    return ts, xs


# A single RK8 step
@njit(cache=True)
def rk8_step(
    func: Callable,
    sv0: NDArray,
    t0: float,
    dt: float,
    args: Tuple = (),
) -> NDArray:
    """Single RK8 step

    Args:
        func (Callable): dynamics function
        t_span (Tuple[float, float]): beginning and end times
        sv0 (NDArray): initial state, doesnt have to be 1d
        dt (float): amount of time to propagate for
        args (Tuple, optional): additional args to func(t, x, *args). Defaults to ().

    Returns:
        NDArray: propagated
    """
    x_shape = np.shape(sv0)
    sv0 = sv0.copy().flatten()
    n = len(sv0)

    K = np.empty((coefs.n_stages + 1, n), dtype=np.float64)
    K[0] = func(t0, sv0, *args)
    for sm1 in range(coefs.N_STAGES - 1):
        s = sm1 + 1
        a = coefs.A[s]
        c = coefs.C[s]
        dsv = np.dot(K[:s].T, a[:s]) * dt
        K[s] = func(t0 + c * dt, sv0 + dsv, *args)

    svnew = sv0 + dt * np.dot(K[:-1].T, coefs.B)

    return np.reshape(svnew, x_shape)


@njit(cache=True)
def dop853_dense_extra(
    func: Callable,
    h: float,
    t0: float,
    xf: NDArray[np.floating],
    x0: NDArray[np.floating],
    f_fin: NDArray[np.floating],
    K_ext: NDArray,
    n: int,
    args: tuple = (),
) -> NDArray[np.floating]:
    """Interpolation for DOP583 requires a couple extra function evaluations. This function does those

    Args:
        func (Callable): Function call with signature (t, x, *args)
        h (float): Step size for the interval
        t0 (float): Time at the beginning of the interval
        xf (NDArray[np.floating]): State at the end of the time interval (Nx, )
        x0 (NDArray[np.floating]): State at the start of the interval (Nx, )
        f_fin (NDArray[np.floating]): Function evaluation at the end of the time interval
        K_ext (NDArray): Extended interpolation matrix MUST BE PREFILLED FROM DOP CALL
        args (tuple): Additional args to func
        n (int): Number of state vars

    Returns:
        NDArray[np.floating]: Filled function evaluation matrix
    """
    for smod in range(3):
        s = smod + coefs.n_stages + 1
        a = coefs.A_EXTRA[smod]
        c = coefs.C_EXTRA[smod]
        dx = np.dot(K_ext[:s].T, a[:s]) * h
        K_ext[s] = func(t0 + c * h, x0 + dx, *args)

    F = np.empty((coefs.INTERPOLATOR_POWER, n), dtype=np.float64)

    f_old = K_ext[0]
    delta_x = xf - x0

    F[0] = delta_x
    F[1] = h * f_old - delta_x
    F[2] = 2 * delta_x - h * (f_fin + f_old)
    F[3:] = h * np.dot(coefs.D, K_ext)
    return F


@njit(cache=True)
def dop853(
    func: Callable[..., NDArray],
    t_span: Tuple[float, float],
    x0: NDArray,
    int_tol: float = 1e-12,
    atol: float | None = None,
    rtol: float | None = None,
    init_step: float = 1.0,
    events: List[Callable[..., float]] = [],
    directions: List[int] | None = None,
    terminals: List[int] | None = None,
    t_eval: NDArray | None = None,
    dense_output: bool = False,
    args: Tuple = (),
) -> Tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    Tuple[NDArray[np.floating], NDArray[np.floating]],
    Tuple[List, List],
]:
    """High order adaptive RK method with interpolation and events location capability.

    Args:
        func (Callable): dynamics function
        t_span (Tuple[float, float]): beginning and end times
        x0 (NDArray): initial state
        int_tol (float, optional): Absolute and relative tolerance (will be assigned to both). Defaults to 1e-12
        atol (float | None, optional): absolute tolerence. Defaults to None. If not None, will override int_tol
        rtol (float | None, optional): rel tolerence. Defaults to None. If not None, will override int_tol
        init_step (float, optional): initial step size. Defaults to 1.0.
        events (List[Callable]): List of ODE events functions. Each function has signature g(t, x, *args) with the same args as f, and returns a float
        directions (List | None): ODE event directions. 1 means only trigger when event is increasing, -1 only triggers when decreasing, 0 triggers on both. If the whole list is None, then all events are assigned direction 0
        terminals (List | None): ODE event terminal counts. Integration will halt at the the termination count specified. 0 means non-terminal event.
        t_eval (NDArray | None, optional): times to evaluate at. Will interpolate if these are specified. If None, returns only what's evaluated by the RK solver. Defaults to None, meaning all events are non-terminal
        dense_output (bool, optional): whether to collect interpolators. Doing so slightly increases computation time, so do not do if not necessary. If t_eval is provided or events are non-empty, then this is set to True
        args (Tuple, optional): additional args to func(t, x, *args). Defaults to ().

    Returns:
        NDArray: ts (N, )
        NDArray: xs (nx, N)
        Tuple: [NDArray, NDArray]: (EOM function evals (Nx, N), intermediate evaluations (Nt-1, 7, Nx))
        Tuple: [NDArray, NDArray]: (Event times [Ne x (Nevents, )], Event states [Ne x (nx, Nevents)])
    """

    nx = len(x0)
    if atol is None:
        atol = int_tol
    if rtol is None:
        rtol = int_tol
    if directions is None:
        directions = [0] * len(events)
    if terminals is None:
        terminals = [0] * len(events)

    if len(events):
        dense_output = True

    if t_eval is not None:
        dense_output = True

    halt = False
    forward = t_span[1] > t_span[0]

    # %% Prepare integrator
    K_ext = np.empty((coefs.N_STAGES_EXTENDED, nx), dtype=np.float64)
    K = K_ext[: coefs.n_stages + 1]

    t0, tf = t_span
    t = t0

    ts = nbList()
    ts.append(t0)

    xs = nbList()
    xs.append(x0)

    fs = nbList()
    fs.append(func(t0, x0, *args))

    Fs = nbList()
    F = np.zeros((coefs.INTERPOLATOR_POWER, nx), np.float64)
    if dense_output:
        Fs.append(F)

    # %% prepare events
    t_events = nbList()
    event_vals = nbList()
    x_events = nbList()

    for event in events:
        t_events_i = nbList.empty_list(nbfloat64)
        t_events.append(t_events_i)
        event_vals_i = nbList.empty_list(nbfloat64)
        event_vals_i.append(event(t0, x0, *args))
        event_vals.append(event_vals_i)
        x_events_i = nbList()
        x_events_i.append(np.zeros((nx,), dtype=np.float64))
        x_events_i.pop()
        x_events.append(x_events_i)

    # %% initialize
    x = x0.copy()
    h = abs(init_step) if forward else -abs(init_step)

    while (t < tf) if forward else (t > tf):
        if (t + h > tf) if forward else (t + h < tf):
            h = tf - t

        # syntax taken from Scipy implementation
        # %% take step
        K[0] = func(t, x, *args)
        for sm1 in range(coefs.N_STAGES - 1):
            s = sm1 + 1
            a = coefs.A[s]
            c = coefs.C[s]
            dy = np.dot(K[:s].T, a[:s]) * h
            K[s] = func(t + c * h, x + dy, *args)

        xnew = x + h * np.dot(K[:-1].T, coefs.B)

        K[-1] = func(t + h, xnew, *args)

        # %% error estimator:
        scale = atol + np.maximum(np.abs(x), np.abs(xnew)) * rtol
        err5 = np.dot(K.T, coefs.E5) / scale
        err3 = np.dot(K.T, coefs.E3) / scale
        err5_norm_2 = np.linalg.norm(err5) ** 2
        err3_norm_2 = np.linalg.norm(err3) ** 2
        denom = err5_norm_2 + 0.01 * err3_norm_2
        error = np.abs(h) * err5_norm_2 / np.sqrt(denom * len(scale))
        # END ERROR ESTIMDATOR

        # %% accept step
        hscale = 0.9 * error ** (-1 / 8) if error != 0 else 2

        # When the step is accepted
        if error <= 1:
            if dense_output:
                F = dop853_dense_extra(func, h, t, xnew, x, K[-1], K_ext, nx, args)
                Fs.append(F)

            # %% Event handling
            for jj, event in enumerate(events):
                g = event(t + h, xnew, *args)
                direction = directions[jj]
                terminal = terminals[jj]

                event_vals[jj].append(g)
                ev_vals = event_vals[jj]
                # handle event direction here
                if (
                    direction in [0, 1]
                    and np.sign(ev_vals[-2]) < 0
                    and np.sign(ev_vals[-1]) >= 0
                ) or (
                    direction in [-1, 0]
                    and np.sign(ev_vals[-2]) > 0
                    and np.sign(ev_vals[-1]) <= 0
                ):
                    te, xe = interpolate_event(
                        x,
                        xnew,
                        t,
                        t + h,
                        F,
                        ev_vals[-2],
                        ev_vals[-1],
                        event,
                        args,
                        delta=1e-1,
                    )
                    t_events[jj].append(te)
                    x_events[jj].append(xe)

                if len(t_events[jj]) == terminal and terminal > 0:
                    halt = True

            # %% step
            t += h
            x = xnew

            ts.append(t)
            xs.append(x)
            fs.append(K[-1])

            if halt:
                break

        h *= hscale

    # %% end, interpolate
    if t_eval is not None:
        if halt:  # Get rid of excess t_eval
            if forward and t_eval[-1] > t:
                t_eval = np.append(t_eval[t_eval <= t], t)
            elif not forward and t_eval[-1] <= t:
                t_eval = np.append(t_eval[t_eval <= t], t)
        _, xs_out = dop_interpolate(ts, xs, Fs, t_eval)
        ts_out = t_eval
    else:  # convert ts and xs to arrays
        nt = len(ts)
        ts_out = np.empty((nt), dtype=np.float64)
        xs_out = np.empty((nx, nt), dtype=np.float64)
        for jj in range(nt):
            xs_out[:, jj] = xs[jj]
            ts_out[jj] = ts[jj]

    # convert fs and Fs to arrays
    fs_out = np.empty((len(ts), nx), dtype=np.float64)
    Fs_out = np.empty((len(ts), coefs.INTERPOLATOR_POWER, nx), dtype=np.float64)
    for jj in range(len(ts)):
        fs_out[jj] = fs[jj]
    for jj in range(len(ts) - 1):
        Fs_out[jj] = Fs[jj]

    te_out = []
    xe_out = []
    for jj in range(len(events)):
        ne = len(t_events[jj])
        te = np.empty((ne,), np.float64)
        xe = np.empty((ne, nx), np.float64)
        for kk in range(ne):
            te[kk] = t_events[jj][kk]
            xe[kk] = x_events[jj][kk]
        te_out.append(te)
        xe_out.append(xe)

    return (ts_out, xs_out, (fs_out, Fs_out), (te_out, xe_out))
