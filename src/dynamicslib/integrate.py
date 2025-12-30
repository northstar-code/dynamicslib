from numba import njit
from numpy.typing import NDArray
import numpy as np
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


def dop853(
    func: Callable,
    t_span: Tuple[float, float],
    x0: NDArray,
    atol: float = 1e-10,
    rtol: float = 1e-10,
    init_step: float = 1.0,
    args: Tuple = (),
    t_eval: NDArray | None = None,
    dense_output: bool = False,
) -> Tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating] | None,
    NDArray[np.floating] | None,
]:
    """High order adaptive RK method

    Args:
        func (Callable): dynamics function
        t_span (Tuple[float, float]): beginning and end times
        x0 (NDArray): initial state
        atol (float, optional): absolute tolerence. Defaults to 1e-10.
        rtol (float, optional): rel tolerence. Defaults to 1e-10.
        init_step (float, optional): initial step size. Defaults to 1.0.
        t_eval (NDArray | None, optional): times to evaluate at. Will interpolate if these are specified. If None, returns only what's evaluated by the RK solver. Defaults to None
        args (Tuple, optional): additional args to func(t, x, *args). Defaults to ().

    Returns:
        Tuple[NDArray, NDArray, NDArray, NDArray]: ts (N, ), xs (nx, N), EOM function evals (Nx, N), intermediate evaluations (Nt-1, 7, Nx)
    """

    if t_eval is not None:
        dense_output = True

    n = len(x0)

    K_ext = np.empty((coefs.N_STAGES_EXTENDED, n), dtype=np.float64)
    K = K_ext[: coefs.n_stages + 1]

    forward = t_span[1] > t_span[0]
    t0, tf = t_span
    t = t0
    xs = np.expand_dims(x0, axis=0)
    fs = np.expand_dims(func(t0, x0, *args), axis=0)
    if dense_output:
        Fs = np.zeros((0, coefs.INTERPOLATOR_POWER, n), np.float64)
    else:
        Fs = None

    ts = np.array([t0], dtype=np.float64)
    x = x0.copy()
    h = abs(init_step) if forward else -abs(init_step)

    # pp180 of RKEM
    while (t < tf) if forward else (t > tf):
        if (t + h > tf) if forward else (t + h < tf):
            h = tf - t

        # STEP
        K[0] = func(t, x, *args)
        for sm1 in range(coefs.N_STAGES - 1):
            s = sm1 + 1
            a = coefs.A[s]
            c = coefs.C[s]
            dy = np.dot(K[:s].T, a[:s]) * h
            K[s] = func(t + c * h, x + dy, *args)

        xnew = x + h * np.dot(K[:-1].T, coefs.B)

        K[-1] = func(t + h, xnew, *args)

        # END STEP

        # error estimator:
        scale = atol + np.maximum(np.abs(x), np.abs(xnew)) * rtol
        err5 = np.dot(K.T, coefs.E5) / scale
        err3 = np.dot(K.T, coefs.E3) / scale
        err5_norm_2 = np.linalg.norm(err5) ** 2
        err3_norm_2 = np.linalg.norm(err3) ** 2
        denom = err5_norm_2 + 0.01 * err3_norm_2
        error = np.abs(h) * err5_norm_2 / np.sqrt(denom * len(scale))
        # END ERROR ESTIMDATOR

        hscale = 0.9 * error ** (-1 / 8) if error != 0 else 2

        # When the step is accepted
        if error <= 1:
            if dense_output:
                F = dop853_dense_extra(func, h, t, xnew, x, K[-1], K_ext, n, args)
                Fs = np.concatenate((Fs, np.expand_dims(F, axis=0)))
            t += h
            x = xnew
            ts = np.concatenate((ts, np.array([t], dtype=np.float64)))
            xs = np.concatenate((xs, np.expand_dims(x, axis=0)))
            fs = np.concatenate((fs, np.expand_dims(K[-1], axis=0)))

        h *= hscale

    if t_eval is not None:
        _, xs = dop_interpolate(ts, xs, Fs, t_eval)
        ts = t_eval
        return ts, xs.T, None, None

    return ts, xs.T, fs.T, Fs


def dop853_events_support(
    func: Callable[..., NDArray],
    t_span: Tuple[float, float],
    x0: NDArray,
    atol: float = 1e-10,
    rtol: float = 1e-10,
    init_step: float = 1.0,
    args: Tuple = (),
    t_eval: NDArray | None = None,
    dense_output: bool = False,
    events: List[Callable[..., float]] = [],
) -> Tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    Tuple[NDArray[np.floating] | None, NDArray[np.floating] | None],
    Tuple[NDArray[np.floating] | None, NDArray[np.floating] | None],
]:
    """High order adaptive RK method

    Args:
        func (Callable): dynamics function
        t_span (Tuple[float, float]): beginning and end times
        x0 (NDArray): initial state
        atol (float, optional): absolute tolerence. Defaults to 1e-10.
        rtol (float, optional): rel tolerence. Defaults to 1e-10.
        init_step (float, optional): initial step size. Defaults to 1.0.
        t_eval (NDArray | None, optional): times to evaluate at. Will interpolate if these are specified. If None, returns only what's evaluated by the RK solver. Defaults to None
        args (Tuple, optional): additional args to func(t, x, *args). Defaults to ().

    Returns:
        NDArray: ts (N, )
        NDArray: xs (nx, N)
        Tuple: [NDArray, NDArray]: (EOM function evals (Nx, N), intermediate evaluations (Nt-1, 7, Nx))
        Tuple: [NDArray, NDArray]: (Event times [Ne x (Nevents, )], Event states [Ne x (nx, Nevents)])
    """

    nx = len(x0)

    # Prepare events
    # useev = len(events) > 0
    for function in events:
        if not hasattr(function, "terminal"):
            function.terminal = 0
        elif function.terminal < 0:
            function.terminal = 0
        if not hasattr(function, "direction"):
            function.direction = 0
    if len(events):
        dense_output = True

    if t_eval is not None:
        dense_output = True

    # Prepare interpolator
    K_ext = np.empty((coefs.N_STAGES_EXTENDED, nx), dtype=np.float64)
    K = K_ext[: coefs.n_stages + 1]

    forward = t_span[1] > t_span[0]
    t0, tf = t_span
    t = t0
    xs = np.expand_dims(x0, axis=0)
    fs = np.expand_dims(func(t0, x0, *args), axis=0)
    if dense_output:
        Fs = np.zeros((0, coefs.INTERPOLATOR_POWER, nx), np.float64)
    else:
        Fs = None

    t_events = [np.empty((0,), dtype=np.float64) for _ in events]
    event_vals = [np.array([g(t0, x0, *args)]) for g in events]
    x_events = [np.empty((0, nx), dtype=np.float64) for _ in events]

    ts = np.array([t0], dtype=np.float64)
    x = x0.copy()
    h = abs(init_step) if forward else -abs(init_step)

    while (t < tf) if forward else (t > tf):
        if (t + h > tf) if forward else (t + h < tf):
            h = tf - t

        # STEP, syntax taken from Scipy implementation
        K[0] = func(t, x, *args)
        for sm1 in range(coefs.N_STAGES - 1):
            s = sm1 + 1
            a = coefs.A[s]
            c = coefs.C[s]
            dy = np.dot(K[:s].T, a[:s]) * h
            K[s] = func(t + c * h, x + dy, *args)

        xnew = x + h * np.dot(K[:-1].T, coefs.B)

        K[-1] = func(t + h, xnew, *args)

        # END STEP

        # error estimator:
        scale = atol + np.maximum(np.abs(x), np.abs(xnew)) * rtol
        err5 = np.dot(K.T, coefs.E5) / scale
        err3 = np.dot(K.T, coefs.E3) / scale
        err5_norm_2 = np.linalg.norm(err5) ** 2
        err3_norm_2 = np.linalg.norm(err3) ** 2
        denom = err5_norm_2 + 0.01 * err3_norm_2
        error = np.abs(h) * err5_norm_2 / np.sqrt(denom * len(scale))
        # END ERROR ESTIMDATOR

        hscale = 0.9 * error ** (-1 / 8) if error != 0 else 2

        # When the step is accepted
        if error <= 1:
            if dense_output:
                F = dop853_dense_extra(func, h, t, xnew, x, K[-1], K_ext, nx, args)
                Fs = np.concatenate((Fs, np.expand_dims(F, axis=0)))

            # Event handling
            halt = False
            for jj, event in enumerate(events):
                g = event(t + h, xnew, *args)
                event_vals[jj] = np.concatenate(
                    (event_vals[jj], np.array([g], dtype=np.float64))
                )
                ev_vals = event_vals[jj]
                # handle event direction here
                if (
                    event.direction in [0, 1]
                    and np.sign(ev_vals[-2]) < 0
                    and np.sign(ev_vals[-1]) >= 0
                ) or (
                    event.direction in [-1, 0]
                    and np.sign(ev_vals[-2]) > 0
                    and np.sign(ev_vals[-1]) <= 0
                ):
                    te, xe = interpolate_event(
                        x, xnew, t, t + h, F, ev_vals[-2], ev_vals[-1], event, args
                    )
                    t_events[jj] = np.concatenate(
                        (t_events[jj], np.array([te], dtype=np.float64))
                    )
                    x_events[jj] = np.concatenate(
                        (x_events[jj], np.expand_dims(xe, axis=0))
                    )

                if len(t_events[jj]) == event.terminal and event.terminal > 0:
                    halt = True

            t += h
            x = xnew
            ts = np.concatenate((ts, np.array([t], dtype=np.float64)))
            xs = np.concatenate((xs, np.expand_dims(x, axis=0)))
            fs = np.concatenate((fs, np.expand_dims(K[-1], axis=0)))
            if halt:
                break

        h *= hscale

    x_events = [xe.T for xe in x_events]

    if t_eval is not None:
        _, xs = dop_interpolate(ts, xs, Fs, t_eval)
        ts = t_eval
        return ts, xs.T, (None, None), (t_events, x_events)

    return ts, xs.T, (fs.T, Fs), (t_events, x_events)
