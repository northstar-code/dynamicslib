import threading
import queue
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple, List, Any
from numba import njit, prange
from numba import types
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from dynamicslib.consts import muEM
from dynamicslib.integrator import dop853, interp_hermite


# %% generic CR3BP stuff
@njit(cache=True)
def get_L1(mu=muEM, tol=1e-14):
    # find x_L1
    x = 1 - 2 * mu
    f = np.inf
    while abs(f) > tol:
        f = -(1 - mu) / (x + mu) ** 2 + mu / (x - 1 + mu) ** 2 + x
        df = 2 * (1 - mu) / (x + mu) ** 3 - 2 * mu / (x - 1 + mu) ** 3 + 1
        dx = -f / df
        x += dx
    return x


@njit(cache=True)
def get_L2(mu=muEM, tol=1e-14):
    # find x_L2
    x = 1 + mu
    f = np.inf
    while abs(f) > tol:
        f = -(1 - mu) / (x + mu) ** 2 - mu / (x - 1 + mu) ** 2 + x
        df = 2 * (1 - mu) / (x + mu) ** 3 + 2 * mu / (x - 1 + mu) ** 3 + 1
        dx = -f / df
        x += dx
    return x


@njit(cache=True)
def get_L3(mu=muEM, tol=1e-14):
    # find x_L3
    x = -1 - mu
    f = np.inf
    while abs(f) > tol:
        f = (1 - mu) / (x + mu) ** 2 + mu / (x - 1 + mu) ** 2 + x
        df = -2 * (1 - mu) / (x + mu) ** 3 - 2 * mu / (x - 1 + mu) ** 3 + 1
        dx = -f / df
        x += dx
    return x


def get_Lpts(mu: float = muEM):
    lagrange_points = np.array(
        [
            [get_L1(mu), get_L2(mu), get_L3(mu), 1 / 2 - mu, 1 / 2 - mu],
            [0, 0, 0, -np.sqrt(3) / 2, np.sqrt(3) / 2],
        ],
    )
    return lagrange_points


@njit(cache=True)
def U_hess(pos: NDArray[np.floating], mu: float = muEM) -> NDArray[np.floating]:
    r1 = pos - np.array([-mu, 0, 0])
    r2 = pos - np.array([1 - mu, 0, 0])
    r1mag = np.linalg.norm(r1)
    r2mag = np.linalg.norm(r2)
    Uxx = (
        np.diag(np.array([1, 1, 0]))
        + 3 * (1 - mu) / r1mag**5 * np.outer(r1, r1)
        - (1 - mu) / r1mag**3 * np.eye(3)
        + 3 * mu / r2mag**5 * np.outer(r2, r2)
        - mu / r2mag**3 * np.eye(3)
    )

    return Uxx


@njit(cache=True)
def get_A(state: NDArray[np.floating], mu: float = muEM) -> NDArray[np.floating]:
    pos = state[:3]
    Uxx = U_hess(pos, mu)
    O = np.zeros((3, 3))
    I = np.eye(3)
    Omega = np.array([[0, 2, 0], [-2, 0, 0], [0, 0, 0]])
    A1 = np.concatenate((O, I), axis=1)
    A2 = np.concatenate((Uxx, Omega), axis=1)
    A = np.concatenate((A1, A2), axis=0)
    return A


@njit(cache=True)
def eom(_, state: NDArray[np.floating], mu: float = muEM) -> NDArray[np.floating]:
    x, y, z, vx, vy, vz = state[:6]
    xyz = state[:3]
    r1vec = xyz + np.array([mu, 0, 0])
    r2vec = xyz + np.array([mu - 1, 0, 0])
    r1mag = np.linalg.norm(r1vec)
    r2mag = np.linalg.norm(r2vec)

    ddxyz = (
        -(1 - mu) * r1vec / r1mag**3
        - mu * r2vec / r2mag**3
        + np.array([2 * vy + x, -2 * vx + y, 0])
    )

    dstate = np.zeros(6)
    dstate[:3] = state[3:]
    dstate[3:] = ddxyz
    return dstate


@njit(cache=True)
def coupled_stm_eom(
    _, state: NDArray[np.floating], mu: float = muEM
) -> NDArray[np.floating]:
    pv = state[:6]
    dpv = eom(None, pv, mu)
    stm = state[6:].reshape((6, 6))
    A = get_A(pv, mu)  # pv[:3]
    dstm = A @ stm

    dstate = np.array([*dpv, *dstm.flatten()])
    return dstate


# DEPRICATED: only useful for scipy integrators that use jacobians
# @njit(cache=True)
# def coupled_stm_eom_jac(
#     _, state: NDArray[np.floating], mu: float = muEM
# ) -> NDArray[np.floating]:
#     # fmt: off
#     x, y, z, vx, vy, vz, Phi11, Phi12, Phi13, Phi14, Phi15, Phi16, Phi21, Phi22, Phi23, Phi24, Phi25, Phi26, Phi31, Phi32, Phi33, Phi34, Phi35, Phi36, Phi41, Phi42, Phi43, Phi44, Phi45, Phi46, Phi51, Phi52, Phi53, Phi54, Phi55, Phi56, Phi61, Phi62, Phi63, Phi64, Phi65, Phi66 = state
#     x1 = x+mu
#     x2 = x+mu-1
#     r1 = np.sqrt(x1**2+y**2+z**2)
#     r2 = np.sqrt(x2**2+y**2+z**2)
#     out = np.array([x, y, z, -mu*x2/r2**3 + 2*vy + x + x1*(mu - 1)/r1**3, -mu*y/r2**3 - 2*vx + y + y*(mu - 1)/r1**3, -mu*z/r2**3 + z*(mu - 1)/r1**3, Phi41, Phi42, Phi43, Phi44, Phi45, Phi46, Phi51, Phi52, Phi53, Phi54, Phi55, Phi56, Phi61, Phi62, Phi63, Phi64, Phi65, Phi66, Phi11*(-mu/r2**3 + 3*mu*x2**2/r2**5 + 1 - (1 - mu)/r1**3 + x1**2*(3 - 3*mu)/r1**5) + Phi21*(3*mu*x2*y/r2**5 + x1*y*(3 - 3*mu)/r1**5) + Phi31*(3*mu*x2*z/r2**5 + x1*z*(3 - 3*mu)/r1**5) + 2*Phi51, Phi12*(-mu/r2**3 + 3*mu*x2**2/r2**5 + 1 - (1 - mu)/r1**3 + x1**2*(3 - 3*mu)/r1**5) + Phi22*(3*mu*x2*y/r2**5 + x1*y*(3 - 3*mu)/r1**5) + Phi32*(3*mu*x2*z/r2**5 + x1*z*(3 - 3*mu)/r1**5) + 2*Phi52, Phi13*(-mu/r2**3 + 3*mu*x2**2/r2**5 + 1 - (1 - mu)/r1**3 + x1**2*(3 - 3*mu)/r1**5) + Phi23*(3*mu*x2*y/r2**5 + x1*y*(3 - 3*mu)/r1**5) + Phi33*(3*mu*x2*z/r2**5 + x1*z*(3 - 3*mu)/r1**5) + 2*Phi53, Phi14*(-mu/r2**3 + 3*mu*x2**2/r2**5 + 1 - (1 - mu)/r1**3 + x1**2*(3 - 3*mu)/r1**5) + Phi24*(3*mu*x2*y/r2**5 + x1*y*(3 - 3*mu)/r1**5) + Phi34*(3*mu*x2*z/r2**5 + x1*z*(3 - 3*mu)/r1**5) + 2*Phi54, Phi15*(-mu/r2**3 + 3*mu*x2**2/r2**5 + 1 - (1 - mu)/r1**3 + x1**2*(3 - 3*mu)/r1**5) + Phi25*(3*mu*x2*y/r2**5 + x1*y*(3 - 3*mu)/r1**5) + Phi35*(3*mu*x2*z/r2**5 + x1*z*(3 - 3*mu)/r1**5) + 2*Phi55, Phi16*(-mu/r2**3 + 3*mu*x2**2/r2**5 + 1 - (1 - mu)/r1**3 + x1**2*(3 - 3*mu)/r1**5) + Phi26*(3*mu*x2*y/r2**5 + x1*y*(3 - 3*mu)/r1**5) + Phi36*(3*mu*x2*z/r2**5 + x1*z*(3 - 3*mu)/r1**5) + 2*Phi56, Phi11*(3*mu*x2*y/r2**5 + x1*y*(3 - 3*mu)/r1**5) + Phi21*(-mu/r2**3 + 3*mu*y**2/r2**5 + 1 - (1 - mu)/r1**3 + y**2*(3 - 3*mu)/r1**5) + Phi31*(3*mu*y*z/r2**5 + y*z*(3 - 3*mu)/r1**5) - 2*Phi41, Phi12*(3*mu*x2*y/r2**5 + x1*y*(3 - 3*mu)/r1**5) + Phi22*(-mu/r2**3 + 3*mu*y**2/r2**5 + 1 - (1 - mu)/r1**3 + y**2*(3 - 3*mu)/r1**5) + Phi32*(3*mu*y*z/r2**5 + y*z*(3 - 3*mu)/r1**5) - 2*Phi42, Phi13*(3*mu*x2*y/r2**5 + x1*y*(3 - 3*mu)/r1**5) + Phi23*(-mu/r2**3 + 3*mu*y**2/r2**5 + 1 - (1 - mu)/r1**3 + y**2*(3 - 3*mu)/r1**5) + Phi33*(3*mu*y*z/r2**5 + y*z*(3 - 3*mu)/r1**5) - 2*Phi43, Phi14*(3*mu*x2*y/r2**5 + x1*y*(3 - 3*mu)/r1**5) + Phi24*(-mu/r2**3 + 3*mu*y**2/r2**5 + 1 - (1 - mu)/r1**3 + y**2*(3 - 3*mu)/r1**5) + Phi34*(3*mu*y*z/r2**5 + y*z*(3 - 3*mu)/r1**5) - 2*Phi44, Phi15*(3*mu*x2*y/r2**5 + x1*y*(3 - 3*mu)/r1**5) + Phi25*(-mu/r2**3 + 3*mu*y**2/r2**5 + 1 - (1 - mu)/r1**3 + y**2*(3 - 3*mu)/r1**5) + Phi35*(3*mu*y*z/r2**5 + y*z*(3 - 3*mu)/r1**5) - 2*Phi45, Phi16*(3*mu*x2*y/r2**5 + x1*y*(3 - 3*mu)/r1**5) + Phi26*(-mu/r2**3 + 3*mu*y**2/r2**5 + 1 - (1 - mu)/r1**3 + y**2*(3 - 3*mu)/r1**5) + Phi36*(3*mu*y*z/r2**5 + y*z*(3 - 3*mu)/r1**5) - 2*Phi46, Phi11*(3*mu*x2*z/r2**5 + x1*z*(3 - 3*mu)/r1**5) + Phi21*(3*mu*y*z/r2**5 + y*z*(3 - 3*mu)/r1**5) + Phi31*(-mu/r2**3 + 3*mu*z**2/r2**5 - (1 - mu)/r1**3 + z**2*(3 - 3*mu)/r1**5), Phi12*(3*mu*x2*z/r2**5 + x1*z*(3 - 3*mu)/r1**5) + Phi22*(3*mu*y*z/r2**5 + y*z*(3 - 3*mu)/r1**5) + Phi32*(-mu/r2**3 + 3*mu*z**2/r2**5 - (1 - mu)/r1**3 + z**2*(3 - 3*mu)/r1**5), Phi13*(3*mu*x2*z/r2**5 + x1*z*(3 - 3*mu)/r1**5) + Phi23*(3*mu*y*z/r2**5 + y*z*(3 - 3*mu)/r1**5) + Phi33*(-mu/r2**3 + 3*mu*z**2/r2**5 - (1 - mu)/r1**3 + z**2*(3 - 3*mu)/r1**5), Phi14*(3*mu*x2*z/r2**5 + x1*z*(3 - 3*mu)/r1**5) + Phi24*(3*mu*y*z/r2**5 + y*z*(3 - 3*mu)/r1**5) + Phi34*(-mu/r2**3 + 3*mu*z**2/r2**5 - (1 - mu)/r1**3 + z**2*(3 - 3*mu)/r1**5), Phi15*(3*mu*x2*z/r2**5 + x1*z*(3 - 3*mu)/r1**5) + Phi25*(3*mu*y*z/r2**5 + y*z*(3 - 3*mu)/r1**5) + Phi35*(-mu/r2**3 + 3*mu*z**2/r2**5 - (1 - mu)/r1**3 + z**2*(3 - 3*mu)/r1**5), Phi16*(3*mu*x2*z/r2**5 + x1*z*(3 - 3*mu)/r1**5) + Phi26*(3*mu*y*z/r2**5 + y*z*(3 - 3*mu)/r1**5) + Phi36*(-mu/r2**3 + 3*mu*z**2/r2**5 - (1 - mu)/r1**3 + z**2*(3 - 3*mu)/r1**5)])
#     # fmt: on
#     return out


@njit(cache=True)
def jacobi_constant(state: NDArray[np.floating], mu: float = muEM) -> float:
    x, y, z = state[:3]
    r1mag = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2mag = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)
    Ugrav = (1 - mu) / r1mag + mu / r2mag
    Ucent = (x**2 + y**2) / 2
    U = Ucent + Ugrav
    JC = 2 * U - np.dot(state[3:], state[3:])
    return JC


@njit(cache=True)
def JCgrad(state: NDArray, mu: float = muEM) -> NDArray[np.floating]:
    x, y, z = state[:3]
    d1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    d2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)
    x2 = x - 1 + mu
    x1 = x + mu
    r1 = np.array([x1, y, z])
    r2 = np.array([x2, y, z])
    return -2 * mu / d2**3 * r2 - 2 * (1 - mu) / d1**3 * r1 + 2 * x


# DEPRICATED: not really useful
# def get_stab(eigval: float, eps: float = 1e-5) -> int:
#     """Get stability modes of a single eigenvalue. Numeric codes are
#     ```
#     0: parabolic
#     1: elliptic
#     2: +hyperbolic
#     3: -hyperbolic
#     4: quadrouple
#     ```

#     Args:
#         eigval (float): eigenvalue
#         eps (float, optional): epsilon for ==1. Defaults to 1e-5.

#     Returns:
#         int: the stability type.
#     """
#     if 1 - eps <= np.abs(eigval) <= eps:
#         if np.abs(np.imag(eigval)) < eps:
#             return 0
#         else:
#             return 1
#     elif np.abs(np.imag(eigval)) < eps:
#         if np.real(eigval) > 0:
#             return 2
#         else:
#             return 3
#     else:
#         return 4


# shortcut to get x,y,z from X
def prop_ic(
    X: NDArray,
    X2xtf_func: Callable,
    mu: float = muEM,
    int_tol=1e-12,
    density_mult: int = 2,
):
    x0, tf = X2xtf_func(X)
    ts, xs, fs = dop853(eom, (0, tf), x0, rtol=int_tol, atol=int_tol, args=(mu,))
    ts, dense_sol = interp_hermite(ts, xs.T, fs.T, n_mult=density_mult)
    x, y, z = dense_sol.T[:3]
    return x, y, z


def prop_ic_fullstate(
    X: NDArray,
    X2xtf_func: Callable,
    mu: float = muEM,
    int_tol=1e-12,
    density_mult: int = 2,
):
    x0, tf = X2xtf_func(X)
    ts, xs, fs = dop853(eom, (0, tf), x0, rtol=int_tol, atol=int_tol, args=(mu,))
    ts, dense_sol = interp_hermite(ts, xs.T, fs.T, n_mult=density_mult)
    return dense_sol.T


def manifold_stepoffs(
    x0: NDArray,
    period: float,
    N: int = 25,
    s: float = 1e-6,
    mu: float = muEM,
    int_tol=1e-12,
) -> Tuple[
    Tuple[NDArray[np.floating], ...],
    Tuple[NDArray[np.floating], ...],
    Tuple[
        Tuple[NDArray[np.floating], ...],
        Tuple[NDArray[np.floating], ...],
        Tuple[NDArray[np.floating], ...],
        Tuple[NDArray[np.floating], ...],
    ],
]:
    """Get manifold start points. Returns 4N points (N of each stable half
    and another N of each unstable half). Return order is (s+ s-), (u+ u-)

    Args:
        x0 (NDArray): Nominal initial condition
        period (float): Period of the orbot
        N (int, optional): Number of stepoff points, evenly spaced in time. Defaults to 25.
        s (float, optional): Stepoff distance. Defaults to 1e-6.
        mu (float, optional): Gravitational parameter. Defaults to muEM.
        int_tol (_type_, optional): Integration tolerance. Defaults to 1e-12.

    Returns:
        Tuple, Tuple: Manifold ICs. Can be propagated elsewhere
    """
    sv0 = np.append(x0, np.eye(6).flatten())
    te = np.linspace(0, period, N + 1)
    ode_out = solve_ivp(
        coupled_stm_eom,
        (0.0, period),
        sv0,
        atol=int_tol,
        rtol=int_tol,
        t_eval=te,
        args=(mu,),
    )
    svs = ode_out.y.T[:-1]
    xs = [sv[:6] for sv in svs]
    mono = ode_out.y.T[-1, 6:].reshape(6, 6)
    stms = [sv[6:].reshape(6, 6) for sv in svs]
    monodromies = [stm @ mono @ np.linalg.inv(stm) for stm in stms]
    eigs = [np.linalg.eig(phi) for phi in monodromies]
    # stable eigenvectors
    vecs_s = [e.eigenvectors[:, np.argmin(np.abs(e.eigenvalues))].real for e in eigs]
    # unstable eigenvectors
    vecs_u = [e.eigenvectors[:, np.argmax(np.abs(e.eigenvalues))].real for e in eigs]

    # Find ICs
    # stable halves
    x0s_s1 = tuple([x + vec * s for x, vec in zip(xs, vecs_s)])
    x0s_s2 = tuple([x - vec * s for x, vec in zip(xs, vecs_s)])
    # unstable halves
    x0s_u1 = tuple([x + vec * s for x, vec in zip(xs, vecs_u)])
    x0s_u2 = tuple([x - vec * s for x, vec in zip(xs, vecs_u)])

    aux = (tuple(xs), tuple(monodromies), tuple(vecs_u), tuple(vecs_s))
    return x0s_u1 + x0s_u2, x0s_s1 + x0s_s2, aux


def integrate_one(
    lock: threading.Lock,
    dict_out: dict,
    index: int,
    tf: float,
    x0: NDArray,
    events: Callable | List,
    mu: float,
    int_tol: float,
):
    """Propagate multiple curves to a common set of event functions. Uses scipy propagator, so may not be great.

    Args:
        lock (threading.Lock): Thread lock to prevent simultaneous writes to dict
        dictr_out (dict): dictionary to fill with outputs
        index (int): Index to place result in dict
        tf (float): Max integration time, in case event doesnt trigger. Also holds information about integration direction.
        x0 (NDArray): Initial condition
        events (Callable | List): Event function or event functions list. Must have signature (t, x, mu) -> float
        mu (float, optional): Mass parameter. Defaults to muEM.
        int_tol (float, optional): Integration tolerance. Defaults to 1e-10.

    Returns:
        dict[int, OdeResult]: Indexed results with ODE output
    """
    try:
        iter(events)
        pass
    except TypeError:
        events = [events]

    ode_out = solve_ivp(
        eom,
        (0.0, tf),
        x0,
        "DOP853",
        atol=int_tol,
        rtol=int_tol,
        args=(mu,),
        events=events,
    )
    with lock:
        dict_out[index] = ode_out


def prop_multiple(
    x0s: NDArray | List,
    events: Callable | List,
    tfmax: float,
    mu: float = muEM,
    int_tol: float = 1e-10,
) -> dict[int, OdeResult]:
    """Propagate multiple curves to a common set of event functions. Uses scipy propagator, so may not be great.

    Args:
        x0s (NDArray | List): List of initial conditions, ordered. Nx6
        events (Callable | List): Event function or event functions list. Must have signature (t, x, mu) -> float
        tfmax (float): Maximum final time, in case event never triggers
        mu (float, optional): Mass parameter. Defaults to muEM.
        int_tol (float, optional): Integration tolerance. Defaults to 1e-10.

    Returns:
        dict[int, OdeResult]: Indexed results with ODE output
    """
    dct_out = {}
    N = len(x0s)
    lock = threading.Lock()

    threads = []
    for ind in range(N):
        x0 = x0s[ind]
        args = (lock, dct_out, ind, tfmax, x0, events, mu, int_tol)
        thread = threading.Thread(target=integrate_one, args=args)
        threads.append(thread)

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return dct_out


# shortcut to get JC and tf from X
def get_JC_tf(X: NDArray, X2xtf_func: Callable, mu: float = muEM):
    x0, tf = X2xtf_func(X)
    jc = jacobi_constant(x0, mu)

    return jc, tf


# basic call
def f_df_CR3_single(
    X: NDArray,
    X2xtf: Callable,
    dF_func: Callable,
    f_func: Callable,
    full_period=False,
    mu: float = muEM,
    int_tol: float = 1e-10,
) -> Tuple[NDArray, NDArray, NDArray]:
    x0, tf = X2xtf(X)
    xstmIC = np.array([*x0, *np.eye(6).flatten()])
    ts, ys, _ = dop853(
        coupled_stm_eom,
        (0.0, tf if full_period else tf / 2),
        xstmIC,
        int_tol,
        int_tol,
        init_step=0.05,
        args=(mu,),
    )
    xf, stm = ys[:6, -1], ys[6:, -1].reshape(6, 6)
    xf = np.array(xf)
    eomf = eom(0, xf, mu)

    dF = dF_func(eomf, stm)
    f = f_func(x0, tf, xf)

    if not full_period:
        G = np.diag([1, -1, 1, -1, 1, -1])
        Omega = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        I = np.identity(3)
        O = np.zeros((3, 3))
        mtx1 = np.block([[O, -I], [I, -2 * Omega]])
        mtx2 = np.block([[-2 * Omega, I], [-I, O]])
        stm_full = G @ mtx1 @ stm.T @ mtx2 @ G @ stm
    else:
        stm_full = stm
    return f, dF, stm_full
