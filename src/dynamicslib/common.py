import threading
import queue
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple, List, Any
from numba import njit, prange
from numba import types
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from dynamicslib.consts import muEM, RMoon, REarth
from dynamicslib.integrate import dop853
from dynamicslib.interpolate import dop_interpolate


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


# %% spatial

# more readable but slower versions:
# @njit(cache=True)
# def U_hess(pos: NDArray[np.floating], mu: float = muEM) -> NDArray[np.floating]:
#     r1 = pos - np.array([-mu, 0, 0])
#     r2 = pos - np.array([1 - mu, 0, 0])
#     r1mag = np.linalg.norm(r1)
#     r2mag = np.linalg.norm(r2)
#     Uxx = (
#         np.diag(np.array([1, 1, 0]))
#         + 3 * (1 - mu) / r1mag**5 * np.outer(r1, r1)
#         - (1 - mu) / r1mag**3 * np.eye(3)
#         + 3 * mu / r2mag**5 * np.outer(r2, r2)
#         - mu / r2mag**3 * np.eye(3)
#     )

#     return Uxx

# @njit(cache=True)
# def get_A(state: NDArray[np.floating], mu: float = muEM) -> NDArray[np.floating]:
#     pos = state[:3]
#     Uxx = U_hess(pos, mu)
#     O = np.zeros((3, 3))
#     I = np.eye(3)
#     Omega = np.array([[0, 2, 0], [-2, 0, 0], [0, 0, 0]])
#     A1 = np.concatenate((O, I), axis=1)
#     A2 = np.concatenate((Uxx, Omega), axis=1)
#     A = np.concatenate((A1, A2), axis=0)
#     return A

# @njit(cache=True)
# def eom(_, state: NDArray[np.floating], mu: float = muEM) -> NDArray[np.floating]:
#     x, y, z, vx, vy, vz = state[:6]
#     xyz = state[:3]
#     r1vec = xyz + np.array([mu, 0, 0])
#     r2vec = xyz + np.array([mu - 1, 0, 0])
#     r1mag = np.linalg.norm(r1vec)
#     r2mag = np.linalg.norm(r2vec)

#     ddxyz = (
#         -(1 - mu) * r1vec / r1mag**3
#         - mu * r2vec / r2mag**3
#         + np.array([2 * vy + x, -2 * vx + y, 0])
#     )

#     dstate = np.zeros(6)
#     dstate[:3] = state[3:]
#     dstate[3:] = ddxyz
#     return dstate

# @njit(cache=True)
# def coupled_stm_eom(
#     _, state: NDArray[np.floating], mu: float = muEM
# ) -> NDArray[np.floating]:
#     pv = state[:6]
#     dpv = eom(0.0, pv, mu)
#     stm = state[6:].reshape((6, 6))
#     A = get_A(pv, mu)  # pv[:3]
#     dstm = A @ stm

#     dstate = np.array([*dpv, *dstm.flatten()])
#     return dstate


@njit(cache=True)
def U_hess(pos: NDArray[np.floating], mu: float = muEM) -> NDArray[np.floating]:
    x, y, z = pos[0], pos[1], pos[2]

    x1 = x + mu
    x2 = x - 1.0 + mu

    r1sq = x1 * x1 + y * y + z * z
    r2sq = x2 * x2 + y * y + z * z

    r1 = np.sqrt(r1sq)
    r2 = np.sqrt(r2sq)

    r1_3 = r1sq * r1
    r1_5 = r1_3 * r1sq
    r2_3 = r2sq * r2
    r2_5 = r2_3 * r2sq

    c1 = 1.0 - mu
    c2 = mu

    # Precompute factors
    a1 = 3.0 * c1 / r1_5
    b1 = c1 / r1_3
    a2 = 3.0 * c2 / r2_5
    b2 = c2 / r2_3

    H = np.empty((3, 3))

    # Diagonal
    H[0, 0] = 1.0 + a1 * x1 * x1 - b1 + a2 * x2 * x2 - b2
    H[1, 1] = 1.0 + a1 * y * y - b1 + a2 * y * y - b2
    H[2, 2] = a1 * z * z - b1 + a2 * z * z - b2

    # Off-diagonal (symmetric)
    H[0, 1] = a1 * x1 * y + a2 * x2 * y
    H[1, 0] = H[0, 1]

    H[0, 2] = a1 * x1 * z + a2 * x2 * z
    H[2, 0] = H[0, 2]

    H[1, 2] = a1 * y * z + a2 * y * z
    H[2, 1] = H[1, 2]

    return H


@njit(cache=True)
def get_A(state: NDArray[np.floating], mu: float = muEM) -> NDArray[np.floating]:
    x, y, z = state[0], state[1], state[2]

    Uxx = U_hess(state, mu)  # or pass pos explicitly

    A = np.zeros((6, 6))

    # identity block
    A[0, 3] = 1.0
    A[1, 4] = 1.0
    A[2, 5] = 1.0

    # Hessian
    for i in range(3):
        for j in range(3):
            A[i + 3, j] = Uxx[i, j]

    # cross block
    A[3, 4] = 2.0
    A[4, 3] = -2.0

    return A


@njit(cache=True)
def eom(_, state: NDArray[np.floating], mu: float = muEM) -> NDArray[np.floating]:
    x, y, z = state[0], state[1], state[2]
    vx, vy, vz = state[3], state[4], state[5]

    x1 = x + mu
    x2 = x - 1.0 + mu

    r1sq = x1 * x1 + y * y + z * z
    r2sq = x2 * x2 + y * y + z * z

    inv_r1 = 1.0 / np.sqrt(r1sq)
    inv_r2 = 1.0 / np.sqrt(r2sq)

    inv_r1_3 = inv_r1 / r1sq
    inv_r2_3 = inv_r2 / r2sq

    c1 = 1.0 - mu
    c2 = mu

    # Allocate output once
    out = np.empty(6)

    # velocity part
    out[0] = vx
    out[1] = vy
    out[2] = vz

    # acceleration (fully expanded)
    out[3] = -c1 * x1 * inv_r1_3 - c2 * x2 * inv_r2_3 + 2.0 * vy + x
    out[4] = -c1 * y * inv_r1_3 - c2 * y * inv_r2_3 - 2.0 * vx + y
    out[5] = -c1 * z * inv_r1_3 - c2 * z * inv_r2_3

    return out


@njit(cache=True)
def coupled_stm_eom(
    _, state: NDArray[np.floating], mu: float = muEM
) -> NDArray[np.floating]:
    out = np.zeros(42)
    x, y, z = state[0], state[1], state[2]
    vx, vy, vz = state[3], state[4], state[5]

    c1 = 1.0 - mu
    c2 = mu

    x1 = x + mu
    x2 = x - 1 + mu

    r1sq = x1 * x1 + y * y + z * z
    r2sq = x2 * x2 + y * y + z * z

    inv_r1 = 1.0 / np.sqrt(r1sq)
    inv_r2 = 1.0 / np.sqrt(r2sq)

    inv_r1_3 = inv_r1 / r1sq
    inv_r2_3 = inv_r2 / r2sq

    inv_r1_5 = inv_r1_3 / r1sq
    inv_r2_5 = inv_r2_3 / r2sq

    # state dynamics
    out[0] = vx
    out[1] = vy
    out[2] = vz
    out[3] = -c1 * x1 * inv_r1_3 - c2 * x2 * inv_r2_3 + 2.0 * vy + x
    out[4] = -c1 * y * inv_r1_3 - c2 * y * inv_r2_3 - 2.0 * vx + y
    out[5] = -c1 * z * inv_r1_3 - c2 * z * inv_r2_3

    ## A matrix

    # Hessian
    a1 = 3.0 * c1 * inv_r1_5
    b1 = c1 * inv_r1_3
    a2 = 3.0 * c2 * inv_r2_5
    b2 = c2 * inv_r2_3

    H = np.empty((3, 3))

    # Diagonal
    H[0, 0] = 1.0 + a1 * x1 * x1 - b1 + a2 * x2 * x2 - b2
    H[1, 1] = 1.0 + a1 * y * y - b1 + a2 * y * y - b2
    H[2, 2] = a1 * z * z - b1 + a2 * z * z - b2

    # Off-diagonal (symmetric)
    H[0, 1] = a1 * x1 * y + a2 * x2 * y
    H[1, 0] = H[0, 1]

    H[0, 2] = a1 * x1 * z + a2 * x2 * z
    H[2, 0] = H[0, 2]

    H[1, 2] = a1 * y * z + a2 * y * z
    H[2, 1] = H[1, 2]

    # expanded matrix multiplication
    for i in range(3):
        ip3 = i + 3
        for j in range(6):
            # dense block (bottom left)
            s = 0.0
            for a in range(3):
                s += H[i, a] * state[6 + 6 * a + j]
            out[6 + 6 * ip3 + j] = s

            # identity block (top right)
            out[6 + 6 * i + j] += state[6 + 6 * ip3 + j]

    # cross block
    for j in range(6):
        out[6 + 6 * 3 + j] += 2 * state[6 + 6 * 4 + j]
        out[6 + 6 * 4 + j] += -2 * state[6 + 6 * 3 + j]

    return out


# %% planar


@njit(cache=True)
def U_hess_planar(pos: NDArray[np.floating], mu: float = muEM) -> NDArray[np.floating]:
    r1 = pos - np.array([-mu, 0])
    r2 = pos - np.array([1 - mu, 0])
    r1mag = np.linalg.norm(r1)
    r2mag = np.linalg.norm(r2)
    Uxx = (
        np.diag(np.array([1, 1]))
        + 3 * (1 - mu) / r1mag**5 * np.outer(r1, r1)
        - (1 - mu) / r1mag**3 * np.eye(2)
        + 3 * mu / r2mag**5 * np.outer(r2, r2)
        - mu / r2mag**3 * np.eye(2)
    )

    return Uxx


@njit(cache=True)
def get_A_planar(state: NDArray[np.floating], mu: float = muEM) -> NDArray[np.floating]:
    pos = state[:2]
    Uxx = U_hess_planar(pos, mu)
    O = np.zeros((2, 2))
    I = np.eye(2)
    Omega = np.array([[0, 2], [-2, 0]])
    A1 = np.concatenate((O, I), axis=1)
    A2 = np.concatenate((Uxx, Omega), axis=1)
    A = np.concatenate((A1, A2), axis=0)
    return A


@njit(cache=True)
def eom_planar(
    _, state: NDArray[np.floating], mu: float = muEM
) -> NDArray[np.floating]:
    x, y, vx, vy = state[:4]
    xyz = state[:2]
    r1vec = xyz + np.array([mu, 0])
    r2vec = xyz + np.array([mu - 1, 0])
    r1mag = np.linalg.norm(r1vec)
    r2mag = np.linalg.norm(r2vec)

    ddxyz = (
        -(1 - mu) * r1vec / r1mag**3
        - mu * r2vec / r2mag**3
        + np.array([2 * vy + x, -2 * vx + y])
    )

    dstate = np.append(state[2:], ddxyz)
    return dstate


@njit(cache=True)
def coupled_stm_eom_planar(
    _, state: NDArray[np.floating], mu: float = muEM
) -> NDArray[np.floating]:
    pv = state[:4]
    dpv = eom_planar(0.0, pv, mu)
    stm = state[4:].reshape((4, 4))
    A = get_A_planar(pv, mu)  # pv[:3]
    dstm = A @ stm

    dstate = np.array([*dpv, *dstm.flatten()])
    return dstate


# %% singularity-removed TODO


@njit(cache=True)
def U_hess_nosingular(
    pos: NDArray[np.floating], mu: float = muEM, R1: float = REarth, R2: float = RMoon
) -> NDArray[np.floating]:
    r1 = pos - np.array([-mu, 0, 0])
    r2 = pos - np.array([1 - mu, 0, 0])
    r1mag = np.linalg.norm(r1)
    r2mag = np.linalg.norm(r2)

    if r1mag < R1:
        Uxx = (
            np.diag(np.array([1, 1, 0]))
            + (1 - mu) / R1**2 / r1mag**3 * np.outer(r1, r1)
            - (1 - mu) / R1**2 / r1mag * np.eye(3)
            + 3 * mu / r2mag**5 * np.outer(r2, r2)
            - mu / r2mag**3 * np.eye(3)
        )
    elif r2mag < R2:
        Uxx = (
            np.diag(np.array([1, 1, 0]))
            + 3 * (1 - mu) / r1mag**5 * np.outer(r1, r1)
            - (1 - mu) / r1mag**3 * np.eye(3)
            + mu / R2**2 / r2mag**3 * np.outer(r2, r2)
            - mu / R2**2 / r2mag * np.eye(3)
        )

    else:

        Uxx = (
            np.diag(np.array([1, 1, 0]))
            + 3 * (1 - mu) / r1mag**5 * np.outer(r1, r1)
            - (1 - mu) / r1mag**3 * np.eye(3)
            + 3 * mu / r2mag**5 * np.outer(r2, r2)
            - mu / r2mag**3 * np.eye(3)
        )

    return Uxx


@njit(cache=True)
def get_A_nosingular(
    state: NDArray[np.floating], mu: float = muEM, R1: float = REarth, R2: float = RMoon
) -> NDArray[np.floating]:
    pos = state[:3]
    Uxx = U_hess_nosingular(pos, mu, R1, R2)
    O = np.zeros((3, 3))
    I = np.eye(3)
    Omega = np.array([[0, 2, 0], [-2, 0, 0], [0, 0, 0]])
    A1 = np.concatenate((O, I), axis=1)
    A2 = np.concatenate((Uxx, Omega), axis=1)
    A = np.concatenate((A1, A2), axis=0)
    return A


@njit(cache=True)
def eom_nosingular(
    _,
    state: NDArray[np.floating],
    mu: float = muEM,
    R1: float = REarth,
    R2: float = RMoon,
) -> NDArray[np.floating]:
    x, y, z, vx, vy, vz = state[:6]
    xyz = state[:3]
    r1vec = xyz + np.array([mu, 0, 0])
    r2vec = xyz + np.array([mu - 1, 0, 0])
    r1mag = np.linalg.norm(r1vec)
    r2mag = np.linalg.norm(r2vec)

    if r1mag < R1:
        sf = r1mag / R1
        ddxyz = (
            -(sf**2) * (1 - mu) * r1vec / r1mag**3
            - mu * r2vec / r2mag**3
            + np.array([2 * vy + x, -2 * vx + y, 0])
        )
    elif r2mag < R2:
        sf = r2mag / R2
        ddxyz = (
            -(1 - mu) * r1vec / r1mag**3
            - sf**2 * mu * r2vec / r2mag**3
            + np.array([2 * vy + x, -2 * vx + y, 0])
        )

    else:
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
def coupled_stm_eom_nosingular(
    _,
    state: NDArray[np.floating],
    mu: float = muEM,
    R1: float = REarth,
    R2: float = RMoon,
) -> NDArray[np.floating]:
    pv = state[:6]
    dpv = eom_nosingular(0.0, pv, mu, R1, R2)
    stm = state[6:].reshape((6, 6))
    A = get_A_nosingular(pv, mu, R1, R2)  # pv[:3]
    dstm = A @ stm

    dstate = np.array([*dpv, *dstm.flatten()])
    return dstate


# %% tools


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


def prop_ic_fullstate(
    X: NDArray,
    X2xtf_func: Callable,
    mu: float = muEM,
    int_tol=1e-12,
    density_mult: int = 2,
):
    x0, tf = X2xtf_func(X)
    ts, xs1, (_, Fs), _ = dop853(
        eom, (0.0, tf), x0, int_tol, args=(mu,), dense_output=True
    )
    ts, xs = dop_interpolate(ts, xs1.T, Fs, n_mult=density_mult)
    return xs


def prop(
    x0: NDArray,
    tf: float,
    mu: float = muEM,
    int_tol: float = 1e-11,
    density_mult: int | None = 2,
):
    dense = True if density_mult is not None and density_mult > 1 else False
    ts, xs1, (_, Fs), _ = dop853(
        eom, (0.0, tf), x0, int_tol, args=(mu,), dense_output=dense
    )
    if dense:
        ts, xs = dop_interpolate(ts, xs1.T, Fs, n_mult=density_mult)
        return xs
    else:
        return xs1


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
    ode_out = dop853(
        coupled_stm_eom, (0.0, period), sv0, int_tol, t_eval=te, args=(mu,)
    )[1]
    svs = ode_out.T[:-1]
    xs = [sv[:6] for sv in svs]
    mono = ode_out.T[-1, 6:].reshape(6, 6)
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


# def integrate_one(
#     lock: threading.Lock,
#     dict_out: dict,
#     index: int,
#     tf: float,
#     x0: NDArray,
#     events: Callable | List,
#     mu: float,
#     int_tol: float,
# ):
#     """Propagate multiple curves to a common set of event functions. Uses scipy propagator, so may not be great.

#     Args:
#         lock (threading.Lock): Thread lock to prevent simultaneous writes to dict
#         dictr_out (dict): dictionary to fill with outputs
#         index (int): Index to place result in dict
#         tf (float): Max integration time, in case event doesnt trigger. Also holds information about integration direction.
#         x0 (NDArray): Initial condition
#         events (Callable | List): Event function or event functions list. Must have signature (t, x, mu) -> float
#         mu (float, optional): Mass parameter. Defaults to muEM.
#         int_tol (float, optional): Integration tolerance. Defaults to 1e-10.

#     Returns:
#         dict[int, OdeResult]: Indexed results with ODE output
#     """
#     try:
#         iter(events)
#         pass
#     except TypeError:
#         events = [events]

#     ode_out = solve_ivp(
#         eom,
#         (0.0, tf),
#         x0,
#         "DOP853",
#         atol=int_tol,
#         rtol=int_tol,
#         args=(mu,),
#         events=events,
#     )
#     with lock:
#         dict_out[index] = ode_out


# def prop_multiple(
#     x0s: NDArray | List,
#     events: Callable | List,
#     tfmax: float,
#     mu: float = muEM,
#     int_tol: float = 1e-10,
# ) -> dict[int, OdeResult]:
#     """Propagate multiple curves to a common set of event functions. Uses scipy propagator, so may not be great.

#     Args:
#         x0s (NDArray | List): List of initial conditions, ordered. Nx6
#         events (Callable | List): Event function or event functions list. Must have signature (t, x, mu) -> float
#         tfmax (float): Maximum final time, in case event never triggers
#         mu (float, optional): Mass parameter. Defaults to muEM.
#         int_tol (float, optional): Integration tolerance. Defaults to 1e-10.

#     Returns:
#         dict[int, OdeResult]: Indexed results with ODE output
#     """
#     dct_out = {}
#     N = len(x0s)
#     lock = threading.Lock()

#     threads = []
#     for ind in range(N):
#         x0 = x0s[ind]
#         args = (lock, dct_out, ind, tfmax, x0, events, mu, int_tol)
#         thread = threading.Thread(target=integrate_one, args=args)
#         threads.append(thread)

#     for thread in threads:
#         thread.start()
#     for thread in threads:
#         thread.join()

#     return dct_out


@njit
def hit_moon_dispatcher(i, t, x, args):
    """Event function for collision with the primaries
    args are mu, R(B1), R(B2)

    Will only break if somehow a single integration step passes from just outside of one body to inside the other
    """
    if len(args) == 3:
        mu, r1, r2 = args
    elif len(args) == 1:
        mu = args[0]
    else:
        mu = muEM
        r1 = 6371 / 384400
        r2 = 1740 / 384400

    dy = x[1]
    dz = x[2]
    dx2 = np.abs(x[0] - (1 - mu))
    dx1 = np.abs(x[0] - (-mu))

    if dx1 - r1 < dx2 - r2:
        out = dx1**2 + dy**2 + dz**2 - r1**2
    else:
        out = dx2**2 + dy**2 + dz**2 - r2**2
    return out
    # return dx2**2 + dy**2 - r2**2


# Force recompile again and again
def numba_compile_loop(func, *args, **kwargs):
    while True:
        try:
            out = func(*args, **kwargs)
            break
        except ReferenceError as err:
            if "underlying object has vanished" in str(err):
                print("Numba error, recompiling")
            else:
                print(f"Non-numba-recompile error: {err}")
                break

    return out
