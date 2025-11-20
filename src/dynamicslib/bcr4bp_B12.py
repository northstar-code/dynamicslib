import numpy as np
from numpy.typing import NDArray
from typing import Callable
from numba import njit

from dynamicslib.consts import muEM, LU
from dynamicslib.integrator import dop853, interp_hermite

muEMS = 1.988416e30 / (5.9722e24 + 7.346e22)  # masses from Wiki
aEMS = 1.4960e8 / LU  # Earth SMA from Wikipedia

# NOTE: I have B3 at +x at t=0


@njit(cache=True)
def psuedo_potential(
    t: float,
    state: NDArray[np.floating],
    mu: float = muEM,
    mu3: float = muEMS,
    a3: float = aEMS,
) -> float:
    n3 = np.sqrt((1 + mu3) / a3**3) - 1
    th3 = t * n3
    r3 = a3 * np.array([np.cos(th3), np.sin(th3), 0])

    x, y, z = state[:3]
    r14mag = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r24mag = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)
    Ugrav = (1 - mu) / r14mag + mu / r24mag
    Ucent = (x**2 + y**2) / 2

    r34mag = np.linalg.norm(state[:3] - r3)
    Ub3 = mu3 / r34mag - mu3 / a3**3 * np.dot(state[:3], r3)
    return Ugrav + Ucent + Ub3


@njit(cache=True)
def eom(
    t: float,
    state: NDArray[np.floating],
    mu: float = muEM,
    mu3: float = muEMS,
    a3: float = aEMS,
) -> NDArray[np.floating]:
    x, y, z, vx, vy, vz = state[:6]
    xyz = state[:3]

    n3 = np.sqrt((1 + mu3) / a3**3) - 1
    th3 = t * n3
    r3 = a3 * np.array([np.cos(th3), np.sin(th3), 0])
    r41vec = xyz - np.array([-mu, 0, 0])
    r42vec = xyz - np.array([1 - mu, 0, 0])
    r43vec = xyz - r3
    r41mag = np.linalg.norm(r41vec)
    r42mag = np.linalg.norm(r42vec)
    r43mag = np.linalg.norm(r43vec)

    g1 = -r41vec * (1 - mu) / r41mag**3
    g2 = -r42vec * mu / r42mag**3
    g3 = -r43vec * mu3 / r43mag**3 - r3 * mu3 / a3**3
    accel = g1 + g2 + g3 + np.array([2 * vy + x, -2 * vx + y, 0])
    return np.concat((state[3:], accel))


@njit(cache=True)
def ppot_hess(
    t: float,
    pos: NDArray[np.floating],
    mu: float = muEM,
    mu3: float = muEMS,
    a3: float = aEMS,
) -> NDArray[np.floating]:
    n3 = np.sqrt((1 + mu3) / a3**3) - 1
    th3 = t * n3
    r3 = a3 * np.array([np.cos(th3), np.sin(th3), 0])
    r41vec = pos - np.array([-mu, 0, 0])
    r42vec = pos - np.array([1 - mu, 0, 0])
    r43vec = pos - r3
    r41mag = np.linalg.norm(r41vec)
    r42mag = np.linalg.norm(r42vec)
    r43mag = np.linalg.norm(r43vec)

    # G(n)jac is the Jacobian of gravity from body n
    G1jac = (3 / r41mag**5 * r41vec * r41vec.T - np.eye(3) / r41mag**3) * (1 - mu)
    G2jac = (3 / r42mag**5 * r42vec * r42vec.T - np.eye(3) / r42mag**3) * mu
    G3jac = (3 / r43mag**5 * r43vec * r43vec.T - np.eye(3) / r43mag**3) * mu3
    Uxx = G1jac + G2jac + G3jac + np.diag([1, 1, 0])

    return Uxx


@njit(cache=True)
def get_A(
    t: float,
    state: NDArray[np.floating],
    mu: float = muEM,
    mu3: float = muEMS,
    a3: float = aEMS,
) -> NDArray[np.floating]:
    pos = state[:3]
    Uxx = ppot_hess(t, pos, mu, mu3, a3)
    O = np.zeros((3, 3))
    I = np.eye(3)
    Omega = np.array([[0, 2, 0], [-2, 0, 0], [0, 0, 0]])
    A1 = np.concatenate((O, I), axis=1)
    A2 = np.concatenate((Uxx, Omega), axis=1)
    A = np.concatenate((A1, A2), axis=0)
    return A


@njit(cache=True)
def coupled_stm_eom(
    t: float,
    state: NDArray[np.floating],
    mu: float = muEM,
    mu3: float = muEMS,
    a3: float = aEMS,
) -> NDArray[np.floating]:
    pv = state[:6]
    dpv = eom(t, pv, mu, mu3, a3)
    stm = state[6:].reshape((6, 6))
    A = get_A(t, pv, mu, mu3, a3)
    dstm = A @ stm

    dstate = np.array([*dpv, *dstm.flatten()])
    return dstate


@njit(cache=True)
def pseudo_jc(
    t: float,
    state: NDArray[np.floating],
    mu: float = muEM,
    mu3: float = muEMS,
    a3: float = aEMS,
) -> float:
    vel = state[3]
    return 2 * psuedo_potential(t, state, mu, mu3, a3) - np.dot(vel, vel)


# shortcut to get x,y,z from X
# def prop_ic(
#     X: NDArray,
#     X2xtf_func: Callable,
#     mu: float = muEM,
#     int_tol=1e-12,
#     density_mult: int = 2,
# ):
#     x0, tf = X2xtf_func(X)
#     ts, xs, fs = dop853(eom, (0, tf), x0, rtol=int_tol, atol=int_tol, args=(mu,))
#     ts, dense_sol = interp_hermite(ts, xs.T, fs.T, n_mult=density_mult)
#     x, y, z = dense_sol.T[:3]
#     return x, y, z
