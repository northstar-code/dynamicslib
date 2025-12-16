import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple, List
from numba import njit
from tqdm.auto import tqdm

from dynamicslib.consts import muEM, LU
from dynamicslib.integrate import dop853
from dynamicslib.targeter import dc_square

muEMS = 1.988416e30 / (5.9722e24 + 7.346e22)  # masses from Wiki
aEMS = 1.4960e8 / LU  # Earth SMA from Wikipedia

# NOTE: I have B3 at +x at t=0


@njit(cache=True)
def synodic_period(mu3: float = muEMS, a3: float = aEMS):
    n3 = np.sqrt((1 + mu3) / a3**3) - 1
    return np.abs(2 * np.pi / n3)


@njit(cache=True)
def psuedo_potential(
    t: float,
    state: NDArray[np.floating],
    mu: float = muEM,
    mu3: float = muEMS,
    a3: float = aEMS,
    th0: float = 0.0,
    scale: float = 1.0,
) -> float:
    n3 = np.sqrt((1 + mu3) / a3**3) - 1
    th3 = t * n3 + th0
    r3 = a3 * np.array([np.cos(th3), np.sin(th3), 0])

    x, y, z = state[:3]
    r14mag = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r24mag = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)
    Ugrav = (1 - mu) / r14mag + mu / r24mag
    Ucent = (x**2 + y**2) / 2

    r34mag = np.linalg.norm(state[:3] - r3)
    Ub3 = mu3 / r34mag - mu3 / a3**3 * np.dot(state[:3], r3)
    return Ugrav + Ucent + scale * Ub3


@njit(cache=True)
def eom(
    t: float,
    state: NDArray[np.floating],
    mu: float = muEM,
    mu3: float = muEMS,
    a3: float = aEMS,
    th0: float = 0.0,
    scale: float = 1.0,
) -> NDArray[np.floating]:
    x, y, z, vx, vy, vz = state[:6]
    xyz = state[:3]

    n3 = np.sqrt((1 + mu3) / a3**3) - 1
    th3 = t * n3 + th0
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
    accel = g1 + g2 + scale * g3 + np.array([2 * vy + x, -2 * vx + y, 0])
    return np.concat((state[3:], accel))


@njit(cache=True)
def ppot_hess(
    t: float,
    pos: NDArray[np.floating],
    mu: float = muEM,
    mu3: float = muEMS,
    a3: float = aEMS,
    th0: float = 0.0,
    scale: float = 1.0,
) -> NDArray[np.floating]:
    n3 = np.sqrt((1 + mu3) / a3**3) - 1
    th3 = t * n3 + th0
    r3 = a3 * np.array([np.cos(th3), np.sin(th3), 0])
    r41v = pos - np.array([-mu, 0, 0])
    r42v = pos - np.array([1 - mu, 0, 0])
    r43v = pos - r3
    r41mag = np.linalg.norm(r41v)
    r42mag = np.linalg.norm(r42v)
    r43mag = np.linalg.norm(r43v)

    # G(n)jac is the Jacobian of gravity from body n
    G1jac = (3 / r41mag**5 * np.outer(r41v, r41v) - np.eye(3) / r41mag**3) * (1 - mu)
    G2jac = (3 / r42mag**5 * np.outer(r42v, r42v) - np.eye(3) / r42mag**3) * mu
    G3jac = (3 / r43mag**5 * np.outer(r43v, r43v) - np.eye(3) / r43mag**3) * mu3
    Uxx = G1jac + G2jac + scale * G3jac + np.diag(np.array([1.0, 1, 0]))

    return Uxx


@njit(cache=True)
def get_A(
    t: float,
    state: NDArray[np.floating],
    mu: float = muEM,
    mu3: float = muEMS,
    a3: float = aEMS,
    th0: float = 0.0,
    scale: float = 1.0,
) -> NDArray[np.floating]:
    pos = state[:3]
    Uxx = ppot_hess(t, pos, mu, mu3, a3, th0, scale)
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
    th0: float = 0.0,
    scale: float = 1.0,
) -> NDArray[np.floating]:
    pv = state[:6]
    dpv = eom(t, pv, mu, mu3, a3, th0, scale)
    stm = state[6:].reshape((6, 6))
    A = get_A(t, pv, mu, mu3, a3, th0, scale)
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
    scale: float = 1.0,
) -> float:
    vel = state[3]
    return 2 * psuedo_potential(t, state, mu, mu3, a3, th0, scale) - np.dot(vel, vel)


def homotopy_npc(
    X0: NDArray,
    f_df_stm_func: Callable[[NDArray, float], Tuple[NDArray, NDArray, NDArray]],
    N: int = 100,
    tol: float = 1e-10,
    fudge: float = 1.0,
    max_step: float | None = None,
    debug: bool = False,
) -> Tuple[List, List, List]:
    """Natural parameter continuation continuation wrapper.

    Args:
        X0 (NDArray): initial control variables
        f_df_stm_func (Callable): function with signature f, df/dX, STM = f_df_func(X, cont_parameter)
        param0 (float): the initial value of the parameter.
        dparam (float): The step in natural parameter to take each iteration
        N (int): The number of steps after which to terminate
        tol (float, optional): tolerance for convergence. Defaults to 1e-10.
        modified (bool, optional): Whether to use modified algorithm. Defaults to True.
        stop_callback (Callable): Function with signature f(X, current_eigvals, previous_eigvals, *kwargs) which returns True when continuation should terminate. If None, will only terminate when the final length is reached. Defaults to None.
        stop_kwags (dict, optional): keyword arguments to stop_calback. Defaults to {}.
        fudge (float | None, optional): multiply step size by this much in the differential corrector
        debug (bool, optional): whether to print off state updates


    Returns:
        Tuple[List, List]: all Xs, all eigenvalues
    """
    # if no stop callback, make one

    X = X0.copy()
    Xg = X

    scales = list(np.linspace(0, 1, 1 + N))
    _, _, stm = f_df_stm_func(X, 0.0)

    Xs = [X]
    eig_vals = [np.linalg.eigvals(stm)]

    bar = tqdm(total=N)
    i = 0
    # ensure that the stopping condition hasnt been satisfied
    for scale in scales[1:]:
        X, _, stm = dc_square(
            Xg, lambda x: f_df_stm_func(x, scale), tol, fudge, max_step, debug
        )
        Xs.append(X)
        eig_vals.append(np.linalg.eigvals(stm))
        bar.update(1)
        i += 1
        Xg = X
        if len(Xs) > 2:
            Xg += Xs[-1] - Xs[-2]

    bar.close()

    return Xs, eig_vals, scales
