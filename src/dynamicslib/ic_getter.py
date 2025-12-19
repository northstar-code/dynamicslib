from dynamicslib.consts import *
from dynamicslib.common import get_Lpts, get_A
from dynamicslib.common_targetters import planar_perpendicular
from dynamicslib.targeter import dc_underconstrained
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def get_IC_L4_planar(
    short: bool = True, mu: float = muEM, dy: float = -1e-3
) -> Tuple[NDArray[np.floating], float]:
    """Get IC for a planar L4 orbit (short or long period) parameterized by y distance from the Lagrange point

    Args:
        short (bool, optional): Whether short period (if False, long period). Defaults to True.
        mu (float, optional): Mass parameter. Defaults to muEM.
        dy (float, optional): offset in y from Lagrange point (x offset is 0). Defaults to -1e-3.

    Returns:
        NDArray[np.floating]: Initial state
        float: Period
    """
    g = 1 - 27 * mu * (1 - mu)
    s2 = np.abs(np.sqrt(1 / 2 * (-1 - np.sqrt(g)), dtype=np.complex128))  # SP
    s1 = np.abs(np.sqrt(1 / 2 * (-1 + np.sqrt(g)), dtype=np.complex128))  # LP
    sterm = s2 if short else s1
    Uxx = 3 / 4
    Uxy = 3 * np.sqrt(3) / 2 * (1 / 2 - mu)
    Uyy = 9 / 4
    mtx = np.array([[Uxx + sterm**2, Uxy], [Uxy, Uyy + sterm**2]])
    t1, t2 = mtx @ np.array([0, dy])
    beta2 = t1 / (2 * sterm)
    alpha2 = t2 / (-2 * sterm)
    y0 = dy
    vx0 = -sterm * alpha2
    vy0 = -sterm * beta2
    period = 2 * np.pi / sterm

    sv0 = np.array([1 / 2 - mu, y0 + np.sqrt(3) / 2, 0.0, vx0, vy0, 0])
    return sv0, period


def get_IC_L123_planar(
    Lpt: int, step: float = 1e-3, mu: float = muEM
) -> Tuple[NDArray[np.floating], float]:
    """Get IC for a planar L1 L2 or L3 orbit (Lyapunov) parameterized by stepoff distance from the Lagrange point

    Args:
        lpt (int, optional): Which Lagrange point. Must be in [1,2,3]
        step (float, optional): Stepoff dist from Lagrange point, positive to the left. Defaults to 1e-3
        mu (float, optional): Mass parameter. Defaults to muEM.

    Returns:
        NDArray[np.floating]: Initial state
        float: Period
    """
    assert Lpt in [1, 2, 3]
    xL = get_Lpts(mu)[0, Lpt - 1]
    eq_soln = np.array([xL, 0, 0, 0, 0, 0])
    A = get_A(eq_soln, mu)
    evals = np.linalg.eigvals(A)
    eigmag = np.imag(evals[np.abs(np.real(evals)) < 1e-10][0])
    beta3 = (eigmag**2 + A[3, 0]) / (2 * eigmag)
    vy0 = -beta3 * eigmag * step

    period = 2 * np.pi / eigmag

    x0 = np.array([xL + step, 0, 0, 0, vy0, 0])
    return x0, period


def get_IC_vertical(
    Lpt: int, mu: float = muEM, zstep: float = 1e-3
) -> Tuple[NDArray[np.floating], float]:
    """Get IC for a vertical orbit. Won't be super good, so will need to be differentially corrected

    Args:
        Lpt (int, optional): Lagrange point (must be in [1, 2, 3, 4, 5])
        mu (float, optional): Mass parameter. Defaults to muEM.
        zstep (float, optional): Stepoff dist from Lagrange point in z. Defaults to 1e-3


    Returns:
        NDArray[np.floating]: Initial state
        float: Period
    """
    assert Lpt in [1, 2, 3, 4, 5]
    if Lpt in [1, 2, 3]:
        xL = get_Lpts()[0, Lpt - 1]
        yL = 0.0
    else:
        xL = 1 / 2 - mu
        yL = np.sqrt(3) / 2
        if Lpt == 5:
            yL *= -1
    eq_soln = np.array([xL, yL, 0, 0, 0, 0])
    A = get_A(eq_soln, mu)
    s = np.sqrt(np.abs(A[-1, 2]))

    x0 = np.array([xL, 0, zstep, 0, 0, 0])
    period = 2 * np.pi / s
    return x0, period


def get_IC_resonant_kep(
    p: int,
    q: int,
    e: float,
    mu_body: float,
    prograde: bool = True,
    start_high: bool = True,
) -> Tuple[float, float, float]:
    """Get IC for a p:q resonant orbit using Keplerian assumptions

    Args:
        p (int): Ratio numerator
        q (int): Ratio denominator
        e (float): Eccentricity.
        mu_body (float): Mass parameter OF BODY, nondimensionalized
        prograde (bool, optional): Whether prograde (else, retrograde). Defaults to True.
        start_high (bool, optional): Whether we start high or low. Defaults to True.

    Returns:
        Tuple[float, float, float]: r0, v0, and period. Signed.
    """
    period = 2 * np.pi * q

    a = (mu_body * (q / p) ** 2) ** (1 / 3)  # SMA

    r0 = a * (1 + e) if start_high else a * (1 - e)
    v0_inert = np.sqrt(mu_body * (2 / r0 - 1 / a))

    if not prograde:
        v0_inert *= -1
    vrot = r0
    v0 = v0_inert - vrot
    return r0, v0, period


def get_IC_resonant(
    p: int,
    q: int,
    e: float = 0.2,
    mu: float = muEM,
    start_right: bool = True,
    prograde: bool = True,
    start_high: bool = True,
    body: int = 1,
    debug: bool = False,
    fudge: float = 1,
    int_tol: float = 1e-11,
    tol: float = 1e-10,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Get IC of a p:q resonant orbit. Good idea to plot before doing anything with it.

    Args:
        p (int): Numerator
        q (int): Denominator
        e (float, optional): Eccentricity of initial Keplerian estimate. Defaults to 0.2.
        mu (float, optional): CR3BP mass ratio. Defaults to muEM.
        start_right (bool, optional): Start to the right of body (else start to the left). Defaults to True.
        prograde (bool, optional): Whether the Keplerian guess is prograde (else retrograde). Defaults to True.
        start_high (bool, optional): Whether we start high for initial Keplerian guess (else low). Defaults to True.
        body (int, optional): Which body. Must be in [1, 2]. Defaults to 1.
        debug (bool, optional): Print debug values (state differentials). Defaults to False.
        fudge (float, optional): Fudge factor for corrections. Defaults to 1.
        int_tol (float, optional): Integration tolerance. Defaults to 1e-11.
        tol (float, optional): Targetting tolerance. Defaults to 1e-10.


    Returns:
        Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]: Initial guess, tangent vector, eigenvalues
    """
    targetter = planar_perpendicular(int_tol, muEM)
    func = targetter.f_df_stm
    assert body in [1, 2]

    muB = (1 - mu) if body == 1 else mu
    xB = -mu if body == 1 else 1 - mu

    r0, v0, period = get_IC_resonant_kep(p, q, e, muB, prograde, start_high)

    print("T=", period)
    print("r0=", r0)
    print("STARTING")
    x0 = xB + r0 if start_right else xB - r0
    v0 = v0 if start_right else -v0
    Xg = np.array([x0, v0, period / 2])
    X0, _, _ = dc_underconstrained(Xg, func, fudge=fudge, debug=debug, tol=tol)
    _, dF, stm = func(X0)
    svd = np.linalg.svd(dF)
    tangent = svd.Vh[-1]
    eigs = np.linalg.eigvals(stm)
    return X0, tangent, eigs
