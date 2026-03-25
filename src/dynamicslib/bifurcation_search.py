from typing import Callable, List, Any, Iterable
import numpy as np
from scipy.interpolate import CubicSpline
from dynamicslib.interpolate import Hermite_interp_interval
from dynamicslib.targeter import dc_overconstrained


def curve_search(
    eigs: List[np.ndarray] | np.ndarray, bifurcation_type: None | Iterable | Any
):

    eigs = np.array(eigs)
    alpha = 2 - np.sum(eigs, axis=1).real
    beta = (alpha**2 - (np.sum(eigs**2, axis=1).real - 2)) / 2

    def get_per_mult(x, n, q=1):
        a = -2 * np.cos(2 * np.pi * q / n)
        b = 2 - 4 * (np.cos(2 * np.pi * q / n)) ** 2
        return a * x + b

    if bifurcation_type is None:  # tangent
        to_cross = -2 * alpha - 2
    elif isinstance(bifurcation_type, Iterable) and len(bifurcation_type) in [1, 2]:
        # period multiplying
        if len(bifurcation_type) == 1:
            to_cross = get_per_mult(alpha, bifurcation_type[0])
        else:
            to_cross = get_per_mult(alpha, bifurcation_type[0], bifurcation_type[1])
    else:
        to_cross = alpha**2 / 4 + 2

    return beta - to_cross


def new_points(
    eigs_all: np.ndarray,
    Xs_all: np.ndarray,
    tangents_all: np.ndarray,
    f_df_func: Callable,
    bif_type: None,
    targ_tol: float = 1e-12,
    tol: float = 1e-3,
):
    print("START")
    tangents = tangents_all.copy()
    Xs = Xs_all.copy()
    eigs = eigs_all.copy()
    keep_going = True
    while keep_going:
        s_vals = np.cumsum(np.append(0, np.linalg.norm(np.diff(Xs, axis=0), axis=1)))
        curve_vals = curve_search(eigs, bif_type)
        spline = CubicSpline(s_vals, curve_vals)
        roots = spline.roots()
        roots = roots[roots > s_vals[1]]
        roots = roots[roots < s_vals[-1]]
        num_added = 0
        for root in roots[::-1]:
            ind = np.searchsorted(s_vals, root)
            X1, X2 = Xs[ind - 1], Xs[ind]
            T1, T2 = tangents[ind - 1], tangents[ind]
            s1, s2 = s_vals[ind - 1], s_vals[ind]
            if s2 - s1 < tol:
                # print(s1,s2,root)
                continue
            X = Hermite_interp_interval(root, s1, s2, X1, X2, T1, T2)
            X = X.flatten()
            X, df, stm = dc_overconstrained(X, f_df_func, targ_tol, debug=False)
            tangent = np.linalg.svd(df).Vh[-1]
            eig = np.linalg.eigvals(stm)

            eigs = np.insert(eigs, ind, eig, axis=0)
            Xs = np.insert(Xs, ind, X, axis=0)
            tangents = np.insert(tangents, ind, tangent, axis=0)
            num_added += 1
        print(f"Tot added: {num_added}")
        if num_added == 0:
            keep_going = False
    return (eigs, Xs, tangents)
