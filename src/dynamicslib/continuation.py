from dynamicslib.targeter import *
from warnings import warn
from typing import List
from tqdm.auto import tqdm
import pandas as pd
from scipy.interpolate import UnivariateSpline


def arclen_cont(
    X0: NDArray,
    f_df_stm_func: Callable[[NDArray], Tuple[NDArray, NDArray, NDArray]],
    dir0: NDArray | List,
    s: float = 1e-3,
    S: float = 0.5,
    tol: float = 1e-10,
    max_iter: None | int = None,
    max_step: None | float = None,
    fudge: float | None = None,
    exact_tangent: bool = False,
    modified: bool = True,
    stop_callback: Callable | None = None,  # possibly change to also take dF?
    stop_kwags: dict = {},
) -> Tuple[List, List]:
    """Pseudoarclength continuation wrapper. The modified algorithm has a full step size of s, rather than projected step size.

    Args:
        X0 (NDArray): initial control variables
        f_df_stm_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        dir0 (NDArray | List): rough initial stepoff direction. Is mostly just used to switch the direction of the computed tangent vector
        s (float, optional): step size. Defaults to 1e-3.
        S (float, optional): terminate at this arclength. Defaults to 0.5.
        tol (float, optional): tolerance for convergence. Defaults to 1e-10.
        max_iter: (int | None, optional): maximum number of iterations. Will return what it's computed so far if it exceeds that
        fudge: (float | None, optional): multiply step size by this much in the differential corrector
        exact_tangent (bool, optional): whether the tangent vector `dir0` passed in is exact or approximate. If approximate, it is only used to check direction with a dot product. Otherwise, it is used as-is.
        modified (bool, optional): Whether to use modified algorithm. Defaults to True.
        stop_callback (Callable): Function with signature f(X, current_eigvals, previous_eigvals, *kwargs) which returns True when continuation should terminate. If None, will only terminate when the final arclength is reached. Defaults to None.
        stop_kwags (dict, optional): keyword arguments to stop_calback. Defaults to {}.


    Returns:
        Tuple[List, List]: all Xs, all eigenvalues
    """
    # if no stop callback, make one
    if callable(stop_callback):
        stopfunc = lambda X, ecurr, elast: stop_callback(X, ecurr, elast, **stop_kwags)
    else:
        stopfunc = lambda X, ecurr, elast: False

    X = X0.copy()
    tangent_prev = dir0 / np.linalg.norm(dir0)

    _, dF, stm = f_df_stm_func(X0)
    svd = np.linalg.svd(dF)
    tangent = dir0.copy() if exact_tangent else svd.Vh[-1]

    # # if the direction we asked for is normal to the computed tangent, use the second-most tangent vector
    # if np.abs(np.dot(tangent, dir0)) < 1e-5:
    #     print("RESETTING")
    #     tangent = svd.Vh[-1]

    Xs = [X0]
    eig_vals = [np.linalg.eigvals(stm)]

    bar = tqdm(total=S)
    arclen = 0.0

    # ensure that the stopping condition hasnt been satisfied
    while arclen < S and not (arclen > 0 and stopfunc(X, eig_vals[-1], eig_vals[-2])):
        # if we flip flop, undo the flipflop
        if np.dot(tangent, tangent_prev) < 0:
            tangent *= -1
        try:
            X, dF, stm, _ = dc_arclen(
                X,
                tangent,
                f_df_stm_func,
                s,
                tol,
                modified=modified,
                max_iter=max_iter,
                max_step=max_step,
                fudge=fudge,
            )
        except np.linalg.LinAlgError as err:
            print(f"Linear algebra error encountered: {err}")
            print("returning what's been calculated so far")
            break
        except RuntimeError as err:
            print(f"Runtime error encountered: {err}")
            print("returning what's been calculated so far")
            break

        Xs.append(X)

        eig_vals.append(np.linalg.eigvals(stm))
        dS = s if modified else np.linalg.norm(Xs[-1] - Xs[-2])

        tangent_prev = tangent

        svd = np.linalg.svd(dF)
        tangent = svd.Vh[-1]

        arclen += dS
        bar.update(float(dS))

    bar.close()

    return Xs, eig_vals


def arclen_variable_step(
    X0: NDArray,
    f_df_stm_func: Callable[[NDArray], Tuple[NDArray, NDArray, NDArray]],
    dir0: NDArray | List,
    s0: float = 1e-2,
    s_min: float = 1e-3,
    S: float = 0.5,
    tol: float = 1e-10,
    max_iter: int = 10,
    rate: float = 1.05,
    reduce_maxiter: float = 5.0,
    reduce_reverse: float = 2.0,
    exact_tangent: bool = False,
) -> Tuple[List, List]:
    """Pseudoarclength continuation wrapper with variable step size. This modified algorithm has a full step size of s, rather than projected step size.
    At each step, the step size multiplies by num_iters/num_iters_previous, in so that if it takes longer to converge we reduce the step size
    At each step, the step size also multiplies by the dot product between the tangent vector and the step; if this dot product is close to 1, then the curve is not sharp and step size wont be reduced. Else, it will.
    At each step, the step size also multiplies by the parameter `rate`, which should be >1 to ensure step size can recover

    Args:
        X0 (NDArray): initial control variables
        f_df_stm_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        dir0 (NDArray | List): rough initial stepoff direction. Is mostly just used to switch the direction of the computed tangent vector
        s (float, optional): step size. Defaults to 1e-3.
        S (float, optional): terminate at this arclength. Defaults to 0.5.
        tol (float, optional): tolerance for convergence. Defaults to 1e-10.
        max_iter: (int | None, optional): maximum number of iterations. Will return what it's computed so far if it exceeds that
        rate (float, optional): the rate of increase of step size in the absense of any other change
        reduce_maxiter (float, optional): If we hit the maximum iterations on one attempt, reduce the step size by a factor of this much
        reduce_reverse (float, optional): If there's a possibility that the solution curve is reversing backward on itself, reduce the step size by a factor of this much
        exact_tangent (bool, optional): whether the tangent vector `dir0` passed in is exact or approximate. If approximate, it is only used to check direction with a dot product. Otherwise, it is used as-is.


    Returns:
        Tuple[List, List]: all Xs, all eigenvalues
    """
    assert rate >= 1
    assert reduce_maxiter > 1
    assert reduce_reverse > 1

    X = X0.copy()
    tangent_prev = dir0 / np.linalg.norm(dir0)

    _, dF, stm = f_df_stm_func(X0)
    svd = np.linalg.svd(dF)
    tangent = tangent_prev.copy() if exact_tangent else svd.Vh[-1]

    # # if the direction we asked for is normal to the computed tangent, use the second-most tangent vector
    # if np.abs(np.dot(tangent, dir0)) < 1e-5:
    #     print("RESETTING")
    #     tangent = svd.Vh[-1]

    Xs = [X0]
    eig_vals = [np.linalg.eigvals(stm)]

    bar = tqdm(total=S)
    arclen = 0.0
    s = s0

    niters = 0
    niters_prev = max_iter
    # ensure that the stopping condition hasnt been satisfied
    while arclen < S and s >= s_min:
        bar.set_description(f"s = {s:.3e}")

        try:
            X, dF, stm, niters = dc_arclen(
                X, tangent, f_df_stm_func, s, tol, modified=True, max_iter=max_iter
            )
        except np.linalg.LinAlgError as err:
            print(f"Linear algebra error encountered: {err}")
            print("returning what's been calculated so far")
            break
        except RuntimeError as err:
            warn(f"@S={arclen:.3f}: Failed step, decreasing step size and rejecting")
            # reject the last step
            s /= reduce_maxiter
            dS = np.linalg.norm(Xs[-1] - Xs[-2])
            arclen -= dS
            bar.update(-dS)
            Xs.pop()
            X = Xs[-1]
            tangent = tangent_prev
            continue
        if arclen == 0.0:
            niters_prev = niters

        # print(np.dot(tangent, X - Xs[-1])/s)
        dprod_check = np.dot(tangent, X - Xs[-1]) / s
        if dprod_check < 0.5:
            warn(
                "@S={arclen:.3f}: Possibly reversal, decreasing step size and rejecting"
            )
            # reject the last step
            s /= reduce_reverse
            dS = np.linalg.norm(Xs[-1] - Xs[-2])
            arclen -= dS
            bar.update(-dS)
            Xs.pop()
            X = Xs[-1]
            tangent = tangent_prev

        Xs.append(X)

        eig_vals.append(np.linalg.eigvals(stm))

        tangent_prev = tangent

        svd = np.linalg.svd(dF)
        tangent = svd.Vh[-1]
        # if we flip flop, undo the flipflop
        if np.dot(tangent, Xs[-1] - Xs[-2]) < 0:
            tangent *= -1

        arclen += s
        bar.update(float(s))
        s *= (niters_prev / niters) * rate * dprod_check
        niters_prev = niters

    if s < s_min:
        print("Step size smaller than minimum allowable- terminating")

    bar.close()

    return Xs, eig_vals


def natural_param_cont(
    X0: NDArray,
    f_df_stm_func: Callable[
        [float], Callable[[NDArray], Tuple[NDArray, NDArray, NDArray]]
    ],
    param0: float = 0,
    dparam: float = 1e-2,
    N: int = 10,
    tol: float = 1e-10,
    stop_callback: Callable | None = None,
    stop_kwags: dict = {},
    fudge: float = 1,
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
    if callable(stop_callback):
        stopfunc = lambda X, ecurr, elast: stop_callback(X, ecurr, elast, **stop_kwags)
    else:
        stopfunc = lambda X, ecurr, elast: False

    X = X0.copy()

    param = param0
    params = [param0]

    _, dF, stm = f_df_stm_func(param)(X0)

    Xs = [X0]
    eig_vals = [np.linalg.eigvals(stm)]

    bar = tqdm(total=N)
    i = 0
    # ensure that the stopping condition hasnt been satisfied
    while i < N and not (param > param0 and stopfunc(X, eig_vals[-1], eig_vals[-2])):
        X, dF, stm = dc_square(
            X + dparam, f_df_stm_func(param), tol, fudge, None, debug
        )
        params.append(param)
        Xs.append(X)
        eig_vals.append(np.linalg.eigvals(stm))
        param += dparam
        bar.update(1)
        i += 1

    bar.close()

    return Xs, eig_vals, params


def find_bif(
    X0: NDArray | List,
    f_df_stm_func: Callable[[NDArray], Tuple[NDArray, NDArray, NDArray]],
    dir0: NDArray | List,
    s0: float = 1e-2,
    targ_tol: float = 1e-10,
    skip: int = 0,
    bisect_tol: float = 1e-5,
    bif_type: str | Tuple[int, int] | Tuple[int] = "tangent",
    debug: bool = False,
    scale: float = 5,
) -> Tuple[NDArray, NDArray]:
    """Find bifurcation using Broucke stability

    Args:
        X0 (NDArray): initial control variables
        f_df_stm_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        dir0 (NDArray | List): signed initial stepoff direction.
        s0 (float, optional): initial step size. Defaults to 1e-2.
        targ_tol (float, optional): tolerance for targetter convergence. Defaults to 1e-10.
        skip (int, optional): number of crossings to skip. Defaults to 0.
        bisect_tol (float, optional): Tolerance for bisection algorithm. Defaults to 1e-5.
        bif_type (str, optional): bif_type of bifurcation to detect ("tangent", "hopf") OR
            a tuple indicating period-multiplying bifurcation (e.g. (3,)
            for tripling, (5,2) for quintupling with second harmonic). Defaults to "tangent".
        debug (bool, optional): whether to print off function evaluations and steps

    Returns:
        NDArray: Bifurcation control variables, tangent vector
    """
    if isinstance(bif_type, tuple):
        # Period multiplying

        # generally, beta = a*alpha+b where a = -2cos(q2pi/n), 2-4cos^2(q2pi/n) for n-periodic and q\in 1..n/2
        if len(bif_type) == 1:
            n = bif_type[0]
        elif len(bif_type) == 2:
            n = bif_type[0] / bif_type[1]
        else:
            raise ValueError(
                "Period-multiplying bifurcation type must be given as (n,) or (n,m)"
            )
        angle = 2 * np.pi / n
        cos_val = np.cos(angle)
        bisect_func = (
            lambda alpha, beta: -2 * cos_val * alpha + (2 - 4 * cos_val**2) - beta
        )
    else:
        match bif_type.lower():
            case "tangent":
                bisect_func = lambda alpha, beta: beta + 2 + 2 * alpha
            case "hopf":
                bisect_func = lambda alpha, beta: beta - alpha**2 / 4 - 2
            case _:
                raise NotImplementedError("womp womp")

    X = np.array(X0) if isinstance(X0, list) else X0.copy()
    tangent_prev = np.array(dir0) if isinstance(dir0, list) else dir0.copy()
    s = s0

    _, dF, stm = f_df_stm_func(X0)
    svd = np.linalg.svd(dF)
    tangent = svd.Vh[-1]

    Xs = [X0]

    alpha = 2 - np.trace(stm)
    beta = 1 / 2 * (alpha**2 + 2 - np.trace(stm @ stm))
    func_vals = [bisect_func(alpha, beta)]

    while True:
        if np.dot(tangent, tangent_prev) < 0:
            tangent *= -1
        X, dF, stm, _ = dc_arclen(
            X, np.sign(s) * tangent, f_df_stm_func, abs(s), targ_tol
        )

        Xs.append(X.copy())
        tangent_prev = tangent

        # tangent = null_space(dF)
        svd = np.linalg.svd(dF)
        tangent = svd.Vh[-1]

        alpha = 2 - np.trace(stm)
        beta = 1 / 2 * (alpha**2 + 2 - np.trace(stm @ stm))

        func_vals.append(bisect_func(alpha, beta))

        if np.sign(func_vals[-1]) != np.sign(func_vals[-2]):
            if skip == 0:
                if abs(func_vals[-1]) < bisect_tol or abs(s) < bisect_tol:
                    tangent = svd.Vh[-2]
                    print(f"BIFURCATING @ X={X} in the direction of {tangent}")
                    return X, tangent
                else:  # search backward
                    s /= -scale
            else:
                skip -= 1
        if abs(func_vals[-1]) < bisect_tol:
            tangent = svd.Vh[-2]
            print(f"BIFURCATING @ X={X} in the direction of {tangent}")
            return X, tangent

        if debug:
            print(func_vals[-1], func_vals[-2], s)  # , X)


# def find_per_mult(
#     X0: NDArray,
#     f_df_stm_func: Callable[[NDArray], Tuple[NDArray, NDArray, NDArray]],
#     dir0: NDArray | List,
#     s0: float = 1e-3,
#     tol: float = 1e-10,
#     N: int = 3,
#     angeps=1e-4,
#     skip: int = 0,
# ) -> Tuple[NDArray, NDArray]:
#     """Find find period multiplying bifurcation with period N

#     Args:
#         X0 (NDArray): initial control variables
#         f_df_stm_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
#         dir0 (NDArray | List): rough initial stepoff direction. Is mostly just used to switch the direction of the computed tangent vector
#         s0 (float, optional): step size. Defaults to 1e-3.
#         tol (float, optional): tolerance for convergence. Defaults to 1e-10.
#         N (int): multiplier
#         angeps (float, optional): How exact does the argument need to be?

#     Returns:
#         NDArray: Bifurcation control variables, tangent vector
#     """
#     assert N > 1
#     target_eigval = np.cos(2 * np.pi / N) + 1j * np.sin(2 * np.pi / N)

#     s = s0

#     X = X0.copy()
#     dir0 = np.array(dir0)
#     tangent_prev = dir0
#     targang = 2 * np.pi / N

#     _, dF, stm = f_df_stm_func(X0)
#     # svd = np.linalg.svd(dF)
#     tangent = tangent_prev.copy()

#     Xs = [X0]

#     # eigs_prev =
#     eigs = [np.linalg.eigvals(stm)]
#     # justSwitched = False

#     while True:
#         if np.dot(tangent, tangent_prev) < 0:
#             tangent *= -1
#         # if justSwitched:
#         #     tangent *= -1
#         #     justSwitched = False

#         Xs.append(X)

#         # check the argument
#         eigs.append(np.linalg.eigvals(stm))
#         print(eigs[-1])

#         argc = np.argmin(np.abs(eigs[-1] - target_eigval))
#         valc = eigs[-1][argc]
#         angc = np.angle(valc)
#         argp = np.argmin(np.abs(eigs[-2] - target_eigval))
#         valp = eigs[-2][argp]
#         angp = np.angle(valp)

#         if N > 2:
#             # whether weve crossed and are unit magnitude
#             cross1 = (
#                 (angc < targang) and (angp > targang) and abs(abs(valc) - 1) < angeps
#             ) or abs(angc - targang) < angeps
#             cross2 = (
#                 (angc > targang) and (angp < targang) and abs(abs(valc) - 1) < angeps
#             ) or abs(angc - targang) < angeps
#         else:  # untested
#             # whether we crossed x=-1. If we were getting closer but now we're getting further. In other words, if distance is increasing?

#             # current and previous distance to -1
#             distc = np.abs(valc + 1)
#             distp = np.abs(valp + 1)

#             cross1 = (
#                 np.real(valc) < -1
#                 and abs(np.imag(valc)) < angeps
#                 and np.real(valp) > -1
#             )
#             cross2 = (
#                 np.real(valp) < -1
#                 and abs(np.imag(valp)) < angeps
#                 and np.real(valc) > -1
#             )
#         # print(np.angle(eigs[-1]))
#         # print(np.angle(eigs[-1]))

#         if abs(angc - targang) < angeps and skip == 0:
#             break
#         if cross1 or cross2:
#             if skip > 0:
#                 skip -= 1
#             else:
#                 Xs = [X]
#                 eigs = [eigs[-1]]
#                 tangent *= -1
#                 s /= 10

#         tangent_prev = tangent

#         svd = np.linalg.svd(dF)
#         tangent = svd.Vh[-1]

#         X, dF, stm_,  = dc_arclen(X, tangent, f_df_stm_func, s, tol)

#     X[-1] *= N
#     _, dF, _ = f_df_stm_func(X)
#     svd = np.linalg.svd(dF)
#     tangent = svd.Vh[-2]
#     return X, tangent


# def find_any_bif(
#     X0: NDArray,
#     f_df_stm_func: Callable[[NDArray], Tuple[NDArray, NDArray, NDArray]],
#     dir0: NDArray | List,
#     s: float = 1e-3,
#     tol: float = 1e-10,
#     skip_changes: int = 0,
#     stabEps: float = 1e-5,
# ) -> Tuple[NDArray, NDArray]:
#     """Find bifurcation using changes in stability. This function can likely be gotten rid of

#     Args:
#         X0 (NDArray): initial control variables
#         f_df_stm_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
#         dir0 (NDArray | List): rough initial stepoff direction. Is mostly just used to switch the direction of the computed tangent vector
#         s (float, optional): step size. Defaults to 1e-3.
#         tol (float, optional): tolerance for convergence. Defaults to 1e-10.
#         skip_changes (int, optional): number of stability changes to skip. Defaults to 0.
#         stabEps (float, optional): Arbitrary epsilon to determine when an eigenvalue = +/-1. Defaults to 1e-5.

#     Returns:
#         NDArray: Bifurcation control variables, tangent vector
#     """
#     X = X0.copy()
#     tangent_prev = dir0

#     _, dF, stm = f_df_stm_func(X0)
#     svd = np.linalg.svd(dF)
#     tangent = svd.Vh[-1]

#     Xs = [X0]

#     stabs_prev = [None] * 6

#     while True:
#         if np.dot(tangent, tangent_prev) < 0:
#             tangent *= -1
#         X, dF, stm, _ = dc_arclen(X, tangent, f_df_stm_func, s, tol)

#         Xs.append(X)

#         # eval_norms = np.sort(np.abs(eig_vals[-1]))[3:]
#         stabs = sorted([get_stab(e, stabEps) for e in np.linalg.eigvals(stm)])

#         tangent_prev = tangent

#         # tangent = null_space(dF)
#         svd = np.linalg.svd(dF)
#         tangent = svd.Vh[-1]

#         if stabs != stabs_prev and None not in stabs_prev:
#             # if abs(svd.S[-2]) <= 0.5:
#             # if svd.
#             if skip_changes == 0:
#                 tangent = svd.Vh[-2]
#                 print(f"BIFURCATING @ X={X} in the direction of {tangent}")
#                 return X, tangent
#             else:
#                 skip_changes -= 1
#             # else:
#             #     pass

#         stabs_prev = stabs


def arclen_to_fail(
    X0: NDArray,
    f_df_stm_func: Callable[[NDArray], Tuple[NDArray, NDArray, NDArray]],
    dir0: NDArray | List,
    s0: float = 1e-3,
    tol: float = 1e-10,
    max_iter: int = 10,
    multiplier: float = 1.01,
    wait: int = 100,
) -> Tuple[List, List]:
    """Arclength continuation until fail.

    Args:
        X0 (NDArray): initial control variables
        f_df_stm_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        dir0 (NDArray | List): rough initial stepoff direction. Is mostly just used to switch the direction of the computed tangent vector
        s0 (float, optional): initial step size. Defaults to 1e-3.
        tol (float, optional): tolerance for convergence. Defaults to 1e-10.
        max_iter (int): number of iterations to fail at
        multiplier (float): multiply step size by this
        wait (int): after this many steps

    Returns:
        Tuple[List, List]: all Xs, all eigenvalues
    """

    X = X0.copy()
    tangent_prev = dir0.copy()
    tangent = dir0.copy()

    _, dF, stm = f_df_stm_func(X0)
    svd = np.linalg.svd(dF)

    Xs = [X0]
    eig_vals = [np.linalg.eigvals(stm)]

    bar = tqdm(total=0)
    arclen = 0.0

    s = s0
    lastInc = 0

    # ensure that the stopping condition hasnt been satisfied
    while True:
        if arclen - lastInc > wait * s:
            s *= multiplier
        # if we flip flop, undo the flipflop
        if np.dot(tangent, tangent_prev) < 0:
            tangent *= -1
        try:
            X, dF, stm = dc_arclen(
                X, tangent, f_df_stm_func, s, tol, True, max_iter=max_iter
            )
        except Exception as err:
            print(f"Runtime error encountered: {err}")
            print("returning what's been calculated so far")
            break

        Xs.append(X)

        eig_vals.append(np.linalg.eigvals(stm))
        dS = s

        tangent_prev = tangent

        svd = np.linalg.svd(dF)
        tangent = svd.Vh[-1]

        arclen += dS
        bar.update(float(dS))

    bar.close()

    return Xs, eig_vals


def get_bifurcation_funcs(df, bif_type: tuple | str):
    if isinstance(bif_type, tuple):
        if len(bif_type) == 1:
            n = bif_type[0]
        elif len(bif_type) == 2:
            n = bif_type[0] / bif_type[1]
        else:
            raise ValueError(
                "Period-multiplying bifurcation type must be given as (n,) or (n,m)"
            )
        angle = 2 * np.pi / n
        cos_val = np.cos(angle)
        bisect_func = (
            lambda alpha, beta: -2 * cos_val * alpha + (2 - 4 * cos_val**2) - beta
        )
    else:
        match bif_type.lower():
            case "tangent":
                bisect_func = lambda alpha, beta: beta + 2 + 2 * alpha
            case "hopf":
                bisect_func = lambda alpha, beta: beta - alpha**2 / 4 - 2
            case _:
                raise NotImplementedError("womp womp")

    params = [
        "Initial x",
        "Initial y",
        "Initial z",
        "Initial vx",
        "Initial vy",
        "Initial vz",
        "Period",
    ]
    for param in params:
        if param not in df.columns:
            df[param] = 0.0

    eig_df = df[[col for col in df.columns if "Eig" in col]]
    eigs = eig_df.values.astype(np.complex128)
    alpha = 2 - np.sum(eigs, axis=1).real
    beta = (alpha**2 - (np.sum(eigs**2, axis=1).real - 2)) / 2

    beta_bifurcate = bisect_func(alpha, beta)

    func = beta - beta_bifurcate
    inds = df.index
    spline_dict = {param: UnivariateSpline(inds, df[param]) for param in params}
    return UnivariateSpline(inds, func), spline_dict
