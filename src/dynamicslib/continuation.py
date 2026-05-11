from dynamicslib.targeter import *
from warnings import warn
from typing import List
from tqdm.auto import tqdm
import pandas as pd
from scipy.interpolate import UnivariateSpline


def fixed_step_cont(
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
    """Custom arclength-based continuation wrapper. The modified algorithm has a full step size of s, rather than projected step size.

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


def adaptive_cont(
    X0: NDArray,
    f_df_stm_func: Callable[[NDArray], Tuple[NDArray, NDArray, NDArray]],
    dir0: NDArray | List,
    s0: float = 1e-2,
    s_min: float = 1e-3,
    S: float = 0.5,
    tol: float = 1e-10,
    max_iter: int = 10,
    target_iter: int = 6,
    rate: float = 1.15,
    reduce_maxiter: float = 5.0,
    reduce_reverse: float = 2.0,
    exp_direction: float = 10.0,
    exp_iters: float = 0.3,
    exact_tangent: bool = False,
    pseudo: bool = False,
) -> Tuple[List, List, Tuple[List, List, List]]:
    """Custom arclength-based continuation wrapper with variable step size. This modified algorithm has a full step size of s, rather than projected step size.
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
        pseudo (bool, optional): whether to use pseudoarclength rather than custom method

    Returns:
        Tuple[List, List]: all Xs, all eigenvalues
    """
    assert rate >= 1
    assert reduce_maxiter > 1
    assert reduce_reverse > 1
    assert max_iter > target_iter

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
    tangents = [tangent.copy()]
    DFs = [dF]
    s_vals = [0]

    bar = tqdm(total=S)
    arclen = 0.0
    s = s0

    niters = 0

    # ensure that the stopping condition hasnt been satisfied

    try:
        while arclen < S and s >= s_min:
            bar.set_description(f"s = {s:.3e}")
            try:
                X, dF, stm, niters = dc_arclen(
                    X, tangent, f_df_stm_func, s, tol, pseudo=pseudo, max_iter=max_iter
                )
            except np.linalg.LinAlgError as err:
                print(f"Linear algebra error encountered: {err}")
                print("returning what's been calculated so far")
                break
            except KeyboardInterrupt as err:
                print("HALTING, returning what's been calculated so far")
                break
            except RuntimeError as err:
                print("Rejecting step")
                s /= reduce_maxiter
                continue

            # print(np.dot(tangent, X - Xs[-1])/s)
            dprod_check = np.dot(tangent, X - Xs[-1]) / s
            if dprod_check < 0.8:
                print(
                    f"@S={arclen:.3f}: Possibly reversal, decreasing step size and rejecting"
                )
                # reject the last step
                s /= reduce_reverse
                dS = np.linalg.norm(Xs[-1] - Xs[-2])
                arclen -= dS
                bar.update(-dS)
                Xs.pop()
                tangents.pop()
                DFs.pop()
                s_vals.pop()
                X = Xs[-1]
                tangent = tangent_prev
                continue

            Xs.append(X)
            tangents.append(tangent)
            eig_vals.append(np.linalg.eigvals(stm))
            DFs.append(dF)
            arclen += s if not pseudo else np.linalg.norm(Xs[-1] - Xs[-2])
            s_vals.append(arclen)

            tangent_prev = tangent

            svd = np.linalg.svd(dF)
            tangent = svd.Vh[-1]
            # if we flip flop, undo the flipflop
            if np.dot(tangent, Xs[-1] - Xs[-2]) < 0:
                tangent *= -1

            bar.update(float(s))
            s *= (target_iter / niters) ** exp_iters * dprod_check**exp_direction
            if niters <= target_iter:
                s *= rate

            if s < s_min:
                print("Step size smaller than minimum allowable- terminating")
    except KeyboardInterrupt as _:
        print("HALTING, returning what's been calculated so far")
    except SystemError as _:
        print("System Err, returning what's been calculated so far")

    bar.close()

    return Xs, eig_vals, (DFs, tangents, s_vals)


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


# def get_bifurcation_funcs(df, bif_type: tuple | str):
#     if isinstance(bif_type, tuple):
#         if len(bif_type) == 1:
#             n = bif_type[0]
#         elif len(bif_type) == 2:
#             n = bif_type[0] / bif_type[1]
#         else:
#             raise ValueError(
#                 "Period-multiplying bifurcation type must be given as (n,) or (n,m)"
#             )
#         angle = 2 * np.pi / n
#         cos_val = np.cos(angle)
#         bisect_func = (
#             lambda alpha, beta: -2 * cos_val * alpha + (2 - 4 * cos_val**2) - beta
#         )
#     else:
#         match bif_type.lower():
#             case "tangent":
#                 bisect_func = lambda alpha, beta: beta + 2 + 2 * alpha
#             case "hopf":
#                 bisect_func = lambda alpha, beta: beta - alpha**2 / 4 - 2
#             case _:
#                 raise NotImplementedError("womp womp")

#     params = [
#         "Initial x",
#         "Initial y",
#         "Initial z",
#         "Initial vx",
#         "Initial vy",
#         "Initial vz",
#         "Period",
#     ]
#     for param in params:
#         if param not in df.columns:
#             df[param] = 0.0

#     eig_df = df[[col for col in df.columns if "Eig" in col]]
#     eigs = eig_df.values.astype(np.complex128)
#     alpha = 2 - np.sum(eigs, axis=1).real
#     beta = (alpha**2 - (np.sum(eigs**2, axis=1).real - 2)) / 2

#     beta_bifurcate = bisect_func(alpha, beta)

#     func = beta - beta_bifurcate
#     inds = df.index
#     spline_dict = {param: UnivariateSpline(inds, df[param]) for param in params}
#     return UnivariateSpline(inds, func), spline_dict
