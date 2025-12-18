from dynamicslib.integrate import *


def dc_arclen(
    X_prev: NDArray,
    tangent: NDArray,
    f_df_func: Callable,
    s: float = 1e-3,
    tol: float = 1e-8,
    modified: bool = False,
    max_iter: int | None = None,
    fudge: float | None = None,
    max_step: float | None = None,
    debug: bool = False,
    # maxstep: float | None = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Pseudoarclength continuation differential corrector. The modified algorithm has a full step size of s, rather than projected step size.

    Args:
        X_prev (NDArray): previous control variables
        tangent (NDArray): tangent to previous orbit. Would be nice to not have to carry over...
        f_df_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        s (float, optional): step size. Defaults to 1e-3.
        tol (float, optional): tolerance for convergence. Defaults to 1e-8.
        modified (boolean, optional): whether to use modified algorithm. Defaults to False.
        max_iter (int): maximum number of iterations
        fudge (float): multiply step by this much

    Returns:
        Tuple[NDArray, NDArray, NDArray]: X. final dF/dx, full-rev STM
    """
    X = X_prev + s * tangent
    if fudge is None:
        fudge = 1.0

    nX = len(X)
    dF = np.empty((nX - 1, nX))
    stm_full = np.empty((nX, nX))

    G = np.array([np.inf] * nX)
    niters = 0
    dX = np.array([np.inf] * nX)
    while np.linalg.norm(G) > tol and np.linalg.norm(dX) > tol:
        if max_iter is not None and niters > max_iter:
            raise RuntimeError("Exceeded maximum iterations")
        f, dF, stm_full = f_df_func(X)
        delta = X - X_prev
        lastG = np.dot(delta, delta) - s**2 if modified else np.dot(delta, tangent) - s
        lastDG = 2 * delta if modified else tangent
        G = np.array([*f, lastG])
        dG = np.vstack((dF, lastDG))
        dX = -np.linalg.inv(dG) @ G
        if max_step is not None and np.linalg.norm(dX) > max_step:
            dX *= max_step / np.linalg.norm(dX)
        X += dX * fudge
        if debug:
            print(niters, dX)
        niters += 1

    return X, dF, stm_full


def dc_square(
    X_guess: NDArray,
    f_df_func: Callable,
    tol: float = 1e-8,
    fudge: float = 1.0,
    max_step: float | None = None,
    debug: bool = False,
    max_iter: int | None = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Natural parameter continuation differetial corrector

    Args:
        X_guess (NDArray): guess for control variables
        f_df_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        tol (float, optional): tolerance for convergence. Defaults to 1e-8.
        max_iter: maximum number of iterations
        fudge (float): multiply step by this much
        debug (bool): whether to print out steps
        max_iter: int|None: how many iterations to cap at


    Returns:
        Tuple[NDArray, NDArray, NDArray]: X. final dF/dx, full-rev STM
    """

    X = X_guess.copy()

    nX = len(X)
    dF = np.empty((nX, nX))
    stm_full = np.empty((nX, nX))

    f = np.array([np.inf] * nX)
    niters = 0
    dX = np.array([np.inf] * nX)
    while np.linalg.norm(f) > tol and np.linalg.norm(dX) > tol:
        if max_iter is not None and niters > max_iter:
            raise RuntimeError("Exceeded maximum iterations")
        f, dF, stm_full = f_df_func(X)
        dX = -np.linalg.inv(dF) @ f
        if max_step is not None and np.linalg.norm(dX) > max_step:
            dX *= max_step / np.linalg.norm(dX)
        X += fudge * dX
        niters += 1
        if debug:
            print(dX)

    return X, dF, stm_full


def dc_underconstrained(
    X_guess: NDArray,
    f_df_func: Callable,
    tol: float = 1e-8,
    fudge: float = 1.0,
    max_step: float | None = None,
    debug: bool = False,
    max_iter: int | None = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Underconstrained differential corrector

    Args:
        X_guess (NDArray): guess for control variables
        f_df_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        tol (float, optional): tolerance for convergence. Defaults to 1e-8.
        max_iter: maximum number of iterations
        fudge (float): multiply step by this much
        debug (bool): whether to print out steps
        max_iter: int|None: how many iterations to cap at

    Returns:
        Tuple[NDArray, NDArray, NDArray]: X. final dF/dx, full-rev STM
    """
    X = X_guess.copy()
    nX = len(X_guess)
    dF = np.empty((nX, nX))
    stm_full = np.empty((nX, nX))

    f = np.array([np.inf] * nX)
    niters = 0
    dX = np.array([np.inf] * nX)
    while np.linalg.norm(f) > tol and np.linalg.norm(dX) > tol:
        if max_iter is not None and niters > max_iter:
            raise RuntimeError("Exceeded maximum iterations")
        f, dF, stm_full = f_df_func(X)
        dX = -np.linalg.pinv(dF) @ f
        if max_step is not None and np.linalg.norm(dX) > max_step:
            dX *= max_step / np.linalg.norm(dX)
        X += fudge * dX
        niters += 1
        if debug:
            print(dX)

    return X, dF, stm_full


def dc_overconstrained(
    X_guess: NDArray,
    f_df_func: Callable,
    tol: float = 1e-8,
    fudge: float = 1.0,
    max_step: float | None = None,
    debug: bool = False,
    max_iter: int | None = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Underconstrained differential corrector

    Args:
        X_guess (NDArray): guess for control variables
        f_df_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        tol (float, optional): tolerance for convergence. Defaults to 1e-8.
        max_iter: maximum number of iterations
        fudge (float): multiply step by this much
        debug (bool): whether to print out steps
        max_iter: int|None: how many iterations to cap at

    Returns:
        Tuple[NDArray, NDArray, NDArray]: X. final dF/dx, full-rev STM
    """
    X = X_guess.copy()
    nX = len(X_guess)
    dF = np.empty((nX, nX))
    stm_full = np.empty((nX, nX))

    f = np.array([np.inf] * nX)
    niters = 0
    dX = np.array([np.inf] * nX)
    while np.linalg.norm(f) > tol and np.linalg.norm(dX) > tol:
        if max_iter is not None and niters > max_iter:
            raise RuntimeError("Exceeded maximum iterations")
        f, dF, stm_full = f_df_func(X)
        dX = -np.linalg.pinv(dF) @ f
        if max_step is not None and np.linalg.norm(dX) > max_step:
            dX *= max_step / np.linalg.norm(dX)
        X += fudge * dX
        niters += 1
        if debug:
            print(dX)

    return X, dF, stm_full


"""
example funcs for arclength continuation:
```
def xtf2X(x0, tf):
    return np.array([x0[0], x0[2], x0[-2], tf / 2])


def X2xtf(X):
    return np.array([X[0], 0, X[1], 0, X[2], 0]), X[-1] * 2


def stmeom2DF(eomf, stm):
    dF = np.array(
        [
            [stm[1, 0], stm[1, 2], stm[1, -2], eomf[1]],
            [stm[-3, 0], stm[-3, 2], stm[-3, -2], eomf[-3]],
            [stm[-1, 0], stm[-1, 2], stm[-1, -2], eomf[-1]],
        ]
    )
    return dF


def f_func(x0, tf, xf):
    return np.array([xf[1], xf[-3], xf[-1]])


func = lambda X: get_f_df(X, X2xtf, stmeom2DF, f_func, False, muEM, 1e-10)
```

Example for natural parameter:

```
# X: [x, tf]
# f: [y, dx]
# param: vy
def xtf2X(x0, tf):
    return np.array([x0[0], tf / 2])


def X2xtf(X, param):
    return np.array([X[0], 0, 0, 0, param, 0]), X[-1] * 2


def stmeom2DF(eomf, stm):
    dF = np.array([[stm[1, 0], eomf[1]], [stm[-3, 0], eomf[-3]]])
    return dF


def f_func(x0, tf, xf):
    return np.array([xf[1], xf[-3]])


func = lambda param: lambda X: f_df_CR3_single(
    X, lambda X: X2xtf(X, param), stmeom2DF, f_func, False, muEM, 1e-10
)
```

"""
