from dynamicslib.common import *
from dynamicslib.integrator import *


# %% continuation
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


def dc_arclen(
    X_prev: NDArray,
    tangent: NDArray,
    f_df_func: Callable,
    s: float = 1e-3,
    tol: float = 1e-8,
    modified: bool = True,
    max_iter: int | None = None,
    fudge: float | None = None,
    debug: bool = False,
    normalize: bool = False,
    # maxstep: float | None = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Pseudoarclength continuation differential corrector. The modified algorithm has a full step size of s, rather than projected step size.

    Args:
        X_prev (NDArray): previous control variables
        tangent (NDArray): tangent to previous orbit. Would be nice to not have to carry over...
        f_df_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        s (float, optional): step size. Defaults to 1e-3.
        tol (float, optional): tolerance for convergence. Defaults to 1e-8.
        modified (boolean, optional): whether to use modified algorithm. Defaults to True.
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
        if normalize:
            dX /= np.linalg.norm(dX)
        X += dX * fudge
        if debug:
            print(niters, dX)
        niters += 1

    return X, dF, stm_full


def dc_npc(
    X_guess: NDArray,
    f_df_func: Callable,
    tol: float = 1e-8,
    fudge: float = 1,
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
        X += fudge * dX
        niters += 1
        if debug:
            print(dX)

    return X, dF, stm_full


def dc_underconstrained(
    X_guess: NDArray,
    f_df_func: Callable,
    tol: float = 1e-8,
    fudge: float = 1,
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
