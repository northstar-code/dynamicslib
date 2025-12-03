import numpy as np
from dynamicslib.common import muEM, jacobi_constant, coupled_stm_eom, eom, JCgrad
from dynamicslib.integrator import dop853
from abc import ABC, abstractmethod
from numpy.typing import NDArray


# class Targetter(ABC):
#     @abstractmethod
#     def get_X(self, *args, **kwargs):
#         return np.empty((0,),np.float64)
#     @abstractmethod
#     def get_x0(self, *args, **kwargs):
#         return np.empty((6,),np.float64)
#     @abstractmethod
#     def get_tf(self, *args, **kwargs):
#         return 0.
#     @abstractmethod
#     def DF(self, *args, **kwargs):
#         return np.empty((0,0),np.float64)
#     @abstractmethod
#     def f(self, *args, **kwargs):
#         return np.empty((0),np.float64)
#     @abstractmethod
#     def f_df_stm(self, *args, **kwargs):
#         f = self.f(*args, *kwargs)
#         df = self.DF(*args, *kwargs)
#         stm = np.empty((6,6))
#         return f, df, stm


class JC_fixed_spatial_perpendicular:
    def __init__(self, int_tol: float, JC_targ: float, mu: float = muEM):
        self.int_tol = int_tol
        self.JC_targ = JC_targ
        self.mu = mu

    def get_X(self, x0: NDArray, tf: float):
        return np.array([x0[0], x0[2], x0[-2], tf / 2])

    def get_x0(self, X: NDArray):
        return np.array([X[0], 0, X[1], 0, X[2], 0])

    def get_tf(self, X: NDArray):
        return X[-1] * 2

    def DF(self, x0: NDArray, stm: NDArray, eomf: NDArray):
        x, _, z, _, vy, _ = x0
        x1 = x + self.mu
        x2 = x + self.mu - 1
        JCx = (
            -2 * self.mu * x2 / (z**2 + x2**2) ** (3 / 2)
            + 2 * x
            - 2 * (1 - self.mu) * x1 / (z**2 + x1**2) ** (3 / 2)
        )
        JCz = -2 * self.mu * z / (z**2 + x2**2) ** (3 / 2) - 2 * (
            1 - self.mu
        ) * z / (z**2 + x1**2) ** (3 / 2)
        JCvy = -2 * vy
        dF = np.array(
            [
                [stm[1, 0], stm[1, 2], stm[1, -2], eomf[1]],
                [stm[-3, 0], stm[-3, 2], stm[-3, -2], eomf[-3]],
                [stm[-1, 0], stm[-1, 2], stm[-1, -2], eomf[-1]],
                [JCx, JCz, JCvy, 0.0],
            ]
        )
        return dF

    def f(self, xf: NDArray, JC_target: float | None = None):
        if JC_target is None:
            JC_target = self.JC_targ
        return np.array([xf[1], xf[-3], xf[-1], (jacobi_constant(xf) - JC_target)])

    def f_df_stm(self, X: NDArray, JC_target: float | None = None):
        if JC_target is None:
            JC_target = self.JC_targ
        x0 = self.get_x0(X)
        tf = self.get_tf(X)
        xstmIC = np.array([*x0, *np.eye(6).flatten()])
        ts, ys, _ = dop853(
            coupled_stm_eom,
            (0.0, tf / 2),
            xstmIC,
            self.int_tol,
            self.int_tol,
            args=(self.mu,),
        )
        xf, stm = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        xf = np.array(xf)
        eomf = eom(ts[-1], xf, self.mu)

        dF = self.DF(x0, stm, eomf)
        f = self.f(xf, JC_target)

        G = np.diag([1, -1, 1, -1, 1, -1])
        Omega = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        I = np.identity(3)
        O = np.zeros((3, 3))
        mtx1 = np.block([[O, -I], [I, -2 * Omega]])
        mtx2 = np.block([[-2 * Omega, I], [-I, O]])
        stm_full = G @ mtx1 @ stm.T @ mtx2 @ G @ stm
        return f, dF, stm_full


class spatial_perpendicular:
    def __init__(self, int_tol: float, mu: float = muEM):
        self.int_tol = int_tol
        self.mu = mu

    def get_X(self, x0: NDArray, tf: float):
        return np.array([x0[0], x0[2], x0[-2], tf / 2])

    def get_x0(self, X: NDArray):
        return np.array([X[0], 0, X[1], 0, X[2], 0])

    def get_tf(self, X: NDArray):
        return X[-1] * 2

    def DF(self, stm: NDArray, eomf: NDArray):
        dF = np.array(
            [
                [stm[1, 0], stm[1, 2], stm[1, -2], eomf[1]],
                [stm[-3, 0], stm[-3, 2], stm[-3, -2], eomf[-3]],
                [stm[-1, 0], stm[-1, 2], stm[-1, -2], eomf[-1]],
            ]
        )
        return dF

    def f(self, xf: NDArray):
        return np.array([xf[1], xf[-3], xf[-1]])

    def f_df_stm(self, X: NDArray):
        x0 = self.get_x0(X)
        tf = self.get_tf(X)
        xstmIC = np.array([*x0, *np.eye(6).flatten()])
        ts, ys, _ = dop853(
            coupled_stm_eom,
            (0.0, tf / 2),
            xstmIC,
            self.int_tol,
            self.int_tol,
            args=(self.mu,),
        )
        xf, stm = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        xf = np.array(xf)
        eomf = eom(ts[-1], xf, self.mu)

        dF = self.DF(stm, eomf)
        f = self.f(xf)

        G = np.diag([1, -1, 1, -1, 1, -1])
        Omega = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        I = np.identity(3)
        O = np.zeros((3, 3))
        mtx1 = np.block([[O, -I], [I, -2 * Omega]])
        mtx2 = np.block([[-2 * Omega, I], [-I, O]])
        stm_full = G @ mtx1 @ stm.T @ mtx2 @ G @ stm
        return f, dF, stm_full


class planar_perpendicular:
    def __init__(self, int_tol: float, mu: float = muEM):
        self.int_tol = int_tol
        self.mu = mu

    def get_X(self, x0: NDArray, tf: float):
        return np.array([x0[0], x0[-2], tf / 2])

    def get_x0(self, X: NDArray):
        return np.array([X[0], 0, 0, 0, X[1], 0])

    def get_tf(self, X: NDArray):
        return X[-1] * 2

    def DF(self, stm: NDArray, eomf: NDArray):
        dF = np.array(
            [
                [stm[1, 0], stm[1, -2], eomf[1]],
                [stm[-3, 0], stm[-3, -2], eomf[-3]],
            ]
        )
        return dF

    def f(self, xf: NDArray):
        return np.array([xf[1], xf[-3]])

    def f_df_stm(self, X: NDArray):
        x0 = self.get_x0(X)
        tf = self.get_tf(X)
        xstmIC = np.array([*x0, *np.eye(6).flatten()])
        ts, ys, _ = dop853(
            coupled_stm_eom,
            (0.0, tf / 2),
            xstmIC,
            self.int_tol,
            self.int_tol,
            args=(self.mu,),
        )
        xf, stm = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        xf = np.array(xf)
        eomf = eom(ts[-1], xf, self.mu)

        dF = self.DF(stm, eomf)
        f = self.f(xf)

        G = np.diag([1, -1, 1, -1, 1, -1])
        Omega = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        I = np.identity(3)
        O = np.zeros((3, 3))
        mtx1 = np.block([[O, -I], [I, -2 * Omega]])
        mtx2 = np.block([[-2 * Omega, I], [-I, O]])
        stm_full = G @ mtx1 @ stm.T @ mtx2 @ G @ stm
        return f, dF, stm_full
