import numpy as np
from dynamicslib.common import muEM, jacobi_constant, coupled_stm_eom, eom, JCgrad
from dynamicslib.integrate import dop853, rk8_step
from dynamicslib.interpolate import dop_interpolate
from numpy.typing import NDArray
from typing import Tuple, List
from abc import ABC, abstractmethod

"""
Targetter objects are used in family continuation, differential correction, and conversion between dynamical systems

Each targetter has methods to get control variables (X) given initial state and period, and another pair of methods to go backward

Targetters all have some way of getting the cost function and its Jacobian

The aforementioned methods are all combined in f_df_stm, which takes a control variable (and possibly more args and kwargs) and
return the cost function, its Jacobian, and STM. In family continuation, this is the only function that should be the only integration instance called.
Some targetters (i.e. heterclinic targetter) are less standard, but still do have these methods

Returns:
    _type_: _description_
"""


class Targetter(ABC):
    @abstractmethod
    def __init__(self, int_tol: float, mu: float, *args, **kwargs):
        self.mu = np.nan
        self.int_tol = np.nan

    @abstractmethod
    def get_X(self, x0: NDArray, period: float) -> NDArray:
        pass

    @abstractmethod
    def get_x0(self, X: NDArray) -> NDArray:
        pass

    @abstractmethod
    def get_period(self, X: NDArray) -> float:
        pass

    @abstractmethod
    def DF(self, *args, **kwargs) -> NDArray:
        pass

    @abstractmethod
    def f(self, *args, **kwargs) -> NDArray:
        pass

    @abstractmethod
    def f_df_stm(self, X: NDArray, *args, **kwargs) -> Tuple[NDArray, NDArray, NDArray]:
        pass


class JC_fixed_spatial_perpendicular(Targetter):
    def __init__(self, int_tol: float, JC_targ: float, mu: float = muEM):
        self.int_tol = int_tol
        self.JC_targ = JC_targ
        self.mu = mu

    def get_X(self, x0: NDArray, period: float):
        return np.array([x0[0], x0[2], x0[-2], period / 2])

    def get_x0(self, X: NDArray):
        return np.array([X[0], 0, X[1], 0, X[2], 0])

    def get_period(self, X: NDArray):
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
        period = self.get_period(X)
        xstmIC = np.array([*x0, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom, (0.0, period / 2), xstmIC, self.int_tol, args=(self.mu,)
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


class spatial_perpendicular(Targetter):
    def __init__(self, int_tol: float, mu: float = muEM):
        self.int_tol = int_tol
        self.mu = mu

    def get_X(self, x0: NDArray, period: float):
        return np.array([x0[0], x0[2], x0[-2], period / 2])

    def get_x0(self, X: NDArray):
        return np.array([X[0], 0, X[1], 0, X[2], 0])

    def get_period(self, X: NDArray):
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
        period = self.get_period(X)
        xstmIC = np.array([*x0, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom, (0.0, period / 2), xstmIC, self.int_tol, args=(self.mu,)
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


class fullstate_minus_one(Targetter):
    def __init__(
        self,
        index_fixed: int,
        index_no_enforce: int,
        value_fixed: float,
        int_tol: float,
        mu: float = muEM,
    ):
        self.int_tol = int_tol
        self.mu = mu
        self.ind_fixed = index_fixed
        self.state_val = value_fixed
        self.ind_skip = index_no_enforce
        assert 0 <= self.ind_fixed < 6
        assert 0 <= self.ind_skip < 6

    def get_X(self, x0: NDArray, period: float):
        return np.append(np.delete(x0, self.ind_fixed), period)

    def get_x0(self, X: NDArray):
        states = np.array(X[:-1])
        return np.insert(states, self.ind_fixed, self.state_val)

    def get_period(self, X: NDArray):
        return X[-1]

    def DF(self, stm: NDArray, eomf: NDArray):
        dF = np.hstack((stm - np.eye(6), eomf[:, None]))
        dF = np.delete(dF, self.ind_fixed, 1)
        dF = np.delete(dF, self.ind_skip, 0)
        return dF

    def f(self, x0: NDArray, xf: NDArray):
        state_diff = xf - x0
        out = np.delete(state_diff, self.ind_skip)
        return out

    def f_df_stm(self, X: NDArray):
        x0 = self.get_x0(X)
        period = self.get_period(X)
        xstmIC = np.array([*x0, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom, (0.0, period), xstmIC, self.int_tol, args=(self.mu,)
        )
        xf, stm = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        xf = np.array(xf)
        eomf = eom(ts[-1], xf, self.mu)

        dF = self.DF(stm, eomf)
        f = self.f(x0, xf)
        return f, dF, stm


class axial(Targetter):
    def __init__(
        self,
        int_tol: float,
        mu: float = muEM,
    ):
        self.int_tol = int_tol
        self.mu = mu

    def get_X(self, x0: NDArray, period: float):
        return np.array([x0[0], x0[-2], x0[-1], period / 2])

    def get_x0(self, X: NDArray):
        return np.array([X[0], 0, 0, 0, X[1], X[2]])

    def get_period(self, X: NDArray):
        return X[-1] * 2

    def DF(self, stm: NDArray, eomf: NDArray):
        dF = np.hstack((stm, eomf[:, None]))
        dF = np.delete(dF, [1, 2, 3], 1)
        dF = np.delete(dF, [0, 4, 5], 0)
        return dF

    def f(self, x0: NDArray, xf: NDArray):
        return np.array([xf[1], xf[2], xf[3]])

    def f_df_stm(self, X: NDArray):
        x0 = self.get_x0(X)
        period = self.get_period(X)
        xstmIC = np.array([*x0, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom, (0.0, period / 2), xstmIC, self.int_tol, args=(self.mu,)
        )
        xf, stm_half = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        xf = np.array(xf)
        eomf = eom(ts[-1], xf, self.mu)

        G = np.diag([1, -1, -1, -1, 1, 1])
        stm = G @ np.linalg.inv(stm_half) @ G @ stm_half

        dF = self.DF(stm_half, eomf)
        f = self.f(x0, xf)
        return f, dF, stm


class xy_symmetric(Targetter):
    def __init__(self, int_tol: float, mu: float = muEM):
        self.int_tol = int_tol
        self.mu = mu

    def get_X(self, x0: NDArray, period: float):
        return np.array([x0[0], x0[1], x0[-3], x0[-2], x0[-1], period / 2])

    def get_x0(self, X: NDArray):
        return np.array([X[0], X[1], 0, X[2], X[3], X[4]])

    def get_period(self, X: NDArray):
        return X[-1] * 2

    def DF(self, stm: NDArray, eomf: NDArray):
        dF = np.hstack((stm - np.eye(6), eomf[:, None]))
        dF = np.delete(dF, 2, 1)
        dF = np.delete(dF, -1, 0)
        return dF

    def f(self, x0: NDArray, xf: NDArray):
        return np.array([*(xf - x0)[:3], *(xf - x0)[[-3, -2]]])

    def f_df_stm(self, X: NDArray):
        x0 = self.get_x0(X)
        period = self.get_period(X)
        xstmIC = np.array([*x0, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom, (0.0, period / 2), xstmIC, self.int_tol, args=(self.mu,)
        )
        xf, stm = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        xf = np.array(xf)
        eomf = eom(ts[-1], xf, self.mu)

        dF = self.DF(stm, eomf)
        f = self.f(x0, xf)

        R = np.diag([1, 1, -1, 1, 1, -1])
        stm_full = R @ stm @ R @ stm
        return f, dF, stm_full


class spatial_period_fixed(Targetter):
    def __init__(self, int_tol: float, period: float, mu: float = muEM):
        self.int_tol = int_tol
        self.mu = mu
        self.period = period

    def get_X(self, x0: NDArray, period: float = np.nan):
        return np.array([x0[0], x0[2], x0[-2]])

    def get_x0(self, X: NDArray):
        return np.array([X[0], 0, X[1], 0, X[2], 0])

    def DF(self, stm: NDArray, eomf: NDArray):
        dF = np.array(
            [
                [stm[1, 0], stm[1, 2], stm[1, -2]],
                [stm[-3, 0], stm[-3, 2], stm[-3, -2]],
                [stm[-1, 0], stm[-1, 2], stm[-1, -2]],
            ]
        )
        return dF
    
    def get_period(self, X: NDArray) -> float:
        return self.period

    def f(self, xf: NDArray):
        return np.array([xf[1], xf[-3], xf[-1]])

    def f_df_stm(self, X: NDArray):
        x0 = self.get_x0(X)
        period = self.period
        xstmIC = np.array([*x0, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom, (0.0, period / 2), xstmIC, self.int_tol, args=(self.mu,)
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


class period_fixed_spatial_perpendicular(Targetter):
    def __init__(self, period: float, int_tol: float, mu: float = muEM):
        self.int_tol = int_tol
        self.mu = mu
        self.period = period

    def get_X(self, x0: NDArray, period: float):
        return np.array([x0[0], x0[2], x0[-2]])

    def get_x0(self, X: NDArray):
        return np.array([X[0], 0, X[1], 0, X[2], 0])

    def DF(self, stm: NDArray):
        dF = np.array(
            [
                [stm[1, 0], stm[1, 2], stm[1, -2]],
                [stm[-3, 0], stm[-3, 2], stm[-3, -2]],
                [stm[-1, 0], stm[-1, 2], stm[-1, -2]],
            ]
        )
        return dF

    def f(self, xf: NDArray):
        return np.array([xf[1], xf[-3], xf[-1]])

    def f_df_stm(self, X: NDArray, period: float | None = None):
        if period is None:
            period = self.period
        x0 = self.get_x0(X)
        xstmIC = np.array([*x0, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom, (0.0, period / 2), xstmIC, self.int_tol, args=(self.mu,)
        )
        xf, stm = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        xf = np.array(xf)

        dF = self.DF(stm)
        f = self.f(xf)

        G = np.diag([1, -1, 1, -1, 1, -1])
        Omega = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        I = np.identity(3)
        O = np.zeros((3, 3))
        mtx1 = np.block([[O, -I], [I, -2 * Omega]])
        mtx2 = np.block([[-2 * Omega, I], [-I, O]])
        stm_full = G @ mtx1 @ stm.T @ mtx2 @ G @ stm
        return f, dF, stm_full


class planar_perpendicular(Targetter):
    def __init__(self, int_tol: float, mu: float = muEM):
        self.int_tol = int_tol
        self.mu = mu

    def get_X(self, x0: NDArray, period: float):
        return np.array([x0[0], x0[-2], period / 2])

    def get_x0(self, X: NDArray):
        return np.array([X[0], 0, 0, 0, X[1], 0])

    def get_period(self, X: NDArray):
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
        period = self.get_period(X)
        xstmIC = np.array([*x0, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom, (0.0, period / 2), xstmIC, self.int_tol, args=(self.mu,)
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


class multi_shooter(Targetter):
    # problem: currently super under constrained (6N-1 x 7N-1). I could make it work by forcing segments to be equal duration?
    def __init__(
        self,
        N_segments: int = 2,
        int_tol: float = 1e-11,
        mu: float = muEM,
    ):
        self.int_tol = int_tol
        self.mu = mu
        self.nseg = N_segments
        assert self.nseg > 1

    def get_X(self, x0: NDArray, period: float):
        # Propagate with arbitrary timesteps
        ts, xs, (_, Fs), _ = dop853(
            eom, (0.0, period), x0, self.int_tol, args=(self.mu,), dense_output=True
        )
        # ensure that num timesteps is divisible by num segments
        te, xe = dop_interpolate(ts, xs.T, Fs, n_mult=self.nseg)
        max_ind = len(ts) - 1  # maximum index to new list
        Xsi = []
        for jj in range(self.nseg):
            # get indices to access
            i0 = jj * max_ind
            i1 = (jj + 1) * max_ind

            x0i = xe[:, i0]
            dti = te[i1] - te[i0] if jj < self.nseg - 1 else te[-1] - te[i0]
            Xsi.append(np.append(x0i, dti))
        return np.concat(Xsi)

    def get_x0_segment(self, X: NDArray, segment: int):
        # NOTE: segment is 0-indexed
        i0 = 7 * (segment)
        i1 = 7 * (segment + 1)
        Xsegment = X[i0:i1].copy()
        states = Xsegment[:-1]  # cut off time
        return states

    def get_x0s(self, X: NDArray):
        x0s = []
        for segment in range(self.nseg):
            x0s.append(self.get_x0_segment(X, segment))
        return x0s

    def get_dt_segment(self, X: NDArray, segment: int):
        # NOTE: segment is 0-indexed
        i0 = 7 * (segment)  # Index of control variable to start
        i1 = 7 * (segment + 1)
        Xsegment = X[i0:i1].copy()
        return Xsegment[-1]

    def get_x0(self, X: NDArray):
        return self.get_x0_segment(X, 0)

    def get_dts(self, X: NDArray):
        dts = []
        for segment in range(self.nseg):
            dts.append(self.get_dt_segment(X, segment))
        return dts

    def get_period(self, X: NDArray):
        return sum(self.get_dts(X))

    def f(self, x0_segments: Tuple[NDArray], xf_segments: Tuple[NDArray]):
        assert len(xf_segments) == len(x0_segments) == self.nseg
        # f[j] = xf[j] - x0[j+1] is num segments
        f_segments = []
        for jj in range(self.nseg):
            ind_xf = jj
            ind_x0 = (jj + 1) % self.nseg
            state_diff = xf_segments[ind_xf] - x0_segments[ind_x0]
            f_segments.append(state_diff)
        return np.concat(f_segments)

    def DF(
        self,
        stm_segments: Tuple[NDArray] | List[NDArray],
        eomf_segments: Tuple[NDArray] | List[NDArray],
    ):
        dF = np.zeros((6 * self.nseg, 7 * self.nseg), np.float64)
        for jj in range(self.nseg):
            ind0_xf = 6 * jj
            ind0_x0 = 7 * jj
            ind1_xf = 6 * jj + 6
            ind1_x0 = 7 * jj + 7
            stm_part = np.hstack((stm_segments[jj], eomf_segments[jj][:, None]))
            dF[ind0_xf:ind1_xf, ind0_x0:ind1_x0] = stm_part
            eyestart = 7 * ((jj + 1) % self.nseg)
            eyeend = 7 * ((jj + 1) % self.nseg) + 6
            dF[ind0_xf:ind1_xf, eyestart:eyeend] = -np.eye(6)

        return dF

    def f_df_stm(self, X: NDArray):
        x0s = self.get_x0s(X)
        tfs = self.get_dts(X)
        xfs = []
        stms = []
        eomfs = []
        stm_full = np.eye(6)
        for x0, tf in zip(x0s, tfs):
            sv0 = np.array([*x0, *np.eye(6).flatten()])
            ts, ys, _, _ = dop853(
                coupled_stm_eom, (0.0, tf), sv0, self.int_tol, args=(self.mu,)
            )
            xf, stm = ys[:6, -1], ys[6:, -1].reshape(6, 6)
            xf = np.array(xf)
            eomf = eom(ts[-1], xf, self.mu)
            stms.append(stm)
            eomfs.append(eomf)
            xfs.append(xf)
            stm_full @= stm

        dF = self.DF(stms, eomfs)
        f = self.f(x0s, xfs)
        return f, dF, stm_full


class multi_shooter_minus_one(Targetter):
    # problem: currently super under constrained (6N-1 x 7N-1). I could make it work by forcing segments to be equal duration?
    # Some ideas to fully constrain:
    # - minimize the norm squared of segment-dt variables (optimal, not sure how the partials work out)
    #    - I think I can do (null - proj(null, dt-variables)) I could get what I want.
    # - Equal dt for each one (easy partials, sub-optimal)
    def __init__(
        self,
        index_fixed: int,
        index_no_enforce: int,
        value_fixed: float,
        N_segments: int = 2,
        int_tol: float = 1e-11,
        mu: float = muEM,
    ):
        self.int_tol = int_tol
        self.mu = mu
        self.ind_fixed = index_fixed
        self.state_val = value_fixed
        self.ind_skip = index_no_enforce
        self.nseg = N_segments
        assert 0 <= self.ind_fixed < 6
        assert 0 <= self.ind_skip < 6
        assert self.nseg > 1

    def get_X(self, x0: NDArray, period: float):
        # Propagate with arbitrary timesteps
        ts, xs, (_, Fs), _ = dop853(
            eom, (0.0, period), x0, self.int_tol, args=(self.mu,), dense_output=True
        )
        # ensure that num timesteps is divisible by num segments
        te, xe = dop_interpolate(ts, xs.T, Fs, n_mult=self.nseg)
        max_ind = len(ts) - 1  # maximum index to new list
        Xsi = []
        for jj in range(self.nseg):
            # get indices to access
            i0 = jj * max_ind
            i1 = (jj + 1) * max_ind

            x0i = xe[:, i0]
            if jj == 0:
                x0i = np.delete(x0i, self.ind_fixed)
            dti = te[i1] - te[i0] if jj < self.nseg - 1 else te[-1] - te[i0]
            Xsi.append(np.append(x0i, dti))
        return np.concat(Xsi)

    def get_x0_segment(self, X: NDArray, segment: int):
        # NOTE: segment is 0-indexed
        i0 = (
            7 * (segment) - 1 if segment > 0 else 0
        )  # Index of control variable to start
        i1 = 7 * (segment + 1) - 1  # idx of control variable to end (X0 is one smaller)
        Xsegment = X[i0:i1].copy()
        states = Xsegment[:-1]  # cut off time
        if segment == 0:
            states = np.insert(states, self.ind_fixed, self.state_val)
        return states

    def get_x0s(self, X: NDArray):
        x0s = []
        for segment in range(self.nseg):
            x0s.append(self.get_x0_segment(X, segment))
        return x0s

    def get_dt_segment(self, X: NDArray, segment: int):
        # NOTE: segment is 0-indexed
        i0 = 7 * (segment)  # Index of control variable to start
        i1 = 7 * (segment + 1) - 1  # idx of control variable to end (X0 is one smaller)
        Xsegment = X[i0:i1].copy()
        return Xsegment[-1]

    def get_x0(self, X: NDArray):
        return self.get_x0_segment(X, 0)

    def get_dts(self, X: NDArray):
        dts = []
        for segment in range(self.nseg):
            dts.append(self.get_dt_segment(X, segment))
        return dts

    def get_period(self, X: NDArray):
        return sum(self.get_dts(X))

    def f(self, x0_segments: Tuple[NDArray], xf_segments: Tuple[NDArray]):
        assert len(xf_segments) == len(x0_segments) == self.nseg
        # f[j] = xf[j] - x0[j+1] except where j+1 is num segments
        f_segments = []
        for jj in range(self.nseg):
            ind_xf = jj
            ind_x0 = (jj + 1) % self.nseg
            state_diff = xf_segments[ind_xf] - x0_segments[ind_x0]
            if jj == self.nseg - 1:
                state_diff = np.delete(state_diff, self.ind_skip)
            f_segments.append(state_diff)
        return np.concat(f_segments)

    def DF(
        self,
        stm_segments: Tuple[NDArray] | List[NDArray],
        eomf_segments: Tuple[NDArray] | List[NDArray],
    ):
        dF = np.zeros((6 * self.nseg, 7 * self.nseg), np.float64)
        for jj in range(self.nseg):
            ind0_xf = 6 * jj
            ind0_x0 = 7 * jj
            ind1_xf = 6 * jj + 6
            ind1_x0 = 7 * jj + 7
            stm_part = np.hstack((stm_segments[jj], eomf_segments[jj][:, None]))
            dF[ind0_xf:ind1_xf, ind0_x0:ind1_x0] = stm_part
            eyestart = 7 * ((jj + 1) % self.nseg)
            eyeend = 7 * ((jj + 1) % self.nseg) + 6
            dF[ind0_xf:ind1_xf, eyestart:eyeend] = -np.eye(6)

        dF = np.delete(dF, self.ind_fixed, 1)
        dF = np.delete(dF, self.ind_skip + 6 * (self.nseg - 1), 0)
        return dF

    def f_df_stm(self, X: NDArray):
        x0s = self.get_x0s(X)
        tfs = self.get_dts(X)
        xfs = []
        stms = []
        eomfs = []
        stm_full = np.eye(6)
        for x0, tf in zip(x0s, tfs):
            sv0 = np.array([*x0, *np.eye(6).flatten()])
            ts, ys, _, _ = dop853(
                coupled_stm_eom, (0.0, tf), sv0, self.int_tol, args=(self.mu,)
            )
            xf, stm = ys[:6, -1], ys[6:, -1].reshape(6, 6)
            xf = np.array(xf)
            eomf = eom(ts[-1], xf, self.mu)
            stms.append(stm)
            eomfs.append(eomf)
            xfs.append(xf)
            stm_full @= stm

        dF = self.DF(stms, eomfs)
        f = self.f(x0s, xfs)
        return f, dF, stm_full


class multi_shooter_eq_time(Targetter):
    def __init__(
        self,
        index_fixed: int,
        index_no_enforce: int,
        value_fixed: float,
        N_segments: int = 2,
        int_tol: float = 1e-11,
        mu: float = muEM,
    ):
        self.int_tol = int_tol
        self.mu = mu
        self.ind_fixed = index_fixed
        self.state_val = value_fixed
        self.ind_skip = index_no_enforce
        self.nseg = N_segments
        assert 0 <= self.ind_fixed < 6
        assert 0 <= self.ind_skip < 6
        assert self.nseg > 1

    def get_X(self, x0: NDArray, period: float):
        te = np.linspace(0.0, period, self.nseg + 1)
        ts, xs, _, _ = dop853(
            eom, (0.0, period), x0, self.int_tol, args=(self.mu,), t_eval=te
        )
        Xsi = []
        for jj in range(self.nseg):
            i0 = jj
            i1 = jj + 1
            x0i = xs[:, i0]
            if jj == 0:
                x0i = np.delete(x0i, self.ind_fixed)
            dti = ts[i1] - ts[i0]
            Xsi.append(np.append(x0i, dti))
        return np.concat(Xsi)

    def get_x0_segment(self, X: NDArray, segment: int):
        # NOTE: segment is 0-indexed
        i0 = (
            7 * (segment) - 1 if segment > 0 else 0
        )  # Index of control variable to start
        i1 = 7 * (segment + 1) - 1  # idx of control variable to end (X0 is one smaller)
        Xsegment = X[i0:i1].copy()
        states = Xsegment[:-1]  # cut off time
        if segment == 0:
            states = np.insert(states, self.ind_fixed, self.state_val)
        return states

    def get_x0s(self, X: NDArray):
        x0s = []
        for segment in range(self.nseg):
            x0s.append(self.get_x0_segment(X, segment))
        return x0s

    def get_dt_segment(self, X: NDArray, segment: int):
        # NOTE: segment is 0-indexed
        i0 = 7 * (segment)  # Index of control variable to start
        i1 = 7 * (segment + 1) - 1  # idx of control variable to end (X0 is one smaller)
        Xsegment = X[i0:i1].copy()
        return Xsegment[-1]

    def get_x0(self, X: NDArray):
        return self.get_x0_segment(X, 0)

    def get_dts(self, X: NDArray):
        dts = []
        for segment in range(self.nseg):
            dts.append(self.get_dt_segment(X, segment))
        return dts

    def get_period(self, X: NDArray):
        return sum(self.get_dts(X))

    def f(self, x0_segments: Tuple[NDArray], xf_segments: Tuple[NDArray], t_segments):
        assert len(xf_segments) == len(x0_segments) == self.nseg
        # f[j] = xf[j] - x0[j+1] except where j+1 is num segments
        f_segments = []
        for jj in range(self.nseg):
            ind_xf = jj
            ind_x0 = (jj + 1) % self.nseg
            state_diff = xf_segments[ind_xf] - x0_segments[ind_x0]
            if jj == self.nseg - 1:
                state_diff = np.delete(state_diff, self.ind_skip)
            f_segments.append(state_diff)
        f_segments.append(-np.diff(t_segments))  # current-next
        return np.concat(f_segments)

    def DF(
        self,
        stm_segments: Tuple[NDArray] | List[NDArray],
        eomf_segments: Tuple[NDArray] | List[NDArray],
    ):
        dF = np.zeros((7 * self.nseg - 1, 7 * self.nseg), np.float64)
        for jj in range(self.nseg):
            ind0_xf = 6 * jj
            ind0_x0 = 7 * jj
            ind1_xf = 6 * jj + 6
            ind1_x0 = 7 * jj + 7
            stm_part = np.hstack((stm_segments[jj], eomf_segments[jj][:, None]))
            dF[ind0_xf:ind1_xf, ind0_x0:ind1_x0] = stm_part
            eyestart = 7 * ((jj + 1) % self.nseg)
            eyeend = 7 * ((jj + 1) % self.nseg) + 6
            dF[ind0_xf:ind1_xf, eyestart:eyeend] = -np.eye(6)
            if jj < self.nseg - 1:
                dF[6 * self.nseg + jj, 7 * jj + 6] = 1
                dF[6 * self.nseg + jj, 7 * (jj + 1) + 6] = -1

        dF = np.delete(dF, self.ind_fixed, 1)
        dF = np.delete(dF, self.ind_skip + 6 * (self.nseg - 1), 0)
        return dF

    def f_df_stm(self, X: NDArray):
        x0s = self.get_x0s(X)
        tfs = self.get_dts(X)
        xfs = []
        stms = []
        eomfs = []
        stm_full = np.eye(6)
        for x0, tf in zip(x0s, tfs):
            sv0 = np.array([*x0, *np.eye(6).flatten()])
            ts, ys, _, _ = dop853(
                coupled_stm_eom, (0.0, tf), sv0, self.int_tol, args=(self.mu,)
            )
            xf, stm = ys[:6, -1], ys[6:, -1].reshape(6, 6)
            xf = np.array(xf)
            eomf = eom(ts[-1], xf, self.mu)
            stms.append(stm)
            eomfs.append(eomf)
            xfs.append(xf)
            stm_full @= stm

        dF = self.DF(stms, eomfs)
        f = self.f(x0s, xfs, tfs)
        return f, dF, stm_full


def propagate_X(
    targetter: Targetter, X: NDArray, fraction: float = 1 / 2, int_tol: float = 1e-13
):
    assert 0 < fraction < 1
    x0 = targetter.get_x0(X)
    period = targetter.get_period(X)
    tf = period * fraction
    ts, ys, _, _ = dop853(eom, (0.0, tf), x0, int_tol, args=(targetter.mu,))
    xf = ys[:, -1]
    return targetter.get_X(xf, period)


# %% Heteroclinic connections

# IDEA: go to the point they meet, backprop from one, foreprop from the other


# target full state contenuity
class heterclinic:
    def __init__(
        self,
        int_tol: float,
        # jacobi: float,
        x0_unstable: NDArray,
        x0_stable: NDArray,
        stm_unstable: NDArray,
        stm_stable: NDArray,
        eig_unstable: NDArray,
        eig_stable: NDArray,
        scale: float = 1.0,
        mu: float = muEM,
    ):
        self.int_tol = int_tol
        # self.JC_targ = jacobi
        self.mu = mu
        self.scale = scale
        self.orbit_x_s = x0_stable
        self.orbit_x_u = x0_unstable
        self.stm_s = stm_stable
        self.stm_u = stm_unstable
        self.eig_s = eig_stable
        self.eig_u = eig_unstable

    # X := [d, dt0, proptime] (u, s)
    # def get_X(
    #     self, d_s: float, t0_s: float, dt_s: float, d_u: float, t0_u: float, dt_u: float
    # ):
    #     return np.array([d_u, t0_s, dt_s, d_s, t0_u, dt_u])

    def get_eig_stable(self, X: NDArray):
        stepsize, dt0, tf = X[3:]
        orbit_x0 = self.orbit_x_s
        eig = self.eig_s
        sv0 = np.append(orbit_x0, np.eye(6).flatten())
        # find the new orbit point and STM
        sv_new = rk8_step(coupled_stm_eom, sv0, 0.0, dt0)
        stm = np.reshape(sv_new[6:], (6, 6))  # stm from t0 to dt0
        # monodromy_new = stm @ monodromy @ np.linalg.inv(stm)
        vec_new = stm @ eig  # get the new stepoff direction
        vec_new /= np.linalg.norm(vec_new)
        return vec_new

    def get_x0_stable(self, X: NDArray):
        stepsize, dt0, tf = X[3:]
        vec_new = self.get_eig_stable(X)
        orbit_x0 = self.orbit_x_s
        return orbit_x0 + vec_new * stepsize / self.scale

    def get_eig_unstable(self, X: NDArray):
        stepsize, dt0, tf = X[:3]
        orbit_x0 = self.orbit_x_u
        eig = self.eig_u
        sv0 = np.append(orbit_x0, np.eye(6).flatten())
        # find the new orbit point and STM
        sv_new = rk8_step(coupled_stm_eom, sv0, 0.0, dt0, args=(self.mu,))
        stm = np.reshape(sv_new[6:], (6, 6))  # stm from t0 to dt0
        # monodromy_new = stm @ monodromy @ np.linalg.inv(stm)
        vec_new = stm @ eig  # get the new stepoff direction
        vec_new /= np.linalg.norm(vec_new)
        return vec_new

    def get_x0_unstable(self, X: NDArray):
        stepsize, dt0, tf = X[:3]
        vec_new = self.get_eig_unstable(X)
        orbit_x0 = self.orbit_x_u
        return orbit_x0 + vec_new * stepsize / self.scale

    def get_tf_s(self, X: NDArray):
        return X[3:][-1]

    def get_tf_u(self, X: NDArray):
        return X[:3][-1]

    def DF(
        self,
        x0_u: NDArray,
        stm_u: NDArray,
        eom0_u: NDArray,
        eomf_u: NDArray,
        eig_u: NDArray,
        x0_s: NDArray,
        stm_s: NDArray,
        eom0_s: NDArray,
        eomf_s: NDArray,
        eig_s: NDArray,
    ):
        # X := [d, dt0, proptime] (u, s)

        # Jacobi constant derivatives
        dJC_s = np.append(JCgrad(x0_s), -2 * x0_s[3:])
        dJC_u = np.append(JCgrad(x0_u), -2 * x0_u[3:])

        # # derivative of JC with respect to stepoff distance
        dJC_s_d = np.dot(dJC_s, eig_s) / self.scale
        dJC_u_d = np.dot(dJC_u, eig_u) / self.scale

        # derivative of final state with respect to stepoff dist
        dxdd_s = stm_s @ eig_s / self.scale
        dxdd_u = stm_u @ eig_u / self.scale

        # derivative of final state with respect to start time (assuming stepoff stays identical, which it wont)
        dxd0_s = stm_s @ eom0_s
        dxd0_u = stm_u @ eom0_u

        dF = np.array(
            [
                [-dxdd_u[0], -dxd0_u[0], -eomf_u[0], dxdd_s[0], dxd0_s[0], eomf_s[0]],
                [-dxdd_u[1], -dxd0_u[1], -eomf_u[1], dxdd_s[1], dxd0_s[1], eomf_s[1]],
                [-dxdd_u[2], -dxd0_u[2], -eomf_u[2], dxdd_s[2], dxd0_s[2], eomf_s[2]],
                [-dxdd_u[3], -dxd0_u[3], -eomf_u[3], dxdd_s[3], dxd0_s[3], eomf_s[3]],
                [-dxdd_u[4], -dxd0_u[4], -eomf_u[4], dxdd_s[4], dxd0_s[4], eomf_s[4]],
                [-dxdd_u[5], -dxd0_u[5], -eomf_u[5], dxdd_s[5], dxd0_s[5], eomf_s[5]],
                [dJC_s_d, 0.0, 0.0, dJC_u_d, 0.0, 0.0],
            ]
        )
        return dF

    def f(self, xf_u: NDArray, xf_s: NDArray):
        # if JC_target is None:
        #     JC_target = self.JC_targ
        return np.array(xf_s - xf_u)

    def f_df_stm(self, X: NDArray):
        # X[[0, 3]] = 0
        # if JC_target is None:
        #     JC_target = self.JC_targ
        x0_s = self.get_x0_stable(X)
        x0_u = self.get_x0_unstable(X)
        tf_s = self.get_tf_s(X)
        tf_u = self.get_tf_u(X)
        sv0 = np.array([*x0_s, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom, (0.0, tf_s), sv0, self.int_tol, args=(self.mu,)
        )
        xf_s, stm_s = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        eomf_s = eom(tf_s, xf_s, self.mu)
        eom0_s = eom(0.0, x0_s, self.mu)

        sv0 = np.array([*x0_u, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom, (0.0, tf_u), sv0, self.int_tol, args=(self.mu,)
        )
        xf_u, stm_u = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        eomf_u = eom(tf_u, xf_u, self.mu)
        eom0_u = eom(0.0, x0_u, self.mu)

        eig_u = self.get_eig_unstable(X)
        eig_s = self.get_eig_stable(X)

        dF = self.DF(
            x0_u, stm_u, eom0_u, eomf_u, eig_u, x0_s, stm_s, eom0_s, eomf_s, eig_s
        )
        f = self.f(xf_u, xf_s)
        print(np.linalg.norm(f))

        return f, dF, None


class heterclinic_mod:
    def __init__(
        self,
        int_tol: float,
        jacobi: float,
        # x0_unstable: NDArray,
        # x0_stable: NDArray,
        mu: float = muEM,
    ):
        self.int_tol = int_tol
        self.jacobi = jacobi
        self.mu = mu

    # X := [state, proptime] (u, s)

    def get_x0_u(self, X: NDArray):
        return X[:7][:6]

    def get_x0_s(self, X: NDArray):
        return X[7:][:6]

    def get_tf_u(self, X: NDArray):
        return X[:7][-1]

    def get_tf_s(self, X: NDArray):
        return X[7:][-1]

    def DF(
        self,
        x0_u: NDArray,
        eomf_u: NDArray,
        stm_u: NDArray,
        x0_s: NDArray,
        eomf_s: NDArray,
        stm_s: NDArray,
    ):
        # X := [state, proptime] (u, s)

        # Jacobi constant derivatives
        # dJC_s = np.append(JCgrad(x0_s), -2 * x0_s[3:])
        dJC_u = np.append(JCgrad(x0_u.copy()), -2 * x0_u[3:])

        # df to enforce contenuity
        df_cont_u = np.concat((stm_u.copy(), eomf_u.reshape(-1, 1)), axis=1)
        df_cont_s = np.concat((stm_s.copy(), eomf_s.reshape(-1, 1)), axis=1)
        # add JC constraint (we only have one for unstable for now, may change later)
        dF_u = np.concat((df_cont_u, [np.append(dJC_u, 0)]), axis=0)
        dF_s = np.concat((-df_cont_s, [np.zeros(7)]), axis=0)

        # full DF
        dF = np.concat((dF_u, dF_s), axis=1)
        return dF  # np.delete(dF, [-1], 0)
        # return np.array([np.append(dJC_u, np.zeros(8))])

    def f(self, xf_u: NDArray, xf_s: NDArray, jacobi: float | None = None):
        if jacobi is None:
            jacobi = self.jacobi
        jc_u = jacobi_constant(xf_u.copy())
        out = np.append((xf_u - xf_s), (jc_u - jacobi))
        return out  # np.delete(out, [-1], 0)

    def f_df_stm(self, X: NDArray):
        x0_s = self.get_x0_s(X)
        x0_u = self.get_x0_u(X)
        tf_s = self.get_tf_s(X)
        tf_u = self.get_tf_u(X)
        # print(x0_s)
        sv0 = np.array([*x0_s, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom, (0.0, tf_s), sv0, self.int_tol, args=(self.mu,)
        )
        xf_s, stm_s = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        eomf_s = eom(tf_s, xf_s, self.mu)
        # eom0_s = eom(0.0, x0_s, self.mu)

        sv0 = np.array([*x0_u, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom, (0.0, tf_u), sv0, self.int_tol, args=(self.mu,)
        )
        xf_u, stm_u = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        eomf_u = eom(tf_u, xf_u, self.mu)
        # eom0_u = eom(0.0, x0_u, self.mu)

        dF = self.DF(x0_u, eomf_u, stm_u, x0_s, eomf_s, stm_s)
        f = self.f(xf_u, xf_s)
        print(np.linalg.norm(f))

        return f, dF, None


class manifold_reduced_dim:
    def __init__(
        self,
        int_tol: float,
        x0_unstable: NDArray,
        x0_stable: NDArray,
        eig_unstable: NDArray,
        eig_stable: NDArray,
        dscale: float = 100,
        mu: float = muEM,
    ):
        self.int_tol = int_tol
        self.mu = mu
        self.orbit_x_s = x0_stable
        self.orbit_x_u = x0_unstable
        self.eig_s = eig_stable
        self.eig_u = eig_unstable
        self.dscale = dscale

    # X := [d, proptime] (u, s)

    def get_x0_stable(self, X: NDArray):
        stepsize, tf = X[2:]
        return self.orbit_x_s + self.eig_s * stepsize / self.dscale

    def get_x0_unstable(self, X: NDArray):
        stepsize, tf = X[:2]
        return self.orbit_x_u + self.eig_u * stepsize / self.dscale

    def get_tf_s(self, X: NDArray):
        return X[2:][-1]

    def get_tf_u(self, X: NDArray):
        return X[:2][-1]

    def DF(self, stm_u: NDArray, eomf_u: NDArray, stm_s: NDArray, eomf_s: NDArray):

        dxdd_s = stm_s @ self.eig_s / self.dscale
        dxdd_u = stm_u @ self.eig_u / self.dscale

        dF = np.array(
            [
                [-dxdd_u[0], -eomf_u[0], dxdd_s[0], eomf_s[0]],
                [-dxdd_u[1], -eomf_u[1], dxdd_s[1], eomf_s[1]],
                [-dxdd_u[2], -eomf_u[2], dxdd_s[2], eomf_s[2]],
                [-dxdd_u[3], -eomf_u[3], dxdd_s[3], eomf_s[3]],
                [-dxdd_u[4], -eomf_u[4], dxdd_s[4], eomf_s[4]],
                [-dxdd_u[5], -eomf_u[5], dxdd_s[5], eomf_s[5]],
                # [dJC_s_d, 0.0, 0.0, dJC_u_d, 0.0, 0.0],
            ]
        )
        return dF

    def f(self, xf_u: NDArray, xf_s: NDArray):
        return np.array(xf_s - xf_u)

    def f_df_stm(self, X: NDArray):
        # if JC_target is None:
        #     JC_target = self.JC_targ
        x0_s = self.get_x0_stable(X)
        x0_u = self.get_x0_unstable(X)
        tf_s = self.get_tf_s(X)
        tf_u = self.get_tf_u(X)
        sv0 = np.array([*x0_s, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom, (0.0, tf_s), sv0, self.int_tol, args=(self.mu,)
        )
        xf_s, stm_s = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        eomf_s = eom(tf_s, xf_s, self.mu)
        eom0_s = eom(0.0, x0_s, self.mu)

        sv0 = np.array([*x0_u, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom, (0.0, tf_u), sv0, self.int_tol, args=(self.mu,)
        )
        xf_u, stm_u = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        eomf_u = eom(tf_u, xf_u, self.mu)
        eom0_u = eom(0.0, x0_u, self.mu)

        dF = self.DF(stm_u, eomf_u, stm_s, eomf_s)  # [[0, 1, -3, -2]]
        f = self.f(xf_u, xf_s)
        print(np.linalg.norm(f))
        return f, dF, None


class manifold_higher_dim:
    def __init__(
        self,
        x0_unstable: NDArray,
        eig_u1: NDArray,
        eig_u2: NDArray,
        x0_stable: NDArray,
        eig_s1: NDArray,
        eig_s2: NDArray,
        t_s: float,
        mu: float = muEM,
        int_tol: float = 1e-11,
    ):
        self.int_tol = int_tol
        self.mu = mu
        self.orbit_x_s = x0_stable
        self.orbit_x_u = x0_unstable
        self.eig_s1 = eig_s1
        self.eig_s2 = eig_s2
        self.eig_u1 = eig_u1
        self.eig_u2 = eig_u2
        self.t_s = t_s

    # X := [d1, d2, proptime] (u, s)

    def get_x0_stable(self, X: NDArray):
        d1, d2 = X[3:]
        return self.orbit_x_s + self.eig_s1 * d1 + self.eig_s2 * d2

    def get_x0_unstable(self, X: NDArray):
        d1, d2, tf = X[:3]
        return self.orbit_x_u + self.eig_u1 * d1 + self.eig_u2 * d2

    def get_tf_s(self, X: NDArray):
        # return X[3:][-1]
        return self.t_s

    def get_tf_u(self, X: NDArray):
        return X[:3][-1]

    def DF(self, stm_u: NDArray, eomf_u: NDArray, stm_s: NDArray, eomf_s: NDArray):

        dxdd_s1 = stm_s @ self.eig_s1
        dxdd_u1 = stm_u @ self.eig_u1
        dxdd_s2 = stm_s @ self.eig_s2
        dxdd_u2 = stm_u @ self.eig_u2

        dF_u = -np.array([dxdd_u1, dxdd_u2, eomf_u]).T
        # dF_s = np.array([dxdd_s1, dxdd_s2, eomf_s]).T
        dF_s = np.array([dxdd_s1, dxdd_s2]).T

        dF = np.hstack((dF_u, dF_s))
        return dF

    def f(self, xf_u: NDArray, xf_s: NDArray):
        return np.array(xf_s - xf_u)

    def f_df_stm(self, X: NDArray):
        # if JC_target is None:
        #     JC_target = self.JC_targ
        x0_s = self.get_x0_stable(X)
        x0_u = self.get_x0_unstable(X)
        tf_s = self.get_tf_s(X)
        tf_u = self.get_tf_u(X)
        sv0 = np.array([*x0_s, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom, (0.0, tf_s), sv0, self.int_tol, args=(self.mu,)
        )
        xf_s, stm_s = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        eomf_s = eom(tf_s, xf_s, self.mu)
        # eom0_s = eom(0.0, x0_s, self.mu)

        sv0 = np.array([*x0_u, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom, (0.0, tf_u), sv0, self.int_tol, args=(self.mu,)
        )
        xf_u, stm_u = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        eomf_u = eom(tf_u, xf_u, self.mu)
        # eom0_u = eom(0.0, x0_u, self.mu)

        dF = self.DF(stm_u, eomf_u, stm_s, eomf_s)  # [[0, 1, -3, -2]]
        f = self.f(xf_u, xf_s)
        print(np.linalg.norm(f))
        return f, dF, None
