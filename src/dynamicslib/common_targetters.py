import numpy as np
from dynamicslib.common import muEM, jacobi_constant, coupled_stm_eom, eom, JCgrad
from dynamicslib.integrate import dop853, rk8_step
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
        ts, ys, _, _ = dop853(
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
            coupled_stm_eom,
            (0.0, tf_s),
            sv0,
            self.int_tol,
            self.int_tol,
            args=(self.mu,),
        )
        xf_s, stm_s = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        eomf_s = eom(tf_s, xf_s, self.mu)
        eom0_s = eom(0.0, x0_s, self.mu)

        sv0 = np.array([*x0_u, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom,
            (0.0, tf_u),
            sv0,
            self.int_tol,
            self.int_tol,
            args=(self.mu,),
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
            coupled_stm_eom,
            (0.0, tf_s),
            sv0,
            self.int_tol,
            self.int_tol,
            args=(self.mu,),
        )
        xf_s, stm_s = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        eomf_s = eom(tf_s, xf_s, self.mu)
        # eom0_s = eom(0.0, x0_s, self.mu)

        sv0 = np.array([*x0_u, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom,
            (0.0, tf_u),
            sv0,
            self.int_tol,
            self.int_tol,
            args=(self.mu,),
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
            coupled_stm_eom,
            (0.0, tf_s),
            sv0,
            self.int_tol,
            self.int_tol,
            args=(self.mu,),
        )
        xf_s, stm_s = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        eomf_s = eom(tf_s, xf_s, self.mu)
        eom0_s = eom(0.0, x0_s, self.mu)

        sv0 = np.array([*x0_u, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom,
            (0.0, tf_u),
            sv0,
            self.int_tol,
            self.int_tol,
            args=(self.mu,),
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
            coupled_stm_eom,
            (0.0, tf_s),
            sv0,
            self.int_tol,
            self.int_tol,
            args=(self.mu,),
        )
        xf_s, stm_s = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        eomf_s = eom(tf_s, xf_s, self.mu)
        # eom0_s = eom(0.0, x0_s, self.mu)

        sv0 = np.array([*x0_u, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom,
            (0.0, tf_u),
            sv0,
            self.int_tol,
            self.int_tol,
            args=(self.mu,),
        )
        xf_u, stm_u = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        eomf_u = eom(tf_u, xf_u, self.mu)
        # eom0_u = eom(0.0, x0_u, self.mu)

        dF = self.DF(stm_u, eomf_u, stm_s, eomf_s)  # [[0, 1, -3, -2]]
        f = self.f(xf_u, xf_s)
        print(np.linalg.norm(f))
        return f, dF, None


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
        ts, ys, _, _ = dop853(
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


class fullstate_minus_one:
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
        assert -6 <= self.ind_fixed < 6
        assert -6 <= self.ind_skip < 6

    def get_X(self, x0: NDArray, tf: float):
        return np.append(np.delete(x0, self.ind_fixed), tf)

    def get_x0(self, X: NDArray):
        states = np.array(X[:-1])
        return np.insert(states, self.ind_fixed, self.state_val)

    def get_tf(self, X: NDArray):
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
        tf = self.get_tf(X)
        xstmIC = np.array([*x0, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
            coupled_stm_eom,
            (0.0, tf),
            xstmIC,
            self.int_tol,
            self.int_tol,
            args=(self.mu,),
        )
        xf, stm = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        xf = np.array(xf)
        eomf = eom(ts[-1], xf, self.mu)

        dF = self.DF(stm, eomf)
        f = self.f(x0, xf)
        return f, dF, stm


class xy_symmetric:
    def __init__(self, int_tol: float, mu: float = muEM):
        self.int_tol = int_tol
        self.mu = mu

    def get_X(self, x0: NDArray, tf: float):
        return np.array([x0[0], x0[1], x0[-3], x0[-2], x0[-1], tf / 2])

    def get_x0(self, X: NDArray):
        return np.array([X[0], X[1], 0, X[2], X[3], X[4]])

    def get_tf(self, X: NDArray):
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
        tf = self.get_tf(X)
        xstmIC = np.array([*x0, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
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
        f = self.f(x0, xf)

        G = np.diag([1, -1, 1, -1, 1, -1])
        Omega = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        I = np.identity(3)
        O = np.zeros((3, 3))
        mtx1 = np.block([[O, -I], [I, -2 * Omega]])
        mtx2 = np.block([[-2 * Omega, I], [-I, O]])
        stm_full = G @ mtx1 @ stm.T @ mtx2 @ G @ stm
        return f, dF, stm_full


class spatial_period_fixed:
    def __init__(self, int_tol: float, period: float, mu: float = muEM):
        self.int_tol = int_tol
        self.mu = mu
        self.period = period

    def get_X(self, x0: NDArray):
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

    def f(self, xf: NDArray):
        return np.array([xf[1], xf[-3], xf[-1]])

    def f_df_stm(self, X: NDArray):
        x0 = self.get_x0(X)
        tf = self.period
        xstmIC = np.array([*x0, *np.eye(6).flatten()])
        ts, ys, _, _ = dop853(
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


class period_fixed_spatial_perpendicular:
    def __init__(self, period: float, int_tol: float, mu: float = muEM):
        self.int_tol = int_tol
        self.mu = mu
        self.period = period

    def get_X(self, x0: NDArray, tf: float):
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
            coupled_stm_eom,
            (0.0, period / 2),
            xstmIC,
            self.int_tol,
            self.int_tol,
            args=(self.mu,),
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
        ts, ys, _, _ = dop853(
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
