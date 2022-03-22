import numpy as np
import tensorflow as tf

class DataSet:
    def __init__(self, x_range, y_range, t_range, Nx, Ny, Nt, N_bc):
        self.x_range = x_range
        self.y_range = y_range
        self.t_range = t_range
        self.Nx = Nx
        self.Ny = Ny
        self.Nt = Nt
        self.N_bc = N_bc
        self.e = np.array([[0., 0.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 1.], [-1., 1.], [-1., -1.], [1., -1.]])
        self.w = np.full((9, 1), 0.0)
        self.w[0] = 4 / 9
        self.w[1:5] = 1 / 9
        self.w[5:] = 1 / 36
        self.RT = 100
        self.xi = self.e * np.sqrt(3 * self.RT)
        self.nu = 0.01
        self.tau = self.nu / self.RT
        self.sess = tf.Session()
        self.x_l = self.x_range.min()
        self.x_u = self.x_range.max()
        self.y_l = self.y_range.min()
        self.y_u = self.y_range.max()

    def feq_gradient(self, rou, u, v, x, y, t):

        rou_x = tf.gradients(rou, x)[0]
        u_x = tf.gradients(u, x)[0]
        v_x = tf.gradients(v, x)[0]

        rou_y = tf.gradients(rou, y)[0]
        u_y = tf.gradients(u, y)[0]
        v_y = tf.gradients(v, y)[0]

        rou_t = tf.gradients(rou, t)[0]
        u_t = tf.gradients(u, t)[0]
        v_t = tf.gradients(v, t)[0]
        f_sum = self.feq_xy(rou, u, v, rou_x, rou_y, rou_t, u_x, v_x, u_y, v_y, u_t, v_t)
        return f_sum

    # concat of dfeq / dX
    def feq_xy(self, rou, u, v, rou_x, rou_y, rou_t, u_x, v_x, u_y, v_y, u_t, v_t):
        f_sum = self.dfeq_xy(rou, u, v, rou_x, rou_y, rou_t, u_x, v_x, u_y, v_y, u_t, v_t, 0)
        for i in range(1, 9):
            f_ = self.dfeq_xy(rou, u, v, rou_x, rou_y, rou_t, u_x, v_x, u_y, v_y, u_t, v_t, i)
            f_sum = tf.concat([f_sum, f_], 1)
        return f_sum

    # difference of f_eq for x, y
    def dfeq_xy(self, rou, u, v, rou_x, rou_y, rou_t, u_x, v_x, u_y, v_y, u_t, v_t, i):
        feq_x = self.w[i, :] * rou_x * (1 + (self.xi[i, 0] * u + self.xi[i, 1] * v) / self.RT + (self.xi[i, 0] * u + self.xi[i, 1] * v) ** 2 / 2 / self.RT ** 2 - (u ** 2 + v ** 2) / 2 / self.RT) + \
                self.w[i, :] * rou * ((self.xi[i, 0] * u_x + self.xi[i, 1] * v_x) / self.RT + (self.xi[i, 0] * u + self.xi[i, 1] * v) * (self.xi[i, 0] * u_x + self.xi[i, 1] * v_x) / self.RT ** 2 - (u * u_x + v * v_x) / self.RT)
        # here need to change the equations
        feq_y = self.w[i, :] * rou_y * (1 + (self.xi[i, 0] * u + self.xi[i, 1] * v) / self.RT + (self.xi[i, 0] * u + self.xi[i, 1] * v) ** 2 / 2 / self.RT ** 2 - (u ** 2 + v ** 2) / 2 / self.RT) + \
                self.w[i, :] * rou * ((self.xi[i, 0] * u_y + self.xi[i, 1] * v_y) / self.RT + (self.xi[i, 0] * u + self.xi[i, 1] * v) * (self.xi[i, 0] * u_y + self.xi[i, 1] * v_y) / self.RT ** 2 - (u * u_y + v * v_y) / self.RT)

        feq_t = self.w[i, :] * rou_t * (1 + (self.xi[i, 0] * u + self.xi[i, 1] * v) / self.RT + (self.xi[i, 0] * u + self.xi[i, 1] * v) ** 2 / 2 / self.RT ** 2 - (u ** 2 + v ** 2) / 2 / self.RT) + \
                self.w[i, :] * rou * ((self.xi[i, 0] * u_t + self.xi[i, 1] * v_t) / self.RT + (self.xi[i, 0] * u + self.xi[i, 1] * v) * (self.xi[i, 0] * u_t + self.xi[i, 1] * v_t) / self.RT ** 2 - (u * u_t + v * v_t) / self.RT)

        dfeq_xy = self.xi[i, 0] * feq_x + self.xi[i, 1] * feq_y + feq_t
        return dfeq_xy

    # concat of f_eq_i
    def f_eq(self, rou, u, v):
        f_eq_sum = self.f_eqk(rou, u, v, 0)
        for i in range(1, 9):
            f_eq = self.f_eqk(rou, u, v, i)
            f_eq_sum = tf.concat([f_eq_sum, f_eq], 1)
        return f_eq_sum

    # f_eq equation
    def f_eqk(self, rou, u, v, k):
        f_eqk = self.w[k, :] * rou * (1 + (self.xi[k, 0]*u + self.xi[k, 1]*v) / self.RT + (self.xi[k, 0]*u + self.xi[k, 1]*v) ** 2 / 2 / self.RT ** 2 - (u*u + v*v) / 2 / self.RT)
        return f_eqk

    # the mean pde
    def bgk(self, f_neq, rou, u, v, x, y, t):
        feq_pre = self.feq_gradient(rou, u, v, x, y, t)
        R_sum = 0
        for k in range(9):
            fneq_x = tf.gradients(f_neq[:, k][:, None], x)[0]
            fneq_y = tf.gradients(f_neq[:, k][:, None], y)[0]
            fneq_t = tf.gradients(f_neq[:, k][:, None], t)[0]
            R = (fneq_t + self.xi[k, 0] * fneq_x + self.xi[k, 1] * fneq_y + feq_pre[:, k][:, None] + 1 / self.tau * (f_neq[:, k][:, None])) ** 2
            R_sum = R_sum + R
        return R_sum

    # the equation residual
    def Eq_res(self, f_neq, rou, u, v, x, y, t):
        feq_pre = self.feq_gradient(rou, u, v, x, y, t)
        Eq_sum = x * 0
        for k in range(9):
            fneq_x = tf.gradients(f_neq[:, k][:, None], x)[0]
            fneq_y = tf.gradients(f_neq[:, k][:, None], y)[0]
            fneq_t = tf.gradients(f_neq[:, k][:, None], t)[0]
            Eq = tf.abs(fneq_t + self.xi[k, 0] * fneq_x + self.xi[k, 1] * fneq_y + feq_pre[:, k][:, None] + 1 / self.tau * (f_neq[:, k][:, None]))
            Eq_sum = tf.concat([Eq_sum, Eq], 1)
        return Eq_sum[:, 1:]

    # boundary condition
    def inward_judge(self, x, y):
        x = tf.where(tf.equal(x, 2.0), x * 0 - 3.0, x)
        x = tf.where(tf.equal(x, -0.5), x * 0 + 3.0, x)
        x = tf.where(tf.equal(tf.abs(x), 3.0), x / 3.0, x * 0.0)
        y = tf.where(tf.equal(y, 1.5), y * 0 - 3.0, y)
        y = tf.where(tf.equal(y, -0.5), y * 0 + 3.0, y)
        y = tf.where(tf.equal(tf.abs(y), 3.0), y / 3.0, y * 0.0)
        return x, y

    def bgk_cond(self, f_neq, rou, u, v, x, y, t):
        feq_pre = self.feq_gradient(rou, u, v, x, y, t)
        R_sum = 0
        for k in range(9):
            fneq_x = tf.gradients(f_neq, x)[0]
            fneq_y = tf.gradients(f_neq, y)[0]
            fneq_t = tf.gradients(f_neq, t)[0]
            R = (fneq_t + self.xi[k, 0] * fneq_x + self.xi[k, 1] * fneq_y + feq_pre[:, k][:, None] + 1 / self.tau * (f_neq[:, k][:, None])) ** 2
            R_sum = R_sum + R
        return R_sum

    def fBC(self, f_neq, rou, u, v, x_bc, y_bc, t_bc):
        feq_ex = self.Ex_fneq_(rou, u, v, x_bc, y_bc, t_bc)
        fbc_sum = 0
        for i in range(9):
            f = (f_neq[:, i][:, None] + self.tau * feq_ex[:, i][:, None]) ** 2
            fbc_sum = fbc_sum + f
        return fbc_sum

    def u_train(self, x, y, t):
        u = - np.cos(x) * np.sin(y) * np.exp(-2 * t * self.nu)
        return u
    def v_train(self, x, y, t):
        v = np.sin(x) * np.cos(y) * np.exp(-2 * t * self.nu)
        return v

    def p_func(self, x, y, t):
        p = -0.25 * (np.cos(2 * x) + np.cos(2 * y)) * np.exp(-4 * t * self.nu) + self.RT
        return p

    def rou_func(self, x, y, t):
        rou = self.p_func(x, y, t) / self.RT
        return rou

    def Ex_fneq_(self, rou, u, v, x, y, t):
        rou_x = (0.5 * np.sin(2 * x) * np.exp(-4 * t * self.nu)) / self.RT
        rou_y = (0.5 * np.sin(2 * y) * np.exp(-4 * t * self.nu)) / self.RT
        rou_t = ((np.cos(2 * x) + np.cos(2 * y)) * np.exp(-4 * t * self.nu)) / self.RT
        u_x = np.sin(x) * np.sin(y) * np.exp(-2 * t * self.nu)
        u_y = -np.cos(x) * np.cos(y) * np.exp(-2 * t * self.nu)
        u_t = 2 * self.nu * np.cos(x) * np.sin(y) * np.exp(-2 * t * self.nu)
        v_x = np.cos(x) * np.cos(y) * np.exp(-2 * t * self.nu)
        v_y = -np.sin(x) * np.sin(y) * np.exp(-2 * t * self.nu)
        v_t = -2 * self.nu * np.sin(x) * np.cos(y) * np.exp(-2 * t * self.nu)
        f_sum = self.feq_xy(rou, u, v, rou_x, rou_y, rou_t, u_x, v_x, u_y, v_y, u_t, v_t)
        f_sum = tf.cast(f_sum, dtype=tf.float32)
        return f_sum

    def Ex_func(self, x_star, y_star, t_star):
        u = self.u_train(x_star, y_star, t_star)
        v = self.v_train(x_star, y_star, t_star)
        rou = self.rou_func(x_star, y_star, t_star)
        f_eq = self.f_eq(rou, u, v)
        # excat gradient need to change
        f_neq = -self.tau * (self.Ex_fneq_(rou, u, v, x_star, y_star, t_star))
        f_neq = tf.cast(f_neq, dtype=tf.float64)
        f_i = f_neq + f_eq
        # tensor change to array
        f_eq = f_eq.eval(session=self.sess)
        f_neq = f_neq.eval(session=self.sess)
        f_i = f_i.eval(session=self.sess)

        return u, v, f_eq, f_neq, f_i

    def Data_Generation(self):
        x_l = self.x_range.min()
        x_u = self.x_range.max()
        y_l = self.y_range.min()
        y_u = self.y_range.max()
        t_l = self.t_range.min()
        t_u = self.t_range.max()

        # domain data
        x_data = np.random.random((16000, 1)) * (x_u - x_l) + x_l
        y_data = np.random.random((16000, 1)) * (y_u - y_l) + y_l
        t_data = np.random.random((16000, 1)) * (t_u - t_l) + t_l

        # initial condition data
        x_ini = np.linspace(self.x_range[0], self.x_range[1], self.Nx)
        y_ini = np.linspace(self.y_range[0], self.y_range[1], self.Ny)
        x_ini, y_ini = np.meshgrid(x_ini, y_ini)
        x_ini = np.ravel(x_ini).T[:, None]
        y_ini = np.ravel(y_ini).T[:, None]
        t_ini = np.zeros_like(x_ini)

        # boundary condition data
        """x_1 = (x_u - x_l) * np.random.random((300, 1)) + x_l
        x_2 = (x_u - x_l) * np.random.random((300, 1)) + x_l
        x_3 = np.full((300, 1), -0.5)
        x_4 = np.full((300, 1), 2)
        y_1 = np.full((300, 1), -0.5)
        y_2 = np.full((300, 1), 1.5)
        y_3 = (y_u - y_l) * np.random.random((300, 1)) + y_l
        y_4 = (y_u - y_l) * np.random.random((300, 1)) + y_l
        x_b = np.vstack((x_1, x_2, x_3, x_4))
        y_b = np.vstack((y_1, y_2, y_3, y_4))"""
        # y = pi
        x_1 = np.linspace(self.x_range[0], self.x_range[1], self.N_bc)
        t_1 = np.linspace(self.t_range[0], self.t_range[1], self.N_bc)
        x_1, t_1 = np.meshgrid(x_1, t_1)
        x_1 = np.ravel(x_1).T[:, None]
        t_1 = np.ravel(t_1).T[:, None]
        y_1 = np.ones_like(x_1) * np.pi
        # y = - pi
        x_2 = np.linspace(self.x_range[0], self.x_range[1], self.N_bc)
        t_2 = np.linspace(self.t_range[0], self.t_range[1], self.N_bc)
        x_2, t_2 = np.meshgrid(x_2, t_2)
        x_2 = np.ravel(x_2).T[:, None]
        t_2 = np.ravel(t_2).T[:, None]
        y_2 = np.ones_like(x_2) * (-np.pi)
        # x = - pi
        y_3 = np.linspace(self.y_range[0], self.y_range[1], self.N_bc)
        t_3 = np.linspace(self.t_range[0], self.t_range[1], self.N_bc)
        y_3, t_3 = np.meshgrid(y_3, t_3)
        y_3 = np.ravel(y_3).T[:, None]
        t_3 = np.ravel(t_3).T[:, None]
        x_3 = np.ones_like(y_3) * (-np.pi)
        # x = pi
        y_4 = np.linspace(self.y_range[0], self.y_range[1], self.N_bc)
        t_4 = np.linspace(self.t_range[0], self.t_range[1], self.N_bc)
        y_4, t_4 = np.meshgrid(y_4, t_4)
        y_4 = np.ravel(y_4).T[:, None]
        t_4 = np.ravel(t_4).T[:, None]
        x_4 = np.ones_like(y_4) * np.pi

        x_b = np.vstack((x_1, x_2, x_3, x_4))
        y_b = np.vstack((y_1, y_2, y_3, y_4))
        t_b = np.vstack((t_1, t_2, t_3, t_4))

        u_b = self.u_train(x_b, y_b, t_b)
        v_b = self.v_train(x_b, y_b, t_b)
        rou_b = self.rou_func(x_b, y_b, t_b)

        u_ini = self.u_train(x_ini, y_ini, t_ini)
        v_ini = self.v_train(x_ini, y_ini, t_ini)
        rou_ini = self.rou_func(x_ini, y_ini, t_ini)

        return x_data, y_data, t_data, x_ini, y_ini, t_ini, x_b, y_b, t_b, u_b, v_b, rou_b, u_ini, v_ini, rou_ini
