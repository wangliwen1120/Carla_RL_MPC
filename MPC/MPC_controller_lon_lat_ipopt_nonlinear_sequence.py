from __future__ import division
import numpy as np
import casadi as ca
import time


class MPC_controller_lon_lat_ipopt_nonlinear_sequence:

    def __init__(self, param):
        self.rou = None
        self.ru = None
        self.rdu = None
        self.q = None
        self.Q1 = None
        self.Q2 = None
        self.Ru = None
        self.Rdu = None
        self.param = param
        self.T = self.param.T
        self.L = self.param.L
        self.N = self.param.N
        self.Nx = self.param.mpc_Nx
        self.Nu = self.param.mpc_Nu
        self.Ny = self.param.mpc_Ny
        self.Np = self.param.mpc_Np
        self.Nc = self.param.mpc_Nc
        self.Cy = self.param.mpc_Cy
        self.Lane = self.param.lanewidth
        self.stop_line = self.param.dstop
        self.u0_last = 0
        self.u1_last = 0

        # 纵向约束
        self.v_min = 0.0
        self.v_max = 70 / 3.6
        self.delta_f_min = -0.388
        self.delta_f_max = 0.388

        self.d_v_min = -4 / 1
        self.d_v_max = 3 / 1
        self.d_delta_f_min = -0.082 * 3
        self.d_delta_f_max = 0.082 * 3
        self.delta_a_min = -3
        self.delta_a_max = 3
        self.delta_d_delta_f_min = -0.0082
        self.delta_d_delta_f_max = 0.0082
        self.e_min = 0  # 松弛因子的约束
        # self.e_max = 0.02
        self.e_max = 2
        # 约束矩阵
        self.u = np.zeros((self.Nu, 1))
        self.x_max_ext = np.zeros([self.Np * self.Nx, 1])
        self.x_min_ext = np.zeros([self.Np * self.Nx, 1])
        self.y_max_ext = np.zeros([self.Np * self.Ny, 1])
        self.y_min_ext = np.zeros([self.Np * self.Ny, 1])

        self.x_ref = np.zeros((self.Np, 1))
        self.y_ref = np.zeros((self.Np, 1))
        self.phi_ref = np.zeros((self.Np, 1))

        self.x_ref_left = np.zeros((self.Np, 1))
        self.y_ref_left = np.zeros((self.Np, 1))
        self.phi_ref_left = np.zeros((self.Np, 1))

        self.Y_ref = None
        self.Y_ref_left = None
        self.next_states = np.zeros((self.Nx, self.Np)).copy().T
        self.u0 = np.array([0, 0] * (self.Nc)).reshape(-1, 2).T

        self.obj_x_ref = np.zeros((self.Np, 1))
        self.obj_y_ref = np.zeros((self.Np, 1))
        self.obj_phi_ref = np.zeros((self.Np, 1))
        self.Obj_pred = None

    def calc_input(self, x_current, x_frenet_current, obj_info, ref, ref_left, u_last, q, ru, rdu):
        # 预测时域和控制时域内的分块权重矩阵
        # 权重矩阵
        # y_ref
        for i in range(self.Np):
            self.x_ref[i] = ref[0][i]
            self.y_ref[i] = ref[1][i]
            self.phi_ref[i] = ref[2][i]
            self.x_ref_left[i] = ref_left[0][i]
            self.y_ref_left[i] = ref_left[1][i]
            self.phi_ref_left[i] = ref_left[2][i]

        for i in range(self.Np):
            # 前车状态预测
            self.obj_x_ref[i] = obj_info[0] + obj_info[3] * np.cos(obj_info[2]) * self.T * i
            self.obj_y_ref[i] = obj_info[1] + obj_info[3] * np.sin(obj_info[2]) * self.T * i
            self.obj_phi_ref[i] = obj_info[2] + obj_info[4] * self.T * i

        self.Y_ref = np.concatenate((self.x_ref.T, self.y_ref.T, self.phi_ref.T))
        self.Y_ref_left = np.concatenate((self.x_ref_left.T, self.y_ref_left.T, self.phi_ref_left.T))
        self.Obj_pred = np.concatenate((self.obj_x_ref.T, self.obj_y_ref.T, self.obj_phi_ref.T))

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        fai = ca.SX.sym('theta')
        states = ca.vcat([x, y, fai])

        vx = ca.SX.sym('vx')
        deta_f = ca.SX.sym('deta_f')
        controls = ca.vertcat(vx, deta_f)

        state_trans = ca.vcat([vx * ca.cos(fai), vx * ca.sin(fai), vx * ca.tan(deta_f) / self.L])

        # function
        f = ca.Function('f', [states, controls], [state_trans], ['states', 'control_input'], ['state_trans'])

        U = ca.SX.sym('U', self.Nu, self.Nc)
        X = ca.SX.sym('X', self.Nx, self.Np)
        Ref = ca.SX.sym('Ref', self.Nx + self.Nx + self.Nx, self.Np)  # ego_lane,left_lane,obs_info

        self.q = 1.0
        self.ru = 0
        self.rdu = 0.3
        self.S = 0.0
        self.Q1 = self.q * np.eye(self.Nx)
        self.Q2 = (1 - self.q) * np.eye(self.Nx)
        self.Ru = self.ru * np.eye(self.Nu)
        self.Rdu = self.rdu * np.eye(self.Nu)

        # cost function
        obj = 0
        g1 = []
        g2 = []
        g3 = []

        g1.append(X[:, 0] - Ref[:self.Nx, 0])

        for i in range(self.Nc):
            if i == 0:
                dU_cost = ((U[0, 0] - self.u0_last) ** 2 + (U[0, 1] - self.u1_last) ** 2) * self.rdu
            else:
                dU_cost = ca.mtimes([(U[:, i] - U[:, i - 1]).T, self.Rdu, (U[:, i] - U[:, i - 1])])
            U_cost = ca.mtimes([U[:, i].T, self.Ru, U[:, i]])
            obj = obj + U_cost + dU_cost

        for i in range(self.Np):
            Obj_cost = self.S / (((X[0, i] - Ref[2*self.Nx, -1]) ** 2) + ((X[1, i] - Ref[2*self.Nx+1, -1]) ** 2))
            obj = obj + Obj_cost

        for i in range(self.Np):
            Ref_ter_1 = ca.mtimes([(X[:, -1] - Ref[:self.Nx, -1]).T, self.Q1, X[:, -1] - Ref[:self.Nx, -1]])
            Ref_ter_2 = ca.mtimes([(X[:, -1] - Ref[self.Nx:2*self.Nx, -1]).T, self.Q2, X[:, -1] - Ref[self.Nx:2*self.Nx, -1]])
            obj = obj + Ref_ter_1 + Ref_ter_2

        for i in range(self.Np - 1):
            if i in range(self.Nc):
                x_next_ = f(X[:, i], U[:, i]) * self.T + X[:, i]
            else:
                x_next_ = f(X[:, i], U[:, self.Nc - 1]) * self.T + X[:, i]
            g1.append(X[:, i + 1] - x_next_)

        for i in range(self.Nc):
            if i == 0:
                g2.append((U[0, 0] - self.u0_last) / self.T)
                g3.append((U[1, 0] - self.u1_last) / self.T)
            else:
                g2.append((U[:, i] - U[:, i - 1])[0] / self.T)
                g3.append((U[:, i] - U[:, i - 1])[1] / self.T)

        opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

        nlp_prob = {'f': obj, 'x': opt_variables, 'p': Ref, 'g': ca.vertcat(*g1, *g2, *g3)}
        opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 5, 'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}

        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

        lbg = []
        ubg = []
        lbx = []
        ubx = []

        for _ in range(self.Np):
            lbg.append(0)
            lbg.append(0)
            lbg.append(0)
            ubg.append(0)
            ubg.append(0)
            ubg.append(0)
        for _ in range(self.Nc):
            lbg.append(self.d_v_min)
            lbg.append(self.d_delta_f_min)
            ubg.append(self.d_v_max)
            ubg.append(self.d_delta_f_max)

        for _ in range(self.Nc):
            lbx.append(self.v_min)
            lbx.append(self.delta_f_min)
            ubx.append(self.v_max)
            ubx.append(self.delta_f_max)

        for i in range(self.Np):
            lbx.append(-np.inf)
            lbx.append(27)
            lbx.append(-np.inf)
            ubx.append(np.inf)
            ubx.append(38)
            ubx.append(np.inf)

        index_t = []

        Ref = np.concatenate((self.Y_ref, self.Y_ref_left, self.Obj_pred), axis=0)
        Ref[:self.Nx, 0] = np.array([x_current[0][0], x_current[1][0], x_current[2][0]])

        init_control = np.concatenate((self.u0.reshape(-1, 1), self.next_states.reshape(-1, 1)))
        t_ = time.time()
        res = solver(x0=init_control, p=Ref, lbg=lbg,
                     lbx=lbx, ubg=ubg, ubx=ubx)
        index_t.append(time.time() - t_)
        # the feedback is in the series [u0, x0, u1, x1, ...]
        estimated_opt = res['x'].full()
        u0 = estimated_opt[:(self.Nc) * self.Nu].reshape(self.Nc, self.Nu)
        x_m = estimated_opt[(self.Nc) * self.Nu:].reshape(self.Np, self.Nx)
        # self.next_states = np.concatenate((x_m[1:], x_m[-1:]), axis=0)
        # self.u0 = np.concatenate((u0[1:], u0[-1:]))
        self.u0_last = estimated_opt[0]
        self.u1_last = estimated_opt[1]
        print(estimated_opt[0])
        print(estimated_opt[1])

        MPC_unsolved = False
        return np.array([estimated_opt[0], estimated_opt[1]]), MPC_unsolved, x_m
