from __future__ import division
import numpy as np
import casadi as ca
import time


class MPC_controller_lon_lat_ipopt_nonlinear:

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
        self.v_x_ref = self.param.v_x_ref
        self.v_x_0 = self.param.v_x_0
        self.Pos_x_0 = self.param.Pos_x_0

        # 纵向约束
        self.v_min = 0.0
        self.v_max = 70 / 3.6
        self.delta_f_min = -0.194 * 2
        self.delta_f_max = 0.194 * 2

        self.d_v_min = -4 / 1
        self.d_v_max = 3 / 1
        self.d_delta_f_min = -0.082
        self.d_delta_f_max = 0.082
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
        self.u_max_ext = np.zeros([self.Nc * self.Nu, 1])
        self.u_min_ext = np.zeros([self.Nc * self.Nu, 1])
        self.du_max_ext = np.zeros([self.Nc * self.Nu, 1])
        self.du_min_ext = np.zeros([self.Nc * self.Nu, 1])

        self.pos_x_max = np.zeros((self.Np, 1))
        self.pos_x_min = np.zeros((self.Np, 1))
        self.x_ref = np.zeros((self.Np, 1))
        self.pos_y_max = np.zeros((self.Np, 1))
        self.pos_y_min = np.zeros((self.Np, 1))
        self.y_ref = np.zeros((self.Np, 1))
        self.pos_phi_max = np.zeros((self.Np, 1))
        self.pos_phi_min = np.zeros((self.Np, 1))
        self.phi_ref = np.zeros((self.Np, 1))
        self.y_ref_ext = np.zeros([self.Np * self.Ny, 1])

        self.y_ref_left_ext = np.zeros([self.Np * self.Ny, 1])
        self.x_ref_left = np.zeros((self.Np, 1))
        self.y_ref_left = np.zeros((self.Np, 1))
        self.phi_ref_left = np.zeros((self.Np, 1))

        self.x_ext = np.zeros([self.Np * self.Ny, 1])
        self.y_ext = np.zeros([self.Np * self.Ny, 1])
        self.y_error = np.zeros([self.Np * self.Ny, 1])
        self.y_error_left = np.zeros([self.Np * self.Ny, 1])
        self.Y = np.zeros([self.Np, 1])
        self.Y_left = np.zeros([self.Np, 1])
        self.Y_ext = np.zeros([self.Np * self.Nx, 1])
        self.Y_ref = None
        self.Y_ref_left = None
        self.next_states = np.zeros((self.Nx, self.Np)).copy().T
        self.u0 = np.array([1, 2] * (self.Np - 1)).reshape(-1, 2).T

        self.x_predict = np.zeros((self.Np, 1))
        self.y_predict = np.zeros((self.Np, 1))
        self.phi_predict = np.zeros((self.Np, 1))


        self.obj_x_ref = np.zeros((self.Np, 1))
        self.obj_y_ref = np.zeros((self.Np, 1))
        self.obj_phi_ref = np.zeros((self.Np, 1))
        self.Obj_pred = None

        self.y_x_max = np.zeros((self.Np, 1))
        self.y_y_max = np.zeros((self.Np, 1))
        self.y_phi_max = np.zeros((self.Np, 1))
        self.y_x_min = np.zeros((self.Np, 1))
        self.y_y_min = np.zeros((self.Np, 1))
        self.y_phi_min = np.zeros((self.Np, 1))

        for i in range(self.Nc):
            self.u_max_ext[i * self.Nu:(i + 1) * self.Nu] = np.array([[self.v_max], [self.delta_f_max]])
            self.u_min_ext[i * self.Nu:(i + 1) * self.Nu] = np.array([[self.v_min], [self.delta_f_min]])
            self.du_max_ext[i * self.Nu:(i + 1) * self.Nu] = np.array([[self.d_v_max], [self.d_delta_f_max]])
            self.du_min_ext[i * self.Nu:(i + 1) * self.Nu] = np.array([[self.d_v_min], [self.d_delta_f_min]])

    def calc_input(self, x_current, x_frenet_current, obj, ref, ref_left, u_last, q, ru, rdu):
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
            self.obj_x_ref[i] = obj[0] + obj[3] * np.cos(obj[2]) * self.T * i
            self.obj_y_ref[i] = obj[1] + obj[3] * np.sin(obj[2]) * self.T * i
            self.obj_phi_ref[i] = obj[2] + obj[4] * self.T * i

        self.Y_ref = np.concatenate((self.x_ref.T, self.y_ref.T, self.phi_ref.T))
        self.Y_ref_left = np.concatenate((self.x_ref.T, self.y_ref.T - 3.5, self.phi_ref.T))
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

        #       self.Np -1 =self.Nc
        #         U = ca.SX.sym('U', self.Nu, self.Nc)
        U = ca.SX.sym('U', self.Nu, self.Np - 1)
        X = ca.SX.sym('X', self.Nx, self.Np)
        C_R = ca.SX.sym('C_R', self.Nx + self.Nx + self.Nx)

        self.q = 1.0
        self.ru = 0.3
        self.rdu = 0.1
        self.Q1 = self.q * np.eye(self.Nx)
        self.Q2 = (1-self.q) * np.eye(self.Nx)
        self.Ru = self.ru * np.eye(self.Nu)
        self.Rdu = self.rdu * np.eye(self.Nu)

        # cost function
        obj = 0
        g = []
        S = 0
        g.append(X[:, 0] - C_R[:3])

        for i in range(self.Np - 1):
            Ref_1_cost = ca.mtimes([(X[:, i] - C_R[3:6]).T, self.Q1, X[:, i] - C_R[3:6]])
            Ref_2_cost = ca.mtimes([(X[:, i] - C_R[6:]).T, self.Q2, X[:, i] - C_R[6:]])

            # Obj_cost = S * (1/(X[0, i]-self.obj_x_ref[i])**2 + (X[1, i]-self.obj_y_ref[i])**2)
            U_cost = ca.mtimes([U[:, i].T, self.Ru, U[:, i]])
            obj = obj + Ref_1_cost + Ref_2_cost +  U_cost
            x_next_ = f(X[:, i], U[:, i]) * self.T + X[:, i]
            g.append(X[:, i + 1] - x_next_)

        opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

        nlp_prob = {'f': obj, 'x': opt_variables, 'p': C_R, 'g': ca.vertcat(*g)}
        opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 5, 'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}

        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

        lbg = 0.0
        ubg = 0.0
        lbx = []
        ubx = []

        for _ in range(self.Np - 1):
            lbx.append(0)
            lbx.append(-self.delta_f_max)
            ubx.append(self.v_max)
            ubx.append(self.delta_f_max)

        for _ in range(self.Np):
            lbx.append(-np.inf)
            lbx.append(-np.inf)
            lbx.append(-np.inf)
            ubx.append(np.inf)
            ubx.append(np.inf)
            ubx.append(np.inf)

        # x0 = np.array([x_current[0], x_current[1], x_current[2]]).reshape(-1, 1)  # initial state
        # x0_ = x0.copy()

        index_t = []
        Ref_1 = self.Y_ref[:, -1]
        Ref_2 = self.Y_ref_left[:, -1]
        C_R = np.array([x_current[0][0], x_current[1][0], x_current[2][0], Ref_1[0]-36, Ref_1[1], Ref_1[2], Ref_2[0]-36, Ref_2[1], Ref_2[2]])

        init_control = np.concatenate((self.u0.reshape(-1, 1), self.next_states.reshape(-1, 1)))
        t_ = time.time()
        res = solver(x0=init_control, p=C_R, lbg=lbg,
                     lbx=lbx, ubg=ubg, ubx=ubx)
        index_t.append(time.time() - t_)
        # the feedback is in the series [u0, x0, u1, x1, ...]
        estimated_opt = res['x'].full()
        u0 = estimated_opt[:(self.Np - 1) * self.Nu].reshape(self.Np - 1, self.Nu)
        x_m = estimated_opt[(self.Np - 1) * self.Nu:].reshape(self.Np, self.Nx)
        self.next_states = np.concatenate((x_m[1:], x_m[-1:]), axis=0)
        self.u0 = np.concatenate((u0[1:], u0[-1:]))
        print(estimated_opt[0])
        print(estimated_opt[1])

        MPC_unsolved = False
        return np.array([estimated_opt[0], estimated_opt[1]]), MPC_unsolved, x_m
