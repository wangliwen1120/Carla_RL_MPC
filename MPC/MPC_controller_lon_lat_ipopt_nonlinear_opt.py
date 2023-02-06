from __future__ import division
import numpy as np
import casadi as ca
import time

# single shooting --- time too long

class MPC_controller_lon_lat_ipopt_nonlinear_opt:

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

        # 纵向约束
        self.v_min = 0.0
        self.v_max = 70 / 3.6
        self.delta_f_min = -0.194*2
        self.delta_f_max = 0.194*2

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

        self.x_ref = np.zeros((self.Np, 1))
        self.y_ref = np.zeros((self.Np, 1))
        self.phi_ref = np.zeros((self.Np, 1))

        self.x_ref_left = np.zeros((self.Np, 1))
        self.y_ref_left = np.zeros((self.Np, 1))
        self.phi_ref_left = np.zeros((self.Np, 1))

        self.Y_ref = None
        self.Y_ref_left = None
        self.next_states = np.zeros((self.Np, self.Nx))
        self.u0 = np.zeros((self.Nc,self.Nu))

        self.obj_x_ref = np.zeros((self.Np, 1))
        self.obj_y_ref = np.zeros((self.Np, 1))
        self.obj_phi_ref = np.zeros((self.Np, 1))
        self.Obj_pred = None

        for i in range(self.Nc):
            self.u_max_ext[i * self.Nu:(i + 1) * self.Nu] = np.array([[self.v_max], [self.delta_f_max]])
            self.u_min_ext[i * self.Nu:(i + 1) * self.Nu] = np.array([[self.v_min], [self.delta_f_min]])
            self.du_max_ext[i * self.Nu:(i + 1) * self.Nu] = np.array([[self.d_v_max], [self.d_delta_f_max]])
            self.du_min_ext[i * self.Nu:(i + 1) * self.Nu] = np.array([[self.d_v_min], [self.d_delta_f_min]])

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
            # self.x_ref_left[i] = ref[0][i]
            # self.y_ref_left[i] = ref[1][i]-3.5
            # self.phi_ref_left[i] = ref[2][i]

        for i in range(self.Np):
            # 前车状态预测
            self.obj_x_ref[i] = obj_info[0] + obj_info[3] * np.cos(obj_info[2]) * self.T * i
            self.obj_y_ref[i] = obj_info[1] + obj_info[3] * np.sin(obj_info[2]) * self.T * i
            self.obj_phi_ref[i] = obj_info[2] + obj_info[4] * self.T * i

        self.Y_ref = np.concatenate((self.x_ref.T, self.y_ref.T, self.phi_ref.T)).T
        self.Y_ref_left = np.concatenate((self.x_ref_left.T, self.y_ref_left.T, self.phi_ref_left.T)).T
        self.Obj_pred = np.concatenate((self.obj_x_ref.T, self.obj_y_ref.T, self.obj_phi_ref.T)).T

        opti = ca.Opti()
        # control variables, linear velocity v and angle delta
        opt_controls = opti.variable(self.Nc, self.Nu)
        vx = opt_controls[:, 0]
        delta_f = opt_controls[:, 1]
        opt_states = opti.variable(self.Np, self.Nx)
        x = opt_states[:, 0]
        y = opt_states[:, 1]
        fai = opt_states[:, 2]

        # parameters
        opt_x0 = opti.parameter(self.Nx)
        Ref = opti.parameter(self.Np, self.Nx+self.Nx+self.Nx)

        # create model
        # def f(x_, u_):
        #     return ca.vertcat(
        #         *[opt_controls[i, :][0] * ca.cos(opt_states[i, :][2]), opt_controls[i, :][0] * ca.sin(opt_states[i, :][2]), opt_controls[i, :][0]*ca.tan(opt_controls[i, :][1])/self.L])
        #
        # def f_np(x_, u_):
        #     return np.array(
        #         [u_[0] * ca.cos(x_[2]), u_[0] * ca.sin(x_[2]), u_[0] * ca.tan(u_[1]) / self.L])

        # init_condition
        opti.subject_to(opt_states[0, :] == opt_x0.T)
        for i in range(self.Np-1):
            if i in range(self.Nc):
                x_next = opt_states[i, :] \
                         +ca.vertcat(*[opt_controls[i, :][0] * ca.cos(opt_states[i, :][2]),
                                        opt_controls[i, :][0] * ca.sin(opt_states[i, :][2]),
                                        opt_controls[i, :][0]*ca.tan(opt_controls[i, :][1])/self.L]).T * self.T
            else:
                x_next = opt_states[i, :] \
                         + ca.vertcat(*[opt_controls[self.Nc-1, :][0] * ca.cos(opt_states[i, :][2]),
                                        opt_controls[self.Nc-1, :][0] * ca.sin(opt_states[i, :][2]),
                                        opt_controls[self.Nc-1, :][0] * ca.tan(opt_controls[self.Nc-1, :][1]) / self.L]).T * self.T

            opti.subject_to(opt_states[i + 1, :] == x_next)

        # define the cost function
        # some addition parameters
        self.q = 1.0
        self.ru = 0.3
        self.rdu = 0.1
        self.Q1 = self.q * np.eye(self.Nx)
        self.Q2 = (1 - self.q) * np.eye(self.Nx)
        self.Ru = self.ru * np.eye(self.Nu)
        self.Rdu = self.rdu * np.eye(self.Nu)

        # cost function
        obj = 0
        S=10

        for i in range(self.Nc):
            Obj_cost = S / (((opt_states[i, 0] - Ref[i,self.Nx*2]) ** 2) + ((opt_states[i, 1] - Ref[i,self.Nx*2+1]) ** 2))
            Ref_1_cost = ca.mtimes([(opt_states[i,:] - Ref[i,:self.Nx]), self.Q1, (opt_states[i,:] - Ref[i,:self.Nx]).T])
            Ref_2_cost = ca.mtimes([(opt_states[i,:] - Ref[i,self.Nx:self.Nx*2]), self.Q2, (opt_states[i,:] - Ref[i,self.Nx:self.Nx*2]).T])
            U_cost = ca.mtimes([opt_controls[i, :], self.Ru, opt_controls[i, :].T])
            obj = obj + Ref_1_cost + Ref_2_cost + Obj_cost + U_cost

        for i in range(self.Nc, self.Np - 1):
            Obj_cost = S / (((opt_states[i, 0] - Ref[i, self.Nx * 2]) ** 2) + ((opt_states[i, 1] - Ref[i, self.Nx * 2 + 1]) ** 2))
            Ref_1_cost = ca.mtimes(
                [(opt_states[i, :] - Ref[i, :self.Nx]), self.Q1, (opt_states[i, :] - Ref[i, :self.Nx]).T])
            Ref_2_cost = ca.mtimes([(opt_states[i, :] - Ref[i, self.Nx:self.Nx * 2]), self.Q2,
                                    (opt_states[i, :] - Ref[i, self.Nx:self.Nx * 2]).T])

            obj = obj + Ref_1_cost + Ref_2_cost + Obj_cost

        opti.minimize(obj)
        # boundrary and control conditions
        # for i in range(self.Np):
        #     opti.subject_to(opti.bounded(self.stop_line, opt_states[i, 0] - Ref[i, self.Nx * 2], np.inf))
        opti.subject_to(opti.bounded(-np.inf, x, np.inf))
        opti.subject_to(opti.bounded(self.v_min, vx, self.v_max))
        opti.subject_to(opti.bounded(self.delta_f_min, delta_f, self.delta_f_max))

        opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 5, 'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}

        opti.solver('ipopt', opts_setting)

        Ref_parameter = np.concatenate((self.Y_ref, self.Y_ref_left, self.Obj_pred), axis=1)
        opti.set_value(Ref, Ref_parameter)

        index_t = []
        # set parameter, here only update initial state of x (x0)
        opti.set_value(opt_x0, x_current.T)
        # set optimizing target withe init guess
        opti.set_initial(opt_controls, self.u0)  # (N, 2)
        opti.set_initial(opt_states, self.next_states)  # (N+1, 3)
        # solve the problem once again
        t_ = time.time()
        sol = opti.solve()
        index_t.append(time.time() - t_)
        # obtain the control input
        u_res = sol.value(opt_controls)
        next_states_pred = sol.value(opt_states)

        self.u0 = np.concatenate((self.u0[1:], self.u0[-1:]))
        self.next_states = np.concatenate((self.next_states[1:], self.next_states[-1:]))

        print(u_res[0])

        MPC_unsolved = False
        return np.array([u_res[0,0], u_res[0,1]]), MPC_unsolved, next_states_pred



