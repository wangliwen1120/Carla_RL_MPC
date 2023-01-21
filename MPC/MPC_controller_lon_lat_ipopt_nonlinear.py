from __future__ import division
import numpy as np
import casadi as ca


class MPC_controller_lon_lat_ipopt_nonlinear:

    def __init__(self, param):
        self.rou = None
        self.ru = None
        self.rdu = None
        self.q = None
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
        self.delta_f_min = -0.194
        self.delta_f_max = 0.194
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

        self.x_predict = np.zeros((self.Np, 1))
        self.y_predict = np.zeros((self.Np, 1))
        self.phi_predict = np.zeros((self.Np, 1))

        self.obj_x_ref = np.zeros((self.Np, 1))
        self.obj_y_ref = np.zeros((self.Np, 1))
        self.obj_phi_ref = np.zeros((self.Np, 1))

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
            self.x_ref[i] = ref[0][self.Np - 1]
            self.y_ref[i] = ref[1][self.Np - 1]
            self.phi_ref[i] = ref[2][self.Np - 1]
            self.x_ref_left[i] = ref_left[0][self.Np - 1]
            self.y_ref_left[i] = ref_left[1][self.Np - 1]
            self.phi_ref_left[i] = ref_left[2][self.Np - 1]
        for i in range(self.Np):
            self.y_ref_ext[i * self.Ny: (i + 1) * self.Ny, 0] = [self.x_ref[i], self.y_ref[i], self.phi_ref[i]]
            self.y_ref_left_ext[i * self.Ny: (i + 1) * self.Ny, 0] = [self.x_ref_left[i], self.y_ref_left[i],
                                                                      self.phi_ref_left[i]]

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        fai = ca.SX.sym('theta')
        states = ca.vertcat(x, y)
        states = ca.vertcat(states, theta)




        MPC_unsolved = False
        return np.array([[2], [0]]), MPC_unsolved
