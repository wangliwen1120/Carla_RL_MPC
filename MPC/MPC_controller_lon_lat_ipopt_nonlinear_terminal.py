from __future__ import division
import numpy as np
import casadi as ca
import time
import math
import numba

def frenet_to_inertial(s, d, csp):
    """
    transform a point from frenet frame to inertial frame
    input: frenet s and d variable and the instance of global cubic spline class
    output: x and y in global frame
    """
    ix, iy, iz = csp.calc_position(s)
    iyaw = csp.calc_yaw(s)
    x = ix + d * math.cos(iyaw + math.pi / 2.0)
    y = iy + d * math.sin(iyaw + math.pi / 2.0)
    return x, y, iz, iyaw

class MPC_controller_lon_lat_ipopt_nonlinear_terminal:

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
        self.Length = self.param.Length
        self.Width = self.param.Width
        self.Height = self.param.Height
        self.N = self.param.N
        self.Nx = self.param.mpc_Nx
        self.Nu = self.param.mpc_Nu
        self.Ny = self.param.mpc_Ny
        self.Np = self.param.mpc_Np
        self.Nc = self.param.mpc_Nc
        self.Cy = self.param.mpc_Cy
        self.Lane = self.param.lanewidth
        self.stop_line = self.param.dstop

        # 横纵向约束
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

        # ref矩阵
        self.obj_ego_preceding = None
        self.obj_preceding_x_ref = np.zeros((self.Np, 1))
        self.obj_preceding_y_ref = np.zeros((self.Np, 1))
        self.obj_preceding_phi_ref = np.zeros((self.Np, 1))
        self.obj_ego_following = None
        self.obj_following_x_ref = np.zeros((self.Np, 1))
        self.obj_following_y_ref = np.zeros((self.Np, 1))
        self.obj_following_phi_ref = np.zeros((self.Np, 1))
        self.obj_left_preceding = None
        self.obj_left_preceding_x_ref = np.zeros((self.Np, 1))
        self.obj_left_preceding_y_ref = np.zeros((self.Np, 1))
        self.obj_left_preceding_phi_ref = np.zeros((self.Np, 1))
        self.obj_left_following = None
        self.obj_left_following_x_ref = np.zeros((self.Np, 1))
        self.obj_left_following_y_ref = np.zeros((self.Np, 1))
        self.obj_left_following_phi_ref = np.zeros((self.Np, 1))
        self.obj_right_preceding = None
        self.obj_right_preceding_x_ref = np.zeros((self.Np, 1))
        self.obj_right_preceding_y_ref = np.zeros((self.Np, 1))
        self.obj_right_preceding_phi_ref = np.zeros((self.Np, 1))
        self.obj_right_following = None
        self.obj_right_following_x_ref = np.zeros((self.Np, 1))
        self.obj_right_following_y_ref = np.zeros((self.Np, 1))
        self.obj_right_following_phi_ref = np.zeros((self.Np, 1))
        self.obj_left = None
        self.obj_left_x_ref = np.zeros((self.Np, 1))
        self.obj_left_y_ref = np.zeros((self.Np, 1))
        self.obj_left_phi_ref = np.zeros((self.Np, 1))
        self.obj_right = None
        self.obj_right_x_ref = np.zeros((self.Np, 1))
        self.obj_right_y_ref = np.zeros((self.Np, 1))
        self.obj_right_phi_ref = np.zeros((self.Np, 1))

        self.obj_ego_preceding = None
        self.obj_preceding_x_ref = np.zeros((self.Np, 1))
        self.obj_preceding_y_ref = np.zeros((self.Np, 1))
        self.obj_preceding_phi_ref = np.zeros((self.Np, 1))


        self.obj_x_ref = np.zeros((self.Np, 1))
        self.obj_y_ref = np.zeros((self.Np, 1))
        self.obj_phi_ref = np.zeros((self.Np, 1))
        self.obj_actor_id = np.zeros((self.Np, 1))
        self.obj_pos = np.zeros((self.Np, 1))

        self.next_states = np.zeros((self.Nx, self.Np)).copy().T
        self.u0 = np.array([0, 0] * self.Nc).reshape(-1, 2).T

    def calc_input(self, x_current, obj_info, ref, ref_left, u_last, vehicle_ahead,csp,fpath, q, ru, rdu):
        start_time = time.time()
        obj_Mux = []
        vehicle_num = 0
        # # 预测时域内的ref矩阵
        # # 预测时域内的obj矩阵
        # obj_preceding
        if obj_info['Ego_preceding'][0] != None:
            obj_preceding_x = obj_info['Ego_preceding'][2][0]
            obj_preceding_y = obj_info['Ego_preceding'][2][1]
            obj_preceding_phi = obj_info['Ego_preceding'][2][4]
            obj_preceding_speed = obj_info['Ego_preceding'][2][5]
            obj_preceding_delta_f = obj_info['Ego_preceding'][2][6]
            for i in range(self.Np):
                # 前车状态预测
                self.obj_preceding_x_ref[i] = obj_preceding_x + obj_preceding_speed * np.cos(
                    obj_preceding_phi) * self.T * i
                self.obj_preceding_y_ref[i] = obj_preceding_y + obj_preceding_speed * np.sin(
                    obj_preceding_phi) * self.T * i
                self.obj_preceding_phi_ref[i] = obj_preceding_phi + obj_preceding_delta_f * self.T * i
            self.obj_ego_preceding = np.concatenate(
                (self.obj_preceding_x_ref.T, self.obj_preceding_y_ref.T, self.obj_preceding_phi_ref.T))
        # # obj_following
        # if obj_info['Ego_following'][0] != None:
        #     vehicle_num += 1
        #     obj_following_x = obj_info['Ego_following'][2][0]
        #     obj_following_y = obj_info['Ego_following'][2][1]
        #     obj_following_phi = obj_info['Ego_following'][2][4]
        #     obj_following_speed = obj_info['Ego_following'][2][5]
        #     obj_following_delta_f = obj_info['Ego_following'][2][6]
        #     for i in range(self.Np):
        #         # 前车状态预测
        #         self.obj_following_x_ref[i] = obj_following_x + obj_following_speed * np.cos(
        #             obj_following_phi) * self.T * i
        #         self.obj_following_y_ref[i] = obj_following_y + obj_following_speed * np.sin(
        #             obj_following_phi) * self.T * i
        #         self.obj_following_phi_ref[i] = obj_following_phi + obj_following_delta_f * self.T * i
        #     self.obj_ego_following = np.concatenate(
        #         (self.obj_following_x_ref.T, self.obj_following_y_ref.T, self.obj_following_phi_ref.T))
        #
        # # obj_Left_preceding
        # if obj_info['Left_preceding'][0] != None:
        #     vehicle_num += 1
        #     obj_left_preceding_x = obj_info['Left_preceding'][2][0]
        #     obj_left_preceding_y = obj_info['Left_preceding'][2][1]
        #     obj_left_preceding_phi = obj_info['Left_preceding'][2][4]
        #     obj_left_preceding_speed = obj_info['Left_preceding'][2][5]
        #     obj_left_preceding_delta_f = obj_info['Left_preceding'][2][6]
        #     for i in range(self.Np):
        #         # 前车状态预测
        #         self.obj_left_preceding_x_ref[i] = obj_left_preceding_x + obj_left_preceding_speed * np.cos(
        #             obj_left_preceding_phi) * self.T * i
        #         self.obj_left_preceding_y_ref[i] = obj_left_preceding_y + obj_left_preceding_speed * np.sin(
        #             obj_left_preceding_phi) * self.T * i
        #         self.obj_left_preceding_phi_ref[
        #             i] = obj_left_preceding_phi + obj_left_preceding_delta_f * self.T * i
        #     self.obj_left_preceding = np.concatenate(
        #         (self.obj_left_preceding_x_ref.T, self.obj_left_preceding_y_ref.T,
        #          self.obj_left_preceding_phi_ref.T))
        # # obj_Left_following
        # if obj_info['Left_following'][0] != None:
        #     vehicle_num += 1
        #     obj_left_following_x = obj_info['Left_following'][2][0]
        #     obj_left_following_y = obj_info['Left_following'][2][1]
        #     obj_left_following_phi = obj_info['Left_following'][2][4]
        #     obj_left_following_speed = obj_info['Left_following'][2][5]
        #     obj_left_following_delta_f = obj_info['Left_following'][2][6]
        #     for i in range(self.Np):
        #         # 前车状态预测
        #         self.obj_left_following_x_ref[i] = obj_left_following_x + obj_left_following_speed * np.cos(
        #             obj_left_following_phi) * self.T * i
        #         self.obj_left_following_y_ref[i] = obj_left_following_y + obj_left_following_speed * np.sin(
        #             obj_left_following_phi) * self.T * i
        #         self.obj_left_following_phi_ref[
        #             i] = obj_left_following_phi + obj_left_following_delta_f * self.T * i
        #     self.obj_left_following = np.concatenate(
        #         (self.obj_left_following_x_ref.T, self.obj_left_following_y_ref.T,
        #          self.obj_left_following_phi_ref.T))
        # # obj_Right_preceding
        # if obj_info['Right_preceding'][0] != None:
        #     vehicle_num += 1
        #     obj_right_preceding_x = obj_info['Right_preceding'][2][0]
        #     obj_right_preceding_y = obj_info['Right_preceding'][2][1]
        #     obj_right_preceding_phi = obj_info['Right_preceding'][2][4]
        #     obj_right_preceding_speed = obj_info['Right_preceding'][2][5]
        #     obj_right_preceding_delta_f = obj_info['Right_preceding'][2][6]
        #     for i in range(self.Np):
        #         # 前车状态预测
        #         self.obj_right_preceding_x_ref[i] = obj_right_preceding_x + obj_right_preceding_speed * np.cos(
        #             obj_right_preceding_phi) * self.T * i
        #         self.obj_right_preceding_y_ref[i] = obj_right_preceding_y + obj_right_preceding_speed * np.sin(
        #             obj_right_preceding_phi) * self.T * i
        #         self.obj_right_preceding_phi_ref[
        #             i] = obj_right_preceding_phi + obj_right_preceding_delta_f * self.T * i
        #     self.obj_right_preceding = np.concatenate(
        #         (
        #             self.obj_right_preceding_x_ref.T, self.obj_right_preceding_y_ref.T,
        #             self.obj_right_preceding_phi_ref.T))
        # # obj_Right_following
        # if obj_info['Right_following'][0] != None:
        #     vehicle_num += 1
        #     obj_right_following_x = obj_info['Right_following'][2][0]
        #     obj_right_following_y = obj_info['Right_following'][2][1]
        #     obj_right_following_phi = obj_info['Right_following'][2][4]
        #     obj_right_following_speed = obj_info['Right_following'][2][5]
        #     obj_right_following_delta_f = obj_info['Right_following'][2][6]
        #     for i in range(self.Np):
        #         # 前车状态预测
        #         self.obj_right_following_x_ref[i] = obj_right_following_x + obj_right_following_speed * np.cos(
        #             obj_right_following_phi) * self.T * i
        #         self.obj_right_following_y_ref[i] = obj_right_following_y + obj_right_following_speed * np.sin(
        #             obj_right_following_phi) * self.T * i
        #         self.obj_right_following_phi_ref[
        #             i] = obj_right_following_phi + obj_right_following_delta_f * self.T * i
        #     self.obj_right_following = np.concatenate(
        #         (
        #             self.obj_right_following_x_ref.T, self.obj_right_following_y_ref.T,
        #             self.obj_right_following_phi_ref.T))
        # # obj_Left
        # if obj_info['Left'][0] != None:
        #     vehicle_num += 1
        #     obj_left_x = obj_info['Left'][2][0]
        #     obj_left_y = obj_info['Left'][2][1]
        #     obj_left_phi = obj_info['Left'][2][4]
        #     obj_left_speed = obj_info['Left'][2][5]
        #     obj_left_delta_f = obj_info['Left'][2][6]
        #     for i in range(self.Np):
        #         # 前车状态预测
        #         self.obj_left_x_ref[i] = obj_left_x + obj_left_speed * np.cos(obj_left_phi) * self.T * i
        #         self.obj_left_y_ref[i] = obj_left_y + obj_left_speed * np.sin(obj_left_phi) * self.T * i
        #         self.obj_left_phi_ref[i] = obj_left_phi + obj_left_delta_f * self.T * i
        #     self.obj_left = np.concatenate((self.obj_left_x_ref.T, self.obj_left_y_ref.T, self.obj_left_phi_ref.T))
        #     # obj_Right
        # if obj_info['Right'][0] != None:
        #     vehicle_num += 1
        #     obj_right_x = obj_info['Right'][2][0]
        #     obj_right_y = obj_info['Right'][2][1]
        #     obj_right_phi = obj_info['Right'][2][4]
        #     obj_right_speed = obj_info['Right'][2][5]
        #     obj_right_delta_f = obj_info['Right'][2][6]
        #     for i in range(self.Np):
        #         # 前车状态预测
        #         self.obj_right_x_ref[i] = obj_right_x + obj_right_speed * np.cos(obj_right_phi) * self.T * i
        #         self.obj_right_y_ref[i] = obj_right_y + obj_right_speed * np.sin(obj_right_phi) * self.T * i
        #         self.obj_right_phi_ref[i] = obj_right_phi + obj_right_delta_f * self.T * i
        #     self.obj_right = np.concatenate(
        #         (self.obj_right_x_ref.T, self.obj_right_y_ref.T, self.obj_right_phi_ref.T))
        # 预测时域内的obj矩阵
        for j in range(np.size(obj_info['Obj_actor'])):
            obj_x = obj_info['Obj_cartesian'][j][0]
            obj_y = obj_info['Obj_cartesian'][j][1]
            obj_phi = obj_info['Obj_cartesian'][j][4]
            obj_speed = obj_info['Obj_cartesian'][j][5]
            obj_delta_f = obj_info['Obj_cartesian'][j][6]
            for i in range(self.Np):
                self.obj_x_ref[i] = obj_x + obj_speed * np.cos(obj_phi) * self.T * i
                self.obj_y_ref[i] = obj_y + obj_speed * np.sin(obj_phi) * self.T * i
                self.obj_phi_ref[i] = obj_phi + obj_delta_f * self.T * i
                self.obj_actor_id[i] = obj_info['Obj_actor'][j].id
                self.obj_pos[i] = 0  # 'not vehicle_around'
                if obj_info['Ego_preceding'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Ego_preceding'][0].id:
                        self.obj_pos[i] = 1  #'Ego_preceding'
                        vehicle_num += 1
                if obj_info['Ego_following'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Ego_following'][0].id:
                        self.obj_pos[i] = 2   #'Ego_following'
                        vehicle_num += 1
                if obj_info['Left_preceding'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Left_preceding'][0].id:
                        self.obj_pos[i] = 3   #'Ego_preceding'
                        vehicle_num += 1
                if obj_info['Left_following'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Left_following'][0].id:
                        self.obj_pos[i] = 4  #'Left_following'
                        vehicle_num += 1
                if obj_info['Right_preceding'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Right_preceding'][0].id:
                        self.obj_pos[i] = 5  #'Right_preceding'
                        vehicle_num += 1
                if obj_info['Right_following'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Right_following'][0].id:
                        self.obj_pos[i] = 6  #'Right_following'
                        vehicle_num += 1
                if obj_info['Left'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Left'][0].id:
                        self.obj_pos[i] = 7  #'Left'
                        vehicle_num += 1
                if obj_info['Right'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Right'][0].id:
                        self.obj_pos[i] = 8  #'Right'
                        vehicle_num += 1
            obj_Mux.append(np.concatenate(
                (self.obj_x_ref.T, self.obj_y_ref.T, self.obj_phi_ref.T, self.obj_actor_id.T,self.obj_pos.T)))
        vehicle_num = int(vehicle_num/self.Np)

        # 根据数学模型建模
        # 系统状态
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        phi = ca.SX.sym('theta')
        states = ca.vcat([x, y, phi])

        # 控制输入
        v = ca.SX.sym('v')
        delta_f = ca.SX.sym('delta_f')
        controls = ca.vertcat(v, delta_f)

        # dynamic_model
        state_trans = ca.vcat([v * ca.cos(phi), v * ca.sin(phi), v * ca.tan(delta_f) / self.L])

        # function
        f = ca.Function('f', [states, controls], [state_trans], ['states', 'control_input'], ['state_trans'])

        # 开始构建MPC
        # 相关变量，格式(状态长度， 步长)
        U = ca.SX.sym('U', self.Nu, self.Nc)  # 控制输出
        X = ca.SX.sym('X', self.Nx, self.Np)  # 系统状态
        C_R = ca.SX.sym('C_R', self.Nx + self.Nx + self.Nx)  # 构建问题的相关参数
        # 这里给定当前/初始位置，目标终点(本车道/左车道)位置

        # 权重矩阵
        self.q = 1.0
        self.ru = 0
        self.rdu = 0.3
        self.S = 0.1  # Obstacle avoidance function coefficient
        self.Q1 = self.q * np.eye(self.Nx)  # ego_lane: lane_2
        self.Q2 = (1 - self.q) * np.eye(self.Nx)  # left_lane: lane_1
        self.Ru = self.ru * np.eye(self.Nu)
        self.Rdu = self.rdu * np.eye(self.Nu)

        # cost function
        obj = 0  # 初始化优化目标值
        g1 = []  # 用list来存储优化目标的向量
        g2 = []
        g3 = []
        g1.append(X[:, 0] - C_R[:3])

        # U dU cost function
        for i in range(self.Nc):
            if i == 0:
                dU_cost = 0
            else:
                dU_cost = ca.mtimes([(U[:, i] - U[:, i - 1]).T, self.Rdu, (U[:, i] - U[:, i - 1])])
            U_cost = ca.mtimes([U[:, i].T, self.Ru, U[:, i]])
            obj = obj + U_cost + dU_cost

        # Obstacle avoidance cost function
        for j in range(np.size(obj_info['Obj_actor'])):
            if obj_Mux[j][4, 0] != 0:  #'not vehicle_around'
                for i in range(self.Np):
                    Obj_cost = self.S / (((X[0, i] - obj_Mux[j][0, i]) ** 2) + ((X[1, i] - obj_Mux[j][1, i]) ** 2))
                    obj = obj + Obj_cost
        # for j in range(np.size(obj_info['Obj_actor'])):
        #     if obj_info['Ego_preceding'][0] != None:
        #         if obj_Mux[j][3, 0] == obj_info['Ego_preceding'][0].id:
        #             for i in range(self.Np):
        #                 Obj_cost = self.S / (((X[0, i] - obj_Mux[j][0, i]) ** 2) + ((X[1, i] - obj_Mux[j][1, i]) ** 2))
        #                 obj = obj + Obj_cost
        #     if obj_info['Ego_following'][0] != None:
        #         if obj_Mux[j][3, 0] == obj_info['Ego_following'][0].id:
        #             for i in range(self.Np):
        #                 Obj_cost = self.S / (((X[0, i] - obj_Mux[j][0, i]) ** 2) + ((X[1, i] - obj_Mux[j][1, i]) ** 2))
        #                 obj = obj + Obj_cost
        #     if obj_info['Left_preceding'][0] != None:
        #         if obj_Mux[j][3, 0] == obj_info['Left_preceding'][0].id:
        #             for i in range(self.Np):
        #                 Obj_cost = self.S / (((X[0, i] - obj_Mux[j][0, i]) ** 2) + ((X[1, i] - obj_Mux[j][1, i]) ** 2))
        #                 obj = obj + Obj_cost
        #     if obj_info['Left_following'][0] != None:
        #         if obj_Mux[j][3, 0] == obj_info['Left_following'][0].id:
        #             for i in range(self.Np):
        #                 Obj_cost = self.S / (((X[0, i] - obj_Mux[j][0, i]) ** 2) + ((X[1, i] - obj_Mux[j][1, i]) ** 2))
        #                 obj = obj + Obj_cost
        #     if obj_info['Right_preceding'][0] != None:
        #         if obj_Mux[j][3, 0] == obj_info['Right_preceding'][0].id:
        #             for i in range(self.Np):
        #                 Obj_cost = self.S / (((X[0, i] - obj_Mux[j][0, i]) ** 2) + ((X[1, i] - obj_Mux[j][1, i]) ** 2))
        #                 obj = obj + Obj_cost
        #     if obj_info['Right_following'][0] != None:
        #         if obj_Mux[j][3, 0] == obj_info['Right_following'][0].id:
        #             for i in range(self.Np):
        #                 Obj_cost = self.S / (((X[0, i] - obj_Mux[j][0, i]) ** 2) + ((X[1, i] - obj_Mux[j][1, i]) ** 2))
        #                 obj = obj + Obj_cost
        #     if obj_info['Left'][0] != None:
        #         if obj_Mux[j][3, 0] == obj_info['Left'][0].id:
        #             for i in range(self.Np):
        #                 Obj_cost = self.S / (((X[0, i] - obj_Mux[j][0, i]) ** 2) + ((X[1, i] - obj_Mux[j][1, i]) ** 2))
        #                 obj = obj + Obj_cost
        #     if obj_info['Right'][0] != None:
        #         if obj_Mux[j][3, 0] == obj_info['Right'][0].id:
        #             for i in range(self.Np):
        #                 Obj_cost = self.S / (((X[0, i] - obj_Mux[j][0, i]) ** 2) + ((X[1, i] - obj_Mux[j][1, i]) ** 2))
        #                 obj = obj + Obj_cost
        #     else:
        #         obj = obj

        # Obstacle avoidance cost function
        # for i in range(self.Np):
        #     if obj_info['Ego_preceding'][0] != None:
        #         Obj_cost = self.S / (((X[0, i] - self.obj_ego_preceding[0, i]) ** 2) + (
        #                     (X[1, i] - self.obj_ego_preceding[0, i]) ** 2))
        #         obj = obj + Obj_cost
        #     if obj_info['Ego_following'][0] != None:
        #         Obj_cost = self.S / (((X[0, i] - self.obj_ego_following[0, i]) ** 2) + (
        #                     (X[1, i] - self.obj_ego_following[0, i]) ** 2))
        #         obj = obj + Obj_cost
        #     if obj_info['Left_preceding'][0] != None:
        #         Obj_cost = self.S / (((X[0, i] - self.obj_left_preceding[0, i]) ** 2) + (
        #                     (X[1, i] - self.obj_left_preceding[0, i]) ** 2))
        #         obj = obj + Obj_cost
        #     if obj_info['Left_following'][0] != None:
        #         Obj_cost = self.S / (((X[0, i] - self.obj_left_following[0, i]) ** 2) + (
        #                     (X[1, i] - self.obj_left_following[0, i]) ** 2))
        #         obj = obj + Obj_cost
        #     if obj_info['Right_preceding'][0] != None:
        #         Obj_cost = self.S / (((X[0, i] - self.obj_right_preceding[0, i]) ** 2) + (
        #                     (X[1, i] - self.obj_right_preceding[0, i]) ** 2))
        #         obj = obj + Obj_cost
        #     if obj_info['Right_following'][0] != None:
        #         Obj_cost = self.S / (((X[0, i] - self.obj_right_following[0, i]) ** 2) + (
        #                     (X[1, i] - self.obj_right_following[0, i]) ** 2))
        #         obj = obj + Obj_cost
        #     if obj_info['Left'][0] != None:
        #         Obj_cost = self.S / (((X[0, i] - self.obj_left[0, i]) ** 2) + ((X[1, i] - self.obj_left[0, i]) ** 2))
        #         obj = obj + Obj_cost
        #     if obj_info['Right'][0] != None:
        #         Obj_cost = self.S / (((X[0, i] - self.obj_right[0, i]) ** 2) + ((X[1, i] - self.obj_right[0, i]) ** 2))
        #         obj = obj + Obj_cost
        #     else:
        #         obj = obj

        # Terminal cost function
        Ref_ter_1 = ca.mtimes([(X[:, -1] - C_R[3:6]).T, self.Q1, X[:, -1] - C_R[3:6]])
        Ref_ter_2 = ca.mtimes([(X[:, -1] - C_R[6:9]).T, self.Q2, X[:, -1] - C_R[6:9]])
        obj = obj + Ref_ter_1 + Ref_ter_2

        # constraint 1: dynamic constraint
        for i in range(self.Np - 1):
            if i in range(self.Nc):
                x_next_ = f(X[:, i], U[:, i]) * self.T + X[:, i]
            else:
                x_next_ = f(X[:, i], U[:, self.Nc - 1]) * self.T + X[:, i]
            g1.append(X[:, i + 1] - x_next_)

        # constraint 2: dU constraint
        for i in range(self.Nc):
            if i == 0:
                g2.append((U[0, 0] - u_last[0]) / self.T)
                g2.append((U[1, 0] - u_last[1]) / self.T)
            else:
                g2.append((U[:, i] - U[:, i - 1]) / self.T)

        # constraint 3: Obstacle avoidance
        for i in range(self.Np):
            vehicle_ego_center1_x = X[0, i] - self.Length / 4 * np.cos(X[2, i])
            vehicle_ego_center1_y = X[1, i] - self.Length / 4 * np.sin(X[2, i])
            vehicle_ego_center2_x = X[0, i] + self.Length / 4 * np.cos(X[2, i])
            vehicle_ego_center2_y = X[1, i] + self.Length / 4 * np.sin(X[2, i])

            for j in range(np.size(obj_info['Obj_actor'])):
                if obj_Mux[j][4, 0] != 0:    #'not vehicle_around'
                    vehicle_obs_center1_x = obj_Mux[j][0, i] - self.Length / 4 * np.cos(obj_Mux[j][2, i])
                    vehicle_obs_center1_y = obj_Mux[j][1, i] - self.Length / 4 * np.sin(obj_Mux[j][2, i])
                    vehicle_obs_center2_x = obj_Mux[j][0, i] + self.Length / 4 * np.cos(obj_Mux[j][2, i])
                    vehicle_obs_center2_y = obj_Mux[j][1, i] + self.Length / 4 * np.sin(obj_Mux[j][2, i])
                    g3.append((vehicle_ego_center1_x - vehicle_obs_center1_x) ** 2 + (
                            vehicle_ego_center1_y - vehicle_obs_center1_y) ** 2)
                    g3.append((vehicle_ego_center1_x - vehicle_obs_center2_x) ** 2 + (
                            vehicle_ego_center1_y - vehicle_obs_center2_y) ** 2)
                    g3.append((vehicle_ego_center2_x - vehicle_obs_center1_x) ** 2 + (
                            vehicle_ego_center2_y - vehicle_obs_center1_y) ** 2)
                    g3.append((vehicle_ego_center2_x - vehicle_obs_center2_x) ** 2 + (
                            vehicle_ego_center2_y - vehicle_obs_center2_y) ** 2)

        # for i in range(self.Np):
        #     vehicle_ego_center1_x = X[0, i] - self.Length / 4 * np.cos(X[2, i])
        #     vehicle_ego_center1_y = X[1, i] - self.Length / 4 * np.sin(X[2, i])
        #     vehicle_ego_center2_x = X[0, i] + self.Length / 4 * np.cos(X[2, i])
        #     vehicle_ego_center2_y = X[1, i] + self.Length / 4 * np.sin(X[2, i])
        #     if obj_info['Ego_preceding'][0] != None:
        #         vehicle_ego_preceding_obs_center1_x = self.obj_ego_preceding[0, i] - self.Length / 4 * np.cos(
        #             self.obj_ego_preceding[2, i])
        #         vehicle_ego_preceding_obs_center1_y = self.obj_ego_preceding[1, i] - self.Length / 4 * np.sin(
        #             self.obj_ego_preceding[2, i])
        #         vehicle_ego_preceding_obs_center2_x = self.obj_ego_preceding[0, i] + self.Length / 4 * np.cos(
        #             self.obj_ego_preceding[2, i])
        #         vehicle_ego_preceding_obs_center2_y = self.obj_ego_preceding[1, i] + self.Length / 4 * np.sin(
        #             self.obj_ego_preceding[2, i])
        #         g3.append((vehicle_ego_center1_x - vehicle_ego_preceding_obs_center1_x) ** 2 + (
        #                 vehicle_ego_center1_y - vehicle_ego_preceding_obs_center1_y) ** 2)
        #         g3.append((vehicle_ego_center1_x - vehicle_ego_preceding_obs_center2_x) ** 2 + (
        #                 vehicle_ego_center1_y - vehicle_ego_preceding_obs_center2_y) ** 2)
        #         g3.append((vehicle_ego_center2_x - vehicle_ego_preceding_obs_center1_x) ** 2 + (
        #                 vehicle_ego_center2_y - vehicle_ego_preceding_obs_center1_y) ** 2)
        #         g3.append((vehicle_ego_center2_x - vehicle_ego_preceding_obs_center2_x) ** 2 + (
        #                 vehicle_ego_center2_y - vehicle_ego_preceding_obs_center2_y) ** 2)
        #     if obj_info['Ego_following'][0] != None:
        #         vehicle_ego_following_obs_center1_x = self.obj_ego_following[0, i] - self.Length / 4 * np.cos(
        #             self.obj_ego_following[2, i])
        #         vehicle_ego_following_obs_center1_y = self.obj_ego_following[1, i] - self.Length / 4 * np.sin(
        #             self.obj_ego_following[2, i])
        #         vehicle_ego_following_obs_center2_x = self.obj_ego_following[0, i] + self.Length / 4 * np.cos(
        #             self.obj_ego_following[2, i])
        #         vehicle_ego_following_obs_center2_y = self.obj_ego_following[1, i] + self.Length / 4 * np.sin(
        #             self.obj_ego_following[2, i])
        #         g3.append((vehicle_ego_center1_x - vehicle_ego_following_obs_center1_x) ** 2 + (
        #                 vehicle_ego_center1_y - vehicle_ego_following_obs_center1_y) ** 2)
        #         g3.append((vehicle_ego_center1_x - vehicle_ego_following_obs_center2_x) ** 2 + (
        #                 vehicle_ego_center1_y - vehicle_ego_following_obs_center2_y) ** 2)
        #         g3.append((vehicle_ego_center2_x - vehicle_ego_following_obs_center1_x) ** 2 + (
        #                 vehicle_ego_center2_y - vehicle_ego_following_obs_center1_y) ** 2)
        #         g3.append((vehicle_ego_center2_x - vehicle_ego_following_obs_center2_x) ** 2 + (
        #                 vehicle_ego_center2_y - vehicle_ego_following_obs_center2_y) ** 2)
        #     if obj_info['Left_preceding'][0] != None:
        #         vehicle_left_preceding_obs_center1_x = self.obj_left_preceding[0, i] - self.Length / 4 * np.cos(
        #             self.obj_left_preceding[2, i])
        #         vehicle_left_preceding_obs_center1_y = self.obj_left_preceding[1, i] - self.Length / 4 * np.sin(
        #             self.obj_left_preceding[2, i])
        #         vehicle_left_preceding_obs_center2_x = self.obj_left_preceding[0, i] + self.Length / 4 * np.cos(
        #             self.obj_left_preceding[2, i])
        #         vehicle_left_preceding_obs_center2_y = self.obj_left_preceding[1, i] + self.Length / 4 * np.sin(
        #             self.obj_left_preceding[2, i])
        #         g3.append((vehicle_ego_center1_x - vehicle_left_preceding_obs_center1_x) ** 2 + (
        #                 vehicle_ego_center1_y - vehicle_left_preceding_obs_center1_y) ** 2)
        #         g3.append((vehicle_ego_center1_x - vehicle_left_preceding_obs_center2_x) ** 2 + (
        #                 vehicle_ego_center1_y - vehicle_left_preceding_obs_center2_y) ** 2)
        #         g3.append((vehicle_ego_center2_x - vehicle_left_preceding_obs_center1_x) ** 2 + (
        #                 vehicle_ego_center2_y - vehicle_left_preceding_obs_center1_y) ** 2)
        #         g3.append((vehicle_ego_center2_x - vehicle_left_preceding_obs_center2_x) ** 2 + (
        #                 vehicle_ego_center2_y - vehicle_left_preceding_obs_center2_y) ** 2)
        #     if obj_info['Left_following'][0] != None:
        #         vehicle_left_following_obs_center1_x = self.obj_left_following[0, i] - self.Length / 4 * np.cos(
        #             self.obj_left_following[2, i])
        #         vehicle_left_following_obs_center1_y = self.obj_left_following[1, i] - self.Length / 4 * np.sin(
        #             self.obj_left_following[2, i])
        #         vehicle_left_following_obs_center2_x = self.obj_left_following[0, i] + self.Length / 4 * np.cos(
        #             self.obj_left_following[2, i])
        #         vehicle_left_following_obs_center2_y = self.obj_left_following[1, i] + self.Length / 4 * np.sin(
        #             self.obj_left_following[2, i])
        #         g3.append((vehicle_ego_center1_x - vehicle_left_following_obs_center1_x) ** 2 + (
        #                 vehicle_ego_center1_y - vehicle_left_following_obs_center1_y) ** 2)
        #         g3.append((vehicle_ego_center1_x - vehicle_left_following_obs_center2_x) ** 2 + (
        #                 vehicle_ego_center1_y - vehicle_left_following_obs_center2_y) ** 2)
        #         g3.append((vehicle_ego_center2_x - vehicle_left_following_obs_center1_x) ** 2 + (
        #                 vehicle_ego_center2_y - vehicle_left_following_obs_center1_y) ** 2)
        #         g3.append((vehicle_ego_center2_x - vehicle_left_following_obs_center2_x) ** 2 + (
        #                 vehicle_ego_center2_y - vehicle_left_following_obs_center2_y) ** 2)
        #     if obj_info['Right_preceding'][0] != None:
        #         vehicle_right_preceding_obs_center1_x = self.obj_right_preceding[0, i] - self.Length / 4 * np.cos(
        #             self.obj_right_preceding[2, i])
        #         vehicle_right_preceding_obs_center1_y = self.obj_right_preceding[1, i] - self.Length / 4 * np.sin(
        #             self.obj_right_preceding[2, i])
        #         vehicle_right_preceding_obs_center2_x = self.obj_right_preceding[0, i] + self.Length / 4 * np.cos(
        #             self.obj_right_preceding[2, i])
        #         vehicle_right_preceding_obs_center2_y = self.obj_right_preceding[1, i] + self.Length / 4 * np.sin(
        #             self.obj_right_preceding[2, i])
        #         g3.append((vehicle_ego_center1_x - vehicle_right_preceding_obs_center1_x) ** 2 + (
        #                 vehicle_ego_center1_y - vehicle_right_preceding_obs_center1_y) ** 2)
        #         g3.append((vehicle_ego_center1_x - vehicle_right_preceding_obs_center2_x) ** 2 + (
        #                 vehicle_ego_center1_y - vehicle_right_preceding_obs_center2_y) ** 2)
        #         g3.append((vehicle_ego_center2_x - vehicle_right_preceding_obs_center1_x) ** 2 + (
        #                 vehicle_ego_center2_y - vehicle_right_preceding_obs_center1_y) ** 2)
        #         g3.append((vehicle_ego_center2_x - vehicle_right_preceding_obs_center2_x) ** 2 + (
        #                 vehicle_ego_center2_y - vehicle_right_preceding_obs_center2_y) ** 2)
        #     if obj_info['Right_following'][0] != None:
        #         vehicle_right_following_obs_center1_x = self.obj_right_following[0, i] - self.Length / 4 * np.cos(
        #             self.obj_right_following[2, i])
        #         vehicle_right_following_obs_center1_y = self.obj_right_following[1, i] - self.Length / 4 * np.sin(
        #             self.obj_right_following[2, i])
        #         vehicle_right_following_obs_center2_x = self.obj_right_following[0, i] + self.Length / 4 * np.cos(
        #             self.obj_right_following[2, i])
        #         vehicle_right_following_obs_center2_y = self.obj_right_following[1, i] + self.Length / 4 * np.sin(
        #             self.obj_right_following[2, i])
        #         g3.append((vehicle_ego_center1_x - vehicle_right_following_obs_center1_x) ** 2 + (
        #                 vehicle_ego_center1_y - vehicle_right_following_obs_center1_y) ** 2)
        #         g3.append((vehicle_ego_center1_x - vehicle_right_following_obs_center2_x) ** 2 + (
        #                 vehicle_ego_center1_y - vehicle_right_following_obs_center2_y) ** 2)
        #         g3.append((vehicle_ego_center2_x - vehicle_right_following_obs_center1_x) ** 2 + (
        #                 vehicle_ego_center2_y - vehicle_right_following_obs_center1_y) ** 2)
        #         g3.append((vehicle_ego_center2_x - vehicle_right_following_obs_center2_x) ** 2 + (
        #                 vehicle_ego_center2_y - vehicle_right_following_obs_center2_y) ** 2)
        #
        #     if obj_info['Left'][0] != None:
        #         vehicle_left_obs_center1_x = self.obj_left[0, i] - self.Length / 4 * np.cos(
        #             self.obj_left[2, i])
        #         vehicle_left_obs_center1_y = self.obj_left[1, i] - self.Length / 4 * np.sin(
        #             self.obj_left[2, i])
        #         vehicle_left_obs_center2_x = self.obj_left[0, i] + self.Length / 4 * np.cos(
        #             self.obj_left[2, i])
        #         vehicle_left_obs_center2_y = self.obj_left[1, i] + self.Length / 4 * np.sin(
        #             self.obj_left[2, i])
        #         g3.append((vehicle_ego_center1_x - vehicle_left_obs_center1_x) ** 2 + (
        #                 vehicle_ego_center1_y - vehicle_left_obs_center1_y) ** 2)
        #         g3.append((vehicle_ego_center1_x - vehicle_left_obs_center2_x) ** 2 + (
        #                 vehicle_ego_center1_y - vehicle_left_obs_center2_y) ** 2)
        #         g3.append((vehicle_ego_center2_x - vehicle_left_obs_center1_x) ** 2 + (
        #                 vehicle_ego_center2_y - vehicle_left_obs_center1_y) ** 2)
        #         g3.append((vehicle_ego_center2_x - vehicle_left_obs_center2_x) ** 2 + (
        #                 vehicle_ego_center2_y - vehicle_left_obs_center2_y) ** 2)
        #     if obj_info['Right'][0] != None:
        #         vehicle_right_obs_center1_x = self.obj_right[0, i] - self.Length / 4 * np.cos(
        #             self.obj_right[2, i])
        #         vehicle_right_obs_center1_y = self.obj_right[1, i] - self.Length / 4 * np.sin(
        #             self.obj_right[2, i])
        #         vehicle_right_obs_center2_x = self.obj_right[0, i] + self.Length / 4 * np.cos(
        #             self.obj_right[2, i])
        #         vehicle_right_obs_center2_y = self.obj_right[1, i] + self.Length / 4 * np.sin(
        #             self.obj_right[2, i])
        #         g3.append((vehicle_ego_center1_x - vehicle_right_obs_center1_x) ** 2 + (
        #                 vehicle_ego_center1_y - vehicle_right_obs_center1_y) ** 2)
        #         g3.append((vehicle_ego_center1_x - vehicle_right_obs_center2_x) ** 2 + (
        #                 vehicle_ego_center1_y - vehicle_right_obs_center2_y) ** 2)
        #         g3.append((vehicle_ego_center2_x - vehicle_right_obs_center1_x) ** 2 + (
        #                 vehicle_ego_center2_y - vehicle_right_obs_center1_y) ** 2)
        #         g3.append((vehicle_ego_center2_x - vehicle_right_obs_center2_x) ** 2 + (
        #                 vehicle_ego_center2_y - vehicle_right_obs_center2_y) ** 2)


        # 定义优化问题
        # 输入变量，这里需要同时将系统状态X也作为优化变量输入，
        # 根据CasADi要求，必须将它们都变形为一维向量
        opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

        # 定义NLP问题，'f'为目标函数，'x'为需寻找的优化结果（优化目标变量），'p'为系统参数，'g'为约束条件
        # 需要注意的是，用SX表达必须将所有表示成标量或者是一维矢量的形式
        nlp_prob = {'f': obj, 'x': opt_variables, 'p': C_R, 'g': ca.vertcat(*g1, *g2, *g3)}

        # ipopt设置
        opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 5, 'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}

        # 最终目标，获得求解器
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

        # 状态约束
        lbg = []
        ubg = []

        # g1
        for _ in range(self.Np):
            lbg.append(0)
            lbg.append(0)
            lbg.append(0)
            ubg.append(0)
            ubg.append(0)
            ubg.append(0)

        # g2
        for _ in range(self.Nc):
            lbg.append(self.d_v_min)
            lbg.append(self.d_delta_f_min)
            ubg.append(self.d_v_max)
            ubg.append(self.d_delta_f_max)

        # g3
        for i in range(self.Np):
            # for _ in range(np.size(obj_info['Obj_actor'])*4):
            for _ in range(vehicle_num * 4):
                lbg.append(4 * ((self.Length / 4) ** 2 + (self.Width / 2) ** 2))
                ubg.append(np.inf)


        # 控制约束
        lbx = []
        ubx = []

        # v,delta_f,x,y,phi
        for _ in range(self.Nc):
            lbx.append(self.v_min)
            lbx.append(self.delta_f_min)
            ubx.append(self.v_max)
            ubx.append(self.delta_f_max)

        for i in range(self.Np):
            y_min = frenet_to_inertial(fpath.s[i], fpath.d[i] - 4.5, csp)[1]
            y_max = frenet_to_inertial(fpath.s[i], fpath.d[i] + 1.5, csp)[1]
            lbx.append(-np.inf)
            # lbx.append(-np.inf)  #y
            lbx.append(25.5)
            # lbx.append(y_min)
            lbx.append(-np.inf)
            if obj_info['Ego_preceding'][0] != None:  # if no vehicle_ahead
                # ubx.append(self.obj_ego_preceding[0, i])
                ubx.append(self.obj_ego_preceding[0, i] - self.stop_line)
            # if vehicle_ahead != None:  # if no vehicle_ahead
            #     ubx.append(obj_Mux[vehicle_ahead][0, i] - self.stop_line)
            else:
                ubx.append(np.inf)
            # ubx.append(y_max)
            ubx.append(31)
            # ubx.append(np.inf)   #y
            ubx.append(np.inf)

        # index_t = []
        # 初始化优化参数
        C_R = np.array([x_current[0], x_current[1], x_current[2], ref[0], ref[1], ref[2],
                        ref_left[0], ref_left[1], ref_left[2]])

        # 初始化优化目标变量
        init_control = np.concatenate((self.u0.reshape(-1, 1), self.next_states.reshape(-1, 1)))
        t_ = time.time()
        res = solver(x0=init_control, p=C_R, lbg=lbg,
                     lbx=lbx, ubg=ubg, ubx=ubx)
        # index_t.append(time.time() - t_)
        # the feedback is in the series [u0, x0, u1, x1, ...]
        # 获得最优控制结果estimated_opt，u0，x_m
        estimated_opt = res['x'].full()
        u0 = estimated_opt[:self.Nc * self.Nu].reshape(self.Nc, self.Nu)
        x_m = estimated_opt[self.Nc * self.Nu:].reshape(self.Np, self.Nx)
        self.next_states = np.concatenate((x_m[1:], x_m[-1:]), axis=0)

        self.u0 = np.concatenate((u0[1:], u0[-1:]))

        print(estimated_opt[0])
        print(estimated_opt[1])
        print('t_cost =', time.time() - start_time)
        MPC_unsolved = False
        return np.array([estimated_opt[0], estimated_opt[1]]), MPC_unsolved, x_m
