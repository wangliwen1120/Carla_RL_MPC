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
        self.vehicle_nums = 5  # TRAFFIC_MANAGER.N_SPAWN_CARS)
        self.walker_nums = 5  # TRAFFIC_MANAGER.N_SPAWN_PEDESTRAINS)
        self.Width_walker = 0.5

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
        self.obj_x_ref = np.zeros((self.Np, 1))
        self.obj_y_ref = np.zeros((self.Np, 1))
        self.obj_phi_ref = np.zeros((self.Np, 1))
        self.obj_actor_id = np.zeros((self.Np, 1))
        self.obj_pos = np.zeros((self.Np, 1))

        self.next_states = np.zeros((self.Nx, self.Np)).copy().T
        self.u0 = np.array([0, 0] * self.Nc).reshape(-1, 2).T

        self.model_ocp()
        self.cons_g()

    def solve_walker(self, walker_info):
        self.walker_Mux = []
        for j in range(np.size(walker_info['Walker_actor'])):
            obj_x = walker_info['Walker_cartesian'][j][0]
            obj_y = walker_info['Walker_cartesian'][j][1]
            obj_phi = walker_info['Walker_cartesian'][j][4]
            obj_speed = walker_info['Walker_cartesian'][j][5]
            obj_delta_f = walker_info['Walker_cartesian'][j][6]
            for i in range(self.Np):
                self.obj_x_ref[i] = obj_x + obj_speed * np.cos(obj_phi) * self.T * i
                self.obj_y_ref[i] = obj_y + obj_speed * np.sin(obj_phi) * self.T * i
                self.obj_phi_ref[i] = obj_phi + obj_delta_f * self.T * i
                self.obj_actor_id[i] = walker_info['Walker_actor'][j].id
            self.walker_Mux.append(
                np.concatenate((self.obj_x_ref.T, self.obj_y_ref.T, self.obj_phi_ref.T, self.obj_actor_id.T)))

    def solve_obj(self, obj_info):
        self.obj_Mux = []
        self.vehicle_num = 0

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
                        self.obj_pos[i] = 1  # 'Ego_preceding'
                        self.vehicle_num += 1
                if obj_info['Ego_following'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Ego_following'][0].id:
                        self.obj_pos[i] = 2  # 'Ego_following'
                        self.vehicle_num += 1
                if obj_info['Left_preceding'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Left_preceding'][0].id:
                        self.obj_pos[i] = 3  # 'Ego_preceding'
                        self.vehicle_num += 1
                if obj_info['Left_following'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Left_following'][0].id:
                        self.obj_pos[i] = 4  # 'Left_following'
                        self.vehicle_num += 1
                if obj_info['Right_preceding'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Right_preceding'][0].id:
                        self.obj_pos[i] = 5  # 'Right_preceding'
                        self.vehicle_num += 1
                if obj_info['Right_following'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Right_following'][0].id:
                        self.obj_pos[i] = 6  # 'Right_following'
                        self.vehicle_num += 1
                if obj_info['Left'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Left'][0].id:
                        self.obj_pos[i] = 7  # 'Left'
                        self.vehicle_num += 1
                if obj_info['Right'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Right'][0].id:
                        self.obj_pos[i] = 8  # 'Right'
                        self.vehicle_num += 1
            self.obj_Mux.append(np.concatenate(
                (self.obj_x_ref.T, self.obj_y_ref.T, self.obj_phi_ref.T, self.obj_actor_id.T, self.obj_pos.T)))
        self.vehicle_num = int(self.vehicle_num / self.Np)

    def cons_g(self):
        # 状态约束
        self.lbg = []
        self.ubg = []

        # g1
        for _ in range(self.Np):
            self.lbg.append(0)
            self.lbg.append(0)
            self.lbg.append(0)
            self.ubg.append(0)
            self.ubg.append(0)
            self.ubg.append(0)

        # g2
        for _ in range(self.Nc):
            self.lbg.append(self.d_v_min)
            self.lbg.append(self.d_delta_f_min)
            self.ubg.append(self.d_v_max)
            self.ubg.append(self.d_delta_f_max)

        # g3
        for i in range(self.Np):
            for _ in range(self.vehicle_nums + self.walker_nums):
                self.lbg.append(-np.inf)
                self.ubg.append(0)

    def model_ocp(self):
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
        C_R = ca.SX.sym('C_R', 1 + self.Nu + self.Nx + self.Nx + self.Nx +
                        self.vehicle_nums * self.Np * 6 + self.walker_nums * self.Np * 5)  # 构建问题的相关参数
        # 这里给定当前/初始位置，目标终点(本车道/左车道)位置

        # 权重矩阵
        self.q = C_R[0]
        self.ru = 0
        self.rdu = 0.3
        self.S = 0.1  # Obstacle avoidance function coefficient
        self.Q1 = self.q * np.eye(self.Nx)  # ego_lane: lane_2
        self.Q2 = (1 - self.q) * np.eye(self.Nx)  # left_lane: lane_1
        self.Ru = self.ru * np.eye(self.Nu)
        self.Rdu = self.rdu * np.eye(self.Nu)

        # cost function
        obj = 0  # 初始化优化目标值

        # U dU cost function
        for i in range(self.Nc):
            if i == 0:
                dU_cost = 0
            else:
                dU_cost = ca.mtimes([(U[:, i] - U[:, i - 1]).T, self.Rdu, (U[:, i] - U[:, i - 1])])
            U_cost = ca.mtimes([U[:, i].T, self.Ru, U[:, i]])
            obj = obj + U_cost + dU_cost

        # Obstacle avoidance cost function
        for j in range(self.vehicle_nums):
            k = 12 + j * self.Np * 6
            # if C_R[k+3] != 0:  # 'not vehicle_around'   #all vehicle to cal mot vehicle around
            for i in range(self.Np):
                Obj_cost = self.S / (((X[0, i] - ca.if_else(C_R[k + 3 + i * 6] != 0, C_R[k + i * 6], 1e6)) ** 2) +
                                     ((X[1, i] - ca.if_else(C_R[k + 3 + i * 6] != 0, C_R[k + 1 + i * 6], 1e6)) ** 2))
                obj = obj + Obj_cost

        # for jj in range(self.walker_nums):
        #     kk = 12 + self.vehicle_nums * self.Np * 6 + jj * self.Np * 5
        #     # if C_R[k+3] != 0:  # 'not vehicle_around'   #all vehicle to cal mot vehicle around
        #     for i in range(self.Np):
        #         Obj_cost = self.S / (((X[0, i] - ca.if_else(C_R[kk+3+i*5] != 0,C_R[kk+i*5],1e6)) ** 2) +
        #                              ((X[1, i] - ca.if_else(C_R[kk+3+i*5] != 0,C_R[kk+1+i*5],1e6)) ** 2))
        #         obj = obj + Obj_cost

        # Terminal cost function

        Ref_ter_1 = ca.mtimes([(X[:, -1] - C_R[6:9]).T, self.Q1, X[:, -1] - C_R[6:9]])
        Ref_ter_2 = ca.mtimes([(X[:, -1] - C_R[9:12]).T, self.Q2, X[:, -1] - C_R[9:12]])
        obj = obj + Ref_ter_1 + Ref_ter_2

        g1 = []  # 用list来存储优化目标的向量
        g2 = []
        g3 = []
        # constraint 1: dynamic constraint
        g1.append(X[:, 0] - C_R[3:6])
        for i in range(self.Np - 1):
            if i in range(self.Nc):
                x_next_ = f(X[:, i], U[:, i]) * self.T + X[:, i]
            else:
                x_next_ = f(X[:, i], U[:, self.Nc - 1]) * self.T + X[:, i]
            g1.append(X[:, i + 1] - x_next_)

        # constraint 2: dU constraint
        for i in range(self.Nc):
            if i == 0:
                g2.append((U[0, 0] - C_R[1]) / self.T)
                g2.append((U[1, 0] - C_R[2]) / self.T)
            else:
                g2.append((U[:, i] - U[:, i - 1]) / self.T)

        # constraint 3: Obstacle avoidance
        for i in range(self.Np):
            for j in range(self.vehicle_nums):
                k = 12 + self.Np * j * 6
                if i == 0:
                    cons_obs_h = X[1, i] - C_R[k + 1 + i * 6] + C_R[k + 4 + i * 6] * self.Width
                else:
                    cons_obs_h = X[1, i] + (X[0, i] - X[0, i - 1]) * X[2, i] - C_R[k + 1 + i * 6] + C_R[
                        k + 4 + i * 6] * self.Width
                g3.append(C_R[k + 4 + i * 6] * cons_obs_h + C_R[k + 5 + i * 6])

            for jj in range(self.walker_nums):
                kk = 12 + self.Np * self.vehicle_nums * 6 + self.Np * jj * 5
                if i == 0:
                    cons_obs_h = X[1, i] - C_R[kk + 1 + i * 5] + C_R[kk + 3 + i * 5] * self.Width_walker
                else:
                    cons_obs_h = X[1, i] + (X[0, i] - X[0, i - 1]) * X[2, i] - C_R[kk + 1 + i * 5] + C_R[
                        kk + 3 + i * 5] * self.Width_walker
                g3.append(C_R[kk + 3 + i * 5] * cons_obs_h + C_R[kk + 4 + i * 5])

        # 定义优化问题
        # 输入变量，这里需要同时将系统状态X也作为优化变量输入，
        # 根据CasADi要求，必须将它们都变形为一维向量
        opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

        # 定义NLP问题，'f'为目标函数，'x'为需寻找的优化结果（优化目标变量），'p'为系统参数，'g'为约束条件
        # 需要注意的是，用SX表达必须将所有表示成标量或者是一维矢量的形式
        nlp_prob = {'f': obj, 'x': opt_variables, 'p': C_R, 'g': ca.vertcat(*g1, *g2, *g3)}

        # ipopt设置
        opts_setting = {'ipopt.max_iter': 20, 'ipopt.print_level': 0, 'print_time': 0,
                        'ipopt.acceptable_tol': 1e-6, 'ipopt.acceptable_obj_change_tol': 1e-6}

        # 最终目标，获得求解器
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    def calc_input(self, x_current, obj_info, walker_info, ref, ref_left,ref_right, u_last, csp, fpath, q, fpath_point_num):
        ego_f = np.zeros(self.Np)
        ego_r = np.zeros(self.Np)
        current_x_list = np.zeros(self.Np)
        current_y_list = np.zeros(self.Np)
        for i in range(self.Np):
            current_x_list[i] = x_current[0] + u_last[0] * np.cos(x_current[2]) * self.T * i
            current_y_list[i] = x_current[1] + u_last[0] * np.sin(x_current[2]) * self.T * i
            current_phi_list = x_current[2] + u_last[1] * self.T * i

            ego_x_fl = current_x_list[i] - self.Width / 2 * np.sin(current_phi_list) + self.Length / 2 * np.cos(
                current_phi_list)
            ego_x_fr = current_x_list[i] + self.Width / 2 * np.sin(current_phi_list) + self.Length / 2 * np.cos(
                current_phi_list)
            ego_x_rl = current_x_list[i] - self.Width / 2 * np.sin(current_phi_list) - self.Length / 2 * np.cos(
                current_phi_list)
            ego_x_rr = current_x_list[i] + self.Width / 2 * np.sin(current_phi_list) - self.Length / 2 * np.cos(
                current_phi_list)
            ego_f[i] = max(ego_x_fl, ego_x_fr)
            ego_r[i] = min(ego_x_rl, ego_x_rr)

        self.solve_obj(obj_info)
        self.solve_walker(walker_info)
        obj_Mux = self.obj_Mux
        walker_Mux = self.walker_Mux

        obs_list = []  # obs_list:  1. 车的数目  2. x, y, phi, id, l(1) or r(-1),cons or not(0/-1e5) 3.self.Np
        index = [0, 1, 2, 4]
        for i in range(self.vehicle_nums):
            for j in range(self.Np):
                for k in index:
                    obs_list.append(obj_Mux[i][k][j])
                obs_list.append(1 if current_y_list[j] < obj_Mux[i][1][j] else -1)
                obs_list.append(
                    0 if ego_f[j] > obj_Mux[i][0][j] and ego_r[j] < (obj_Mux[i][0][j] + self.Length) else -1e5)

        walker_list = []  # x, y, phi, l(1) or r(-1),cons or not(0/-1e5) 3.self.Np
        for i in range(self.walker_nums):
            for j in range(self.Np):
                for k in range(3):
                    walker_list.append(walker_Mux[i][k][j])
                walker_list.append(1 if current_y_list[j] < walker_Mux[i][1][j] else -1)
                walker_list.append(
                    0 if ego_f[j] > walker_Mux[i][0][j] and ego_r[j] < (walker_Mux[i][0][j] + self.Length) else -1e5)

        # 初始化优化参数
        if q<0:   ## -1: left  0: ref   1:right
            C_R = np.concatenate(([-q, u_last[0], u_last[1]], x_current, ref_left[:3], ref[:3], obs_list, walker_list))
        else:
            C_R = np.concatenate(([q, u_last[0], u_last[1]], x_current, ref_right[:3], ref[:3], obs_list, walker_list))

        # 控制约束
        self.lbx = []
        self.ubx = []

        # v,delta_f,x,y,phi
        for _ in range(self.Nc):
            self.lbx.append(self.v_min)
            self.lbx.append(self.delta_f_min)
            self.ubx.append(self.v_max)
            self.ubx.append(self.delta_f_max)

        for i in range(self.Np):
            if i<fpath_point_num:
                y_min = frenet_to_inertial(fpath.s[i], - 4.2, csp)[1]
                y_max = frenet_to_inertial(fpath.s[i], + 4.2, csp)[1]
            else:
                y_min = frenet_to_inertial(fpath.s[fpath_point_num], - 4.2, csp)[1]
                y_max = frenet_to_inertial(fpath.s[fpath_point_num], + 4.2, csp)[1]
            self.lbx.append(-np.inf)
            self.lbx.append(y_min)
            self.lbx.append(-np.inf)

            walker_x_max = np.inf
            for j in range(self.walker_nums):
                walker_x = walker_info['Walker_cartesian'][j][0]
                walker_y = walker_info['Walker_cartesian'][j][1]
                walker_vx = walker_info['Walker_cartesian'][j][2]
                walker_vy = walker_info['Walker_cartesian'][j][3]
                walker_y_i = walker_y + walker_vy * self.T * i

                walker_before = False
                if current_x_list[i] + self.Length / 2.0 < walker_x \
                        < current_x_list[i] + self.Length / 2.0 + self.stop_line * 2:
                    walker_before = True
                walker_width_danger = False
                if current_y_list[i] - self.Width / 2.0 - self.Width_walker/2.0 < walker_y_i \
                        < current_y_list[i] + self.Width / 2.0 + self.Width_walker/2.0:
                    walker_width_danger = True
                if walker_before and walker_width_danger:
                    walker_x_max = min(walker_x_max, walker_x + walker_vx * self.T * i - self.stop_line)

            if obj_info['Ego_preceding'][0] != None:  # if no vehicle_ahead
                obj_preceding_x = obj_info['Ego_preceding'][2][0]
                obj_preceding_y = obj_info['Ego_preceding'][2][1]
                obj_preceding_phi = obj_info['Ego_preceding'][2][4]
                obj_preceding_speed = obj_info['Ego_preceding'][2][5]
                obj_preceding_delta_f = obj_info['Ego_preceding'][2][6]
                obj_preceding_x_ref = obj_preceding_x + obj_preceding_speed * np.cos(obj_preceding_phi) * self.T * i
                walker_x_max = min(walker_x_max, obj_preceding_x_ref - self.stop_line)

            self.ubx.append(walker_x_max)
            self.ubx.append(y_max)
            self.ubx.append(np.inf)

        # 初始化优化目标变量
        init_control = np.concatenate((self.u0.reshape(-1, 1), self.next_states.reshape(-1, 1)))
        start_time = time.time()
        res = self.solver(x0=init_control, p=C_R, lbg=self.lbg,
                          lbx=self.lbx, ubg=self.ubg, ubx=self.ubx)
        # print('t_cost =', time.time() - start_time)
        # the feedback is in the series [u0, x0, u1, x1, ...]
        # 获得最优控制结果estimated_opt，u0，x_m
        estimated_opt = res['x'].full()
        u0 = estimated_opt[:self.Nc * self.Nu].reshape(self.Nc, self.Nu)
        x_m = estimated_opt[self.Nc * self.Nu:].reshape(self.Np, self.Nx)
        self.next_states = np.concatenate((x_m[1:], x_m[-1:]), axis=0)

        self.u0 = np.concatenate((u0[1:], u0[-1:]))
        MPC_unsolved = False
        stats = self.solver.stats()
        if not stats['success']:
            MPC_unsolved = True
            print("MPC does not have a solution.")

        return np.array([estimated_opt[0], estimated_opt[1]]), MPC_unsolved, x_m
