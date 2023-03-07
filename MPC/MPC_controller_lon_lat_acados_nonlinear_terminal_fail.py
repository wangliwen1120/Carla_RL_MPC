from __future__ import division
import numpy as np
import casadi as ca
import time
import math
import numba
import scipy
from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import os


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


class MPC_controller_lon_lat_acados_nonlinear_terminal:

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
        self.n_nodes = 100
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
        self.obj_x_ref = np.zeros((self.Np, 1))
        self.obj_y_ref = np.zeros((self.Np, 1))
        self.obj_phi_ref = np.zeros((self.Np, 1))
        self.obj_actor_id = np.zeros((self.Np, 1))
        self.obj_pos = np.zeros((self.Np, 1))

        self.next_states = np.zeros((self.Nx, self.Np)).copy().T
        self.u0 = np.array([0, 0] * self.Nc).reshape(-1, 2).T

    def calc_input(self, x_current, obj_info, ref, ref_left, u_last, csp, fpath, q, ru, rdu):
        model = AcadosModel()  # ca.types.SimpleNamespace()
        ocp = AcadosOcp()

        # 根据数学模型建模
        # 系统状态
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        phi = ca.SX.sym('theta')
        states = ca.vertcat(x, y, phi)

        # control inputs
        v = ca.SX.sym('v')
        delta_f = ca.SX.sym('delta_f')
        controls = ca.vertcat(v, delta_f)

        # dynamic_model
        state_trans = [v * ca.cos(phi), v * ca.sin(phi), v * ca.tan(delta_f) / self.L]

        # function
        # dynamic_model
        state_trans = [v * ca.cos(phi), v * ca.sin(phi), v * ca.tan(delta_f) / self.L]

        # function
        f = ca.Function('f', [states, controls], [ca.vcat(state_trans)], ['state', 'control_input'], ['state_trans'])
        # f_expl = ca.vcat(rhs)
        # acados model
        x_dot = ca.SX.sym('x_dot', len(state_trans))
        f_impl = x_dot - f(states, controls)

        # 开始构建MPC
        # 相关变量，格式(状态长度， 步长)
        U = ca.SX.sym('U', self.Nu, self.Nc)  # 控制输出
        X = ca.SX.sym('X', self.Nx, self.Np)  # 系统状态
        C_R = ca.SX.sym('C_R', self.Nx + self.Nx + self.Nx)
        opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

        # 权重矩阵
        self.q = 1
        self.ru = 0
        self.rdu = 0.3
        self.S = 0.1  # Obstacle avoidance function coefficient
        self.Q1 = self.q * np.eye(self.Nx)  # ego_lane: lane_2
        self.Q2 = (1 - self.q) * np.eye(self.Nx)  # left_lane: lane_1
        self.Ru = self.ru * np.eye(self.Nu)
        self.Rdu = self.rdu * np.eye(self.Nu)

        # cost function
        obj = 0  # 初始化优化目标值
        # Terminal cost function
        Ref_ter_1 = ca.mtimes([(X[:, -1] - C_R[3:6]).T, self.Q1, X[:, -1] - C_R[3:6]])
        Ref_ter_2 = ca.mtimes([(X[:, -1] - C_R[6:9]).T, self.Q2, X[:, -1] - C_R[6:9]])
        obj = obj + Ref_ter_1 + Ref_ter_2

        # dynamics: identity
        model.disc_dyn_expr = x
        model.x = x
        model.u = ca.SX.sym('u', 0, 0)
        model.p = []
        model.name = 'vehicle_model'
        ocp.model = model

        # discretization
        Tf = 1
        N = 1
        ocp.dims.N = N
        ocp.solver_options.tf = Tf

        # cost
        ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.model.cost_expr_ext_cost_e = x

        # constarints
        ocp.model.con_h_expr = x ** 2 + x ** 2
        ocp.constraints.lh = np.array([1.0])
        ocp.constraints.uh = np.array([1.0])

        # set options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = 'EXACT'
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = 'SQP'  # SQP_RTI, SQP
        ocp.solver_options.alpha_min = 1e-2
        ocp.solver_options.levenberg_marquardt = 1e-1
        ocp.solver_options.qp_solver_iter_max = 400
        ocp.solver_options.regularize_method = 'MIRROR'
        ocp.solver_options.eps_sufficient_descent = 1e-1
        ocp.solver_options.qp_tol = 5e-7
        ocp.solver_options.print_level = 0
        ocp_solver = AcadosOcpSolver(ocp, json_file=f'{model.name}.json')
        status = ocp_solver.solve()

        MPC_unsolved = False

        return np.array([1, 1]), MPC_unsolved, 1
