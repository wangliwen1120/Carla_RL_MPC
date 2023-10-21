from __future__ import division
import numpy as np
import casadi as ca
import timeit
import sys
import math
import numba
import scipy
import errno
import shutil
from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import os


def safe_mkdir_recursive(directory, overwrite=False):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(directory):
                pass
            else:
                raise
    else:
        if overwrite:
            try:
                shutil.rmtree(directory)
            except:
                print('Error while removing directory {}'.format(directory))


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
        self.d_delta_f_min = -0.082*3
        self.d_delta_f_max = 0.082*3

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

        # self.vehicle_model()
        self.vehicle_model_1()
        # self.ocp_problem()
        # self.ocp_problem_1()
        # self.ocp_problem_2()
        self.ocp_problem_3()

    def vehicle_model_1(self):
        model = AcadosModel()  # ca.types.SimpleNamespace()
        constraint = ca.types.SimpleNamespace()

        # original states
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        phi = ca.SX.sym('theta')

        # control inputs
        v = ca.SX.sym('v')
        delta_f = ca.SX.sym('delta_f')

        # previous control inputs
        v_prev = ca.SX.sym('v_prev')
        delta_f_prev = ca.SX.sym('delta_f_prev')
        p = ca.SX.sym('p', 1 + self.Nu + self.Nx + self.Nx + 1 + 20*2)

        # augmented state and control vectors
        states = ca.vertcat(x, y, phi)
        controls = ca.vertcat(v, delta_f)

        # dynamic model
        state_trans = [v * ca.cos(phi), v * ca.sin(phi), v * ca.tan(delta_f) / self.L]

        # function
        f = ca.Function('f', [states, controls], [ca.vcat(state_trans)], ['state', 'control_input'], ['state_trans'])

        # acados model
        x_dot = ca.SX.sym('x_dot', len(state_trans))
        f_impl = x_dot - f(states, controls)

        model.f_expl_expr = f(states, controls)
        model.f_impl_expr = f_impl
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.p = p
        model.name = 'vehicle_model'

        # constraint
        constraint.v_max = self.v_max
        constraint.v_min = self.v_min
        constraint.delta_f_max = self.delta_f_max
        constraint.delta_f_min = self.delta_f_min
        constraint.expr = ca.vcat([v, delta_f])

        self.model = model
        self.constraint = constraint


    def vehicle_model(self):
        model = AcadosModel()  # ca.types.SimpleNamespace()
        constraint = ca.types.SimpleNamespace()

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
        # state_trans = [v * ca.cos(phi), v * ca.sin(phi), delta_f]

        # function
        f = ca.Function('f', [states, controls], [ca.vcat(state_trans)], ['state', 'control_input'], ['state_trans'])
        # f_expl = ca.vcat(state_trans)
        # acados model
        x_dot = ca.SX.sym('x_dot', len(state_trans))
        f_impl = x_dot - f(states, controls)

        model.f_expl_expr = f(states, controls)
        model.f_impl_expr = f_impl
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.p = []
        model.name = 'vehicle_model'

        # constraint
        constraint.v_max = self.v_max
        constraint.v_min = self.v_min
        constraint.delta_f_max = self.delta_f_max
        constraint.delta_f_min = self.delta_f_min
        constraint.expr = ca.vcat([v, delta_f])

        self.model = model
        self.constraint = constraint

    def ocp_problem_1(self):
        model = AcadosModel()  # ca.types.SimpleNamespace()
        constraint = ca.types.SimpleNamespace()

        # original states
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        phi = ca.SX.sym('theta')

        # control inputs
        v = ca.SX.sym('v')
        delta_f = ca.SX.sym('delta_f')

        # previous control inputs
        v_prev = ca.SX.sym('v_prev')
        delta_f_prev = ca.SX.sym('delta_f_prev')

        p = ca.SX.sym('p', self.Nx+self.Nu)

        # augmented state and control vectors
        states = ca.vertcat(x, y, phi, v_prev, delta_f_prev)
        controls = ca.vertcat(v, delta_f)

        # dynamic model
        state_trans = [v * ca.cos(phi), v * ca.sin(phi), v * ca.tan(delta_f) / self.L, v, delta_f]

        # function
        f = ca.Function('f', [states, controls], [ca.vcat(state_trans)], ['state', 'control_input'], ['state_trans'])

        # acados model
        x_dot = ca.SX.sym('x_dot', len(state_trans))
        f_impl = x_dot - f(states, controls)

        model.f_expl_expr = f(states, controls)
        model.f_impl_expr = f_impl
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.p = p
        model.name = 'vehicle_model'

        # constraint
        constraint.v_max = self.v_max
        constraint.v_min = self.v_min
        constraint.delta_f_max = self.delta_f_max
        constraint.delta_f_min = self.delta_f_min
        constraint.expr = ca.vcat([v, delta_f])


        # Ensure current working directory is current folder
        current_file_parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        os.chdir(current_file_parent_dir)

        # 设置 acados 模型的目录，并在其上加上 '/MPC'
        self.acados_models_dir = './MPC/acados_models'
        # os.chdir(os.path.dirname(os.path.realpath(__file__)))
        # self.acados_models_dir = './acados_models'
        safe_mkdir_recursive(os.path.join(os.getcwd(), self.acados_models_dir))
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, acados_source_path)

        nx = model.x.size()[0]
        self.nx = nx
        nu = model.u.size()[0]
        self.nu = nu
        n_params = model.p.size()[0]

        # create OCP
        ocp = AcadosOcp()
        ocp.acados_include_path = acados_source_path + '/include'
        ocp.acados_lib_path = acados_source_path + '/lib'
        ocp.model = model
        ocp.dims.N = self.Np
        ocp.solver_options.tf = self.Np / 10
        ocp.dims.nh = self.Nu
        ocp.dims.np = n_params
        ocp.parameter_values = np.zeros(n_params)

        # 权重矩阵
        self.q = 1
        self.ru = 0.01
        self.rdu = 0.3
        self.S = 0.1  # Obstacle avoidance funcation coefficient
        self.Q1 = self.q * np.eye(self.Nx)  # ego_lane: lane_2
        self.Q2 = (1 - self.q) * np.eye(self.Nx)  # left_lane: lane_1
        self.Ru = self.ru * np.eye(self.Nu)
        self.Rdu = self.rdu * np.eye(self.Nu)

        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # initial state
        x_ref = np.zeros(nx)
        ocp.constraints.x0 = x_ref

        stage_cost = ca.mtimes([(states[:3] - p[:3]).T, self.Q1, (states[:3] - p[:3])]) \
                     + ca.mtimes([controls.T, self.Ru, controls]) \
                     + ca.mtimes([(controls - states[3:]).T, self.Rdu, (controls - states[3:])])
        terminal_cost = ca.mtimes([(states[:3] - p[:3]).T, self.Q1, (states[:3] - p[:3])])

        ocp.model.cost_expr_ext_cost = stage_cost
        ocp.model.cost_expr_ext_cost_e = terminal_cost

        # set constraints
        ocp.constraints.lbu = np.array([constraint.v_min, constraint.delta_f_min])
        ocp.constraints.ubu = np.array([constraint.v_max, constraint.delta_f_max])
        ocp.constraints.idxbu = np.array([0, 1])
        # define general constraint function
        # v_diff = v - v_prev  # difference between current and previous v
        # delta_f_diff = delta_f - delta_f_prev  # difference between current and previous delta_f
        # ocp.model.con_h_expr = ca.vertcat(v_diff, delta_f_diff)
        # # set lower and upper bounds for the constraints
        # ocp.constraints.lh = np.array([self.d_v_min, self.d_delta_f_min])
        # ocp.constraints.uh = np.array([self.d_v_max, self.d_delta_f_max])
        # ocp.constraints.lh = np.array([-100, -10])
        # ocp.constraints.uh = np.array([100, 10])

        # solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # FULL_CONDENSING_QPOASES
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP
        ocp.solver_options.hessian_approx = 'EXACT'
        ocp.solver_options.integrator_type = 'ERK' #'DISCRETE'
        # ocp.solver_options.print_level = 1
        ocp.solver_options.tol = 1e-6
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # SQP_RTI, SQP
        ocp.solver_options.globalization = 'FIXED_STEP'
        ocp.solver_options.alpha_min = 1e-2
        # ocp.solver_options.__initialize_t_slacks = 0
        # ocp.solver_options.regularize_method = 'CONVEXIFY'
        ocp.solver_options.levenberg_marquardt = 1e-1
        # ocp.solver_options.print_level = 2
        SQP_max_iter = 300
        ocp.solver_options.qp_solver_iter_max = 400
        ocp.solver_options.regularize_method = 'MIRROR'
        # ocp.solver_options.exact_hess_constr = 0
        # ocp.solver_options.line_search_use_sufficient_descent = line_search_use_sufficient_descent
        # ocp.solver_options.globalization_use_SOC = globalization_use_SOC
        ocp.solver_options.eps_sufficient_descent = 1e-1
        ocp.solver_options.qp_tol = 5e-7

        # compile acados ocp
        json_file = os.path.join('./' + model.name + '_acados_ocp.json')
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)
        self.integrator = AcadosSimSolver(ocp, json_file=json_file)

    def ocp_problem_2(self):
        model = AcadosModel()  # ca.types.SimpleNamespace()
        constraint = ca.types.SimpleNamespace()

        # original states
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        phi = ca.SX.sym('theta')

        # control inputs
        v = ca.SX.sym('v')
        delta_f = ca.SX.sym('delta_f')

        # # previous control inputs
        # v_prev = ca.SX.sym('v_prev')
        # delta_f_prev = ca.SX.sym('delta_f_prev')

        p = ca.SX.sym('p', self.Nu+self.Nx+self.Nx)

        # augmented state and control vectors
        states = ca.vertcat(x, y, phi)
        controls = ca.vertcat(v, delta_f)

        # dynamic model
        state_trans = [v * ca.cos(phi), v * ca.sin(phi), v * ca.tan(delta_f) / self.L]

        # function
        f = ca.Function('f', [states, controls], [ca.vcat(state_trans)], ['state', 'control_input'], ['state_trans'])

        # acados model
        x_dot = ca.SX.sym('x_dot', len(state_trans))
        f_impl = x_dot - f(states, controls)

        model.f_expl_expr = f(states, controls)
        model.f_impl_expr = f_impl
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.p = p
        model.name = 'vehicle_model'

        # constraint
        constraint.v_max = self.v_max
        constraint.v_min = self.v_min
        constraint.delta_f_max = self.delta_f_max
        constraint.delta_f_min = self.delta_f_min
        constraint.expr = ca.vcat([v, delta_f])


        # Ensure current working directory is current folder
        current_file_parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        os.chdir(current_file_parent_dir)

        # 设置 acados 模型的目录，并在其上加上 '/MPC'
        self.acados_models_dir = './MPC/acados_models'
        # os.chdir(os.path.dirname(os.path.realpath(__file__)))
        # self.acados_models_dir = './acados_models'
        safe_mkdir_recursive(os.path.join(os.getcwd(), self.acados_models_dir))
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, acados_source_path)

        nx = model.x.size()[0]
        self.nx = nx
        nu = model.u.size()[0]
        self.nu = nu
        n_params = model.p.size()[0]

        # create OCP
        ocp = AcadosOcp()
        ocp.acados_include_path = acados_source_path + '/include'
        ocp.acados_lib_path = acados_source_path + '/lib'
        ocp.model = model
        ocp.dims.N = self.Np
        ocp.solver_options.tf = self.Np / 10
        ocp.dims.nh = self.Nu
        ocp.dims.np = n_params
        ocp.parameter_values = np.zeros(n_params)

        # 权重矩阵
        self.q = 1
        self.ru = 0.5
        self.rdu = 0.02
        self.S = 0.1  # Obstacle avoidance funcation coefficient
        self.Q1 = self.q * np.eye(self.Nx)  # ego_lane: lane_2
        self.Q2 = (1 - self.q) * np.eye(self.Nx)  # left_lane: lane_1
        self.Ru = self.ru * np.eye(self.Nu)
        self.Rdu = self.rdu * np.eye(self.Nu)

        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # initial state
        x_ref = np.zeros(nx)
        ocp.constraints.x0 = x_ref

        stage_cost = ca.mtimes([(states[:3] - p[2:5]).T, self.Q1, (states[:3] - p[2:5])]) \
                     + ca.mtimes([(states[:3] - p[5:8]).T, self.Q2, (states[:3] - p[5:8])]) \
                     + ca.mtimes([controls.T, self.Ru, controls]) \
                     + ca.mtimes([(controls - p[0:2]).T, self.Rdu, (controls - p[0:2])])
        terminal_cost = ca.mtimes([(states[:3] - p[2:5]).T, self.Q1, (states[:3] - p[2:5])]) \
                     + ca.mtimes([(states[:3] - p[5:8]).T, self.Q2, (states[:3] - p[5:8])])

        ocp.model.cost_expr_ext_cost = stage_cost
        ocp.model.cost_expr_ext_cost_e = terminal_cost

        # set constraints
        ocp.constraints.lbu = np.array([constraint.v_min, constraint.delta_f_min])
        ocp.constraints.ubu = np.array([constraint.v_max, constraint.delta_f_max])
        ocp.constraints.idxbu = np.array([0, 1])
        # define general constraint function
        v_diff = v - p[0]  # difference between current and previous v
        delta_f_diff = delta_f - p[1]  # difference between current and previous delta_f
        ocp.model.con_h_expr = ca.vertcat(v_diff / self.T, delta_f_diff / self.T)
        # set lower and upper bounds for the constraints
        ocp.constraints.lh = np.array([self.d_v_min, self.d_delta_f_min])
        ocp.constraints.uh = np.array([self.d_v_max, self.d_delta_f_max])

        # solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # FULL_CONDENSING_QPOASES
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP
        ocp.solver_options.hessian_approx = 'EXACT'
        ocp.solver_options.integrator_type = 'ERK' #'DISCRETE'
        # ocp.solver_options.print_level = 1
        ocp.solver_options.tol = 1e-8
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # SQP_RTI, SQP
        ocp.solver_options.globalization = 'FIXED_STEP'
        ocp.solver_options.alpha_min = 1e-2
        # ocp.solver_options.__initialize_t_slacks = 0
        # ocp.solver_options.regularize_method = 'CONVEXIFY'
        ocp.solver_options.levenberg_marquardt = 1e-1
        # ocp.solver_options.print_level = 2
        SQP_max_iter = 300
        ocp.solver_options.qp_solver_iter_max = 1000
        ocp.solver_options.regularize_method = 'MIRROR'
        # ocp.solver_options.exact_hess_constr = 0
        # ocp.solver_options.line_search_use_sufficient_descent = line_search_use_sufficient_descent
        # ocp.solver_options.globalization_use_SOC = globalization_use_SOC
        ocp.solver_options.eps_sufficient_descent = 1e-1
        ocp.solver_options.qp_tol = 5e-7
        # compile acados ocp
        json_file = os.path.join('./' + model.name + '_acados_ocp.json')
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)
        self.integrator = AcadosSimSolver(ocp, json_file=json_file)

    def ocp_problem_3(self):
        model = self.model
        constraint = self.constraint

        nx = model.x.size()[0]
        self.nx = nx
        nu = model.u.size()[0]
        self.nu = nu
        n_params = model.p.size()[0]

        # Ensure current working directory is current folder
        current_file_parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        os.chdir(current_file_parent_dir)

        # 设置 acados 模型的目录，并在其上加上 '/MPC'
        self.acados_models_dir = './MPC/acados_models'
        # os.chdir(os.path.dirname(os.path.realpath(__file__)))
        # self.acados_models_dir = './acados_models'
        safe_mkdir_recursive(os.path.join(os.getcwd(), self.acados_models_dir))
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, acados_source_path)

        # create OCP
        ocp = AcadosOcp()
        ocp.acados_include_path = acados_source_path + '/include'
        ocp.acados_lib_path = acados_source_path + '/lib'
        ocp.model = model
        ocp.dims.N = self.Np
        ocp.solver_options.tf = self.Np / 10
        ocp.dims.nh = self.Nu
        ocp.dims.np = n_params
        ocp.parameter_values = np.zeros(n_params)

        # 权重矩阵
        self.q = 1
        self.ru = 0.6
        self.rdu = 0.3
        self.S = 10000.0  # Obstacle avoidance funcation coefficient
        self.Q1 = self.q * np.eye(self.Nx)  # ego_lane: lane_2
        self.Q2 = (1 - self.q) * np.eye(self.Nx)  # left_lane: lane_1
        self.Ru = self.ru * np.eye(self.Nu)
        self.Rdu = self.rdu * np.eye(self.Nu)

        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # initial state
        x_ref = np.zeros(nx)
        ocp.constraints.x0 = x_ref

        # Obstacle avoidance cost function
        obj = 0.0
        for j in range(int((n_params-10)/2)):  ##20
            i = 10 + j * 2

            Obj_cost = self.S / (((model.x[0] - model.p[i]) ** 2) + ((model.x[1] - model.p[i+1]) ** 2))

            # Obj_cost = self.S / ca.mtimes(
            #     [(model.x[0:2] - model.p[i:i + 2]).T, (model.x[0:2] - model.p[i:i + 2])]
            obj = obj + Obj_cost

        stage_cost = ca.mtimes([(model.x[:3] - model.p[3:6]).T, self.Q1, (model.x[:3] - model.p[3:6])]) \
                    + ca.mtimes([(model.x[:3] - model.p[6:9]).T, self.Q2, (model.x[:3] - model.p[6:9])]) \
                    + ca.mtimes([model.u.T, self.Ru, model.u]) \
                     + ca.mtimes([(model.u - model.p[1:3]).T, self.Rdu, (model.u - model.p[1:3])]) \
                     + obj

        terminal_cost = ca.mtimes([(model.x[:3] - model.p[3:6]).T, self.Q1, (model.x[:3] - model.p[3:6])]) \
                        + ca.mtimes([(model.x[:3] - model.p[6:9]).T, self.Q2, (model.x[:3] - model.p[6:9])]) \


        ocp.model.cost_expr_ext_cost = stage_cost
        ocp.model.cost_expr_ext_cost_e = terminal_cost

        # set constraints
        ocp.constraints.lbu = np.array([constraint.v_min, constraint.delta_f_min])
        ocp.constraints.ubu = np.array([constraint.v_max, constraint.delta_f_max])
        ocp.constraints.idxbu = np.array([0, 1])
        # define general constraint function
        # v_diff = model.u[0] - model.p[1]  # difference between current and previous v
        # delta_f_diff = model.u[1] - model.p[2]  # difference between current and previous delta_f
        # ocp.model.con_h_expr = ca.vertcat(v_diff/self.T, delta_f_diff/self.T)
        # # set lower and upper bounds for the constraints
        # ocp.constraints.lh = np.array([self.d_v_min, self.d_delta_f_min])
        # ocp.constraints.uh = np.array([self.d_v_max, self.d_delta_f_max])

        # solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # FULL_CONDENSING_QPOASES
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP
        ocp.solver_options.hessian_approx = 'EXACT'
        ocp.solver_options.integrator_type = 'ERK'  # 'DISCRETE'
        # ocp.solver_options.print_level = 1
        ocp.solver_options.tol = 1e-6
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # SQP_RTI, SQP
        ocp.solver_options.globalization = 'FIXED_STEP'
        ocp.solver_options.alpha_min = 1e-2
        # ocp.solver_options.__initialize_t_slacks = 0
        # ocp.solver_options.regularize_method = 'CONVEXIFY'
        ocp.solver_options.levenberg_marquardt = 1e-1
        # ocp.solver_options.print_level = 2
        SQP_max_iter = 300
        ocp.solver_options.qp_solver_iter_max = 400
        ocp.solver_options.regularize_method = 'MIRROR'
        # ocp.solver_options.exact_hess_constr = 0
        # ocp.solver_options.line_search_use_sufficient_descent = line_search_use_sufficient_descent
        # ocp.solver_options.globalization_use_SOC = globalization_use_SOC
        ocp.solver_options.eps_sufficient_descent = 1e-1
        ocp.solver_options.qp_tol = 5e-7
        # compile acados ocp
        json_file = os.path.join('./' + model.name + '_acados_ocp.json')
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)
        self.integrator = AcadosSimSolver(ocp, json_file=json_file)

    def ocp_problem(self):
        # model = self.vehicle_model().model
        # m_constraint = self.vehicle_model().constraint
        model = self.model
        m_constraint = self.constraint

        # Ensure current working directory is current folder
        current_file_parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        os.chdir(current_file_parent_dir)

        # 设置 acados 模型的目录，并在其上加上 '/MPC'
        self.acados_models_dir = './MPC/acados_models'
        # os.chdir(os.path.dirname(os.path.realpath(__file__)))
        # self.acados_models_dir = './acados_models'
        safe_mkdir_recursive(os.path.join(os.getcwd(), self.acados_models_dir))
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, acados_source_path)

        nx = model.x.size()[0]
        self.nx = nx
        nu = model.u.size()[0]
        self.nu = nu
        ny = nx + nu
        n_params = len(model.p)

        # create OCP
        ocp = AcadosOcp()
        ocp.acados_include_path = acados_source_path + '/include'
        ocp.acados_lib_path = acados_source_path + '/lib'
        ocp.model = model
        ocp.dims.N = self.Np
        ocp.solver_options.tf = self.Np / 10

        # initialize parameters
        ocp.dims.np = n_params
        ocp.parameter_values = np.zeros(n_params)

        # 权重矩阵
        self.q = 1
        self.ru = 0.01
        self.rdu = 0.3
        self.S = 0.1  # Obstacle avoidance function coefficient
        self.Q1 = self.q * np.eye(self.Nx)  # ego_lane: lane_2
        self.Q2 = (1 - self.q) * np.eye(self.Nx)  # left_lane: lane_1
        self.Ru = self.ru * np.eye(self.Nu)
        self.Rdu = self.rdu * np.eye(self.Nu)

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(self.Q1, self.Ru)
        ocp.cost.W_e = self.Q1
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        # set constraints
        ocp.constraints.lbu = np.array([m_constraint.v_min, m_constraint.delta_f_min])
        ocp.constraints.ubu = np.array([m_constraint.v_max, m_constraint.delta_f_max])
        ocp.constraints.idxbu = np.array([0, 1])
        # ocp.constraints.lbx = np.array([m_constraint.x_min, m_constraint.y_min])
        # ocp.constraints.ubx = np.array([m_constraint.x_max, m_constraint.y_max])
        # ocp.constraints.idxbx = np.array([0, 1])

        x_ref = np.zeros(nx)
        u_ref = np.zeros(nu)
        # initial state
        ocp.constraints.x0 = x_ref
        ocp.cost.yref = np.concatenate((x_ref, u_ref))
        ocp.cost.yref_e = x_ref

        # solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'

        # explicit Runge-Kutta integrator
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # 'SQP'

        ocp.solver_options.sim_method_num_steps = 2
        ocp.solver_options.qp_solver_cond_N = self.Np
        ocp.solver_options.qp_solver_iter_max = 1000

        # compile acados ocp
        json_file = os.path.join('./' + model.name + '_acados_ocp.json')
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)
        self.integrator = AcadosSimSolver(ocp, json_file=json_file)

    def solve_obj(self,obj_info):
        self.obj_Mux = []
        vehicle_num = 0

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
                        vehicle_num += 1
                if obj_info['Ego_following'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Ego_following'][0].id:
                        self.obj_pos[i] = 2  # 'Ego_following'
                        vehicle_num += 1
                if obj_info['Left_preceding'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Left_preceding'][0].id:
                        self.obj_pos[i] = 3  # 'Ego_preceding'
                        vehicle_num += 1
                if obj_info['Left_following'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Left_following'][0].id:
                        self.obj_pos[i] = 4  # 'Left_following'
                        vehicle_num += 1
                if obj_info['Right_preceding'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Right_preceding'][0].id:
                        self.obj_pos[i] = 5  # 'Right_preceding'
                        vehicle_num += 1
                if obj_info['Right_following'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Right_following'][0].id:
                        self.obj_pos[i] = 6  # 'Right_following'
                        vehicle_num += 1
                if obj_info['Left'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Left'][0].id:
                        self.obj_pos[i] = 7  # 'Left'
                        vehicle_num += 1
                if obj_info['Right'][0] != None:
                    if self.obj_actor_id[i] == obj_info['Right'][0].id:
                        self.obj_pos[i] = 8  # 'Right'
                        vehicle_num += 1
            self.obj_Mux.append(np.concatenate(
                (self.obj_x_ref.T, self.obj_y_ref.T, self.obj_phi_ref.T, self.obj_actor_id.T, self.obj_pos.T)))
        vehicle_num = int(vehicle_num / self.Np)

    def calc_input_1(self, x_current, obj_info, ref, ref_left, u_last, csp, fpath, q, ru, rdu):
        self.solve_obj(obj_info)
        obj_Mux = self.obj_Mux
        for i in range(self.Np):
            objs_xy = np.ones(1 + 2 * 20) * 100000
            objs_xy[0] = np.size(obj_info['Obj_actor'])
            for j in range(int(objs_xy[0])):
                if obj_Mux[j][4, 0] != 0:  # 0：'not vehicle_around'
                    objs_xy[j * 2 + 1] = obj_Mux[j][0, i]
                    objs_xy[j * 2 + 2] = obj_Mux[j][1, i]
            self.solver.set(i, 'p', np.concatenate(([q],u_last,ref[0:3], ref_left[0:3],objs_xy)))
        self.solver.set(self.Np, 'p', np.concatenate(([q],u_last,ref[0:3], ref_left[0:3], objs_xy)))

        start = timeit.default_timer()

        ##  set inertial (stage 0)
        self.solver.set(0, 'lbx', x_current)
        self.solver.set(0, 'ubx', x_current)
        status = self.solver.solve()
        if status != 0:
            raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))

        simU = np.zeros((self.Np - 1, self.Nu))
        simX = np.zeros((self.Np, self.Nx))
        simX[self.Np - 1, :] = self.solver.get(self.Np, 'x')
        for i in range(self.Np - 1):
            simX[i, :] = self.solver.get(i, 'x')
            simU[i, :] = self.solver.get(i, 'u')

        time_record = timeit.default_timer() - start
        # simulate system
        self.integrator.set('x', x_current)
        self.integrator.set('u', simU[0])

        status_s = self.integrator.solve()
        if status_s != 0:
            raise Exception('acados integrator returned status {}. Exiting.'.format(status))

        print("v:", simU[0][0], " delta_f:", simU[0][1])

        print("average estimation time is {}".format(time_record))

        # Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=x0, target_state=xs, robot_states=simX, )

        MPC_unsolved = False

        return simU[0], MPC_unsolved, simX


    def calc_input(self, x_current, obj_info, ref, ref_left, u_last, csp, fpath, q, ru, rdu):
        # x_current = x_current.reshape(1, -1)
        xs = np.concatenate((u_last, ref[0:3], ref_left[0:3]))
        for i in range(self.Np+1):
            # self.solver.set(i, 'p', ref[0:3])
            self.solver.set(i, 'p', xs)

        # for i in range(self.Np):
        #     self.solver.set(i, 'p', np.array([fpath.x[i],fpath.y[i],fpath.yaw[i],u_last[0],u_last[1]]))
        #     # print(np.array([fpath.x[i],fpath.y[i],fpath.yaw[i],u_last[0],u_last[1]]))
        # self.solver.set(self.Np, 'p', xs)

        start = timeit.default_timer()

        ##  set inertial (stage 0)
        self.solver.set(0, 'lbx', x_current)
        self.solver.set(0, 'ubx', x_current)
        status = self.solver.solve()
        if status != 0:
            raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))

        simU = np.zeros((self.Np - 1, self.Nu))
        simX = np.zeros((self.Np, self.Nx))
        simX[self.Np - 1, :] = self.solver.get(self.Np, 'x')
        for i in range(self.Np - 1):
            simX[i, :] = self.solver.get(i, 'x')
            simU[i, :] = self.solver.get(i, 'u')

        time_record = timeit.default_timer() - start
        # simulate system
        self.integrator.set('x', x_current)
        self.integrator.set('u', simU[0])

        status_s = self.integrator.solve()
        if status_s != 0:
            raise Exception('acados integrator returned status {}. Exiting.'.format(status))

        print("v:", simU[0][0], " delta_f:", simU[0][1], " d_u:", (simU[28][1]-simU[27][1])/self.T)

        # print("average estimation time is {}".format(time_record))
        # print("max estimation time is {}".format(time_record.max()))
        # print("min estimation time is {}".format(time_record.min()))
        # Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=x0, target_state=xs, robot_states=simX, )

        MPC_unsolved = False

        return simU[0], MPC_unsolved, simX
