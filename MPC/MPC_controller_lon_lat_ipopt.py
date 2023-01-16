from __future__ import division
import numpy as np
from cvxopt import matrix, solvers
from MPC.qpoases import PyOptions as Options
from MPC.qpoases import PyQProblem as QProblem
from MPC.qpoases import PyPrintLevel as PrintLevel
from cvxopt import matrix

from pyomo.environ import *
import pyomo.environ as pyo
path = '/home/wangliwen/Package/ipopt-linux64/ipopt'

class MPC_controller_lon_lat_ipopt:

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
        self.lanewidth = self.param.lanewidth
        self.dstop = self.param.dstop
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
        self.Y_left = np.zeros([self.Np,1])

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
        self.q = 1
        self.ru = 0.1
        self.rdu = 0.1
        self.rou = 0.005  # rho的值
        self.Q = self.q * np.eye(self.Nx)
        self.Ru = self.ru * np.eye(self.Nu)
        self.Rdu = self.rdu * np.eye(self.Nu)

        self.Cy_ext = np.zeros([self.Np * self.Ny, self.Np * self.Nx])
        self.Q_cell = np.zeros([self.Np * self.Ny, self.Np * self.Ny])
        self.Ru_cell = np.zeros([self.Nc * self.Nu, self.Nc * self.Nu])
        self.Rdu_cell = np.zeros([self.Nc * self.Nu, self.Nc * self.Nu])

        for i in range(self.Np):
            self.Cy_ext[i * self.Ny:(i + 1) * self.Ny, i * self.Ny: (i + 1) * self.Ny] = self.Cy
        for i in range(self.Np - 2):
            self.Q_cell[i * self.Ny:(i + 1) * self.Ny, i * self.Ny: (i + 1) * self.Ny] = self.Q
        for i in range(self.Np - 2, self.Np):
            self.Q_cell[i * self.Ny:(i + 1) * self.Ny, i * self.Ny: (i + 1) * self.Ny] = self.Q
        for i in range(self.Nc - 1):
            self.Ru_cell[i * self.Nu: (i + 1) * self.Nu, i * self.Nu: (i + 1) * self.Nu] = self.Ru
        for i in range(self.Nc - 1 + 1, self.Nc):
            self.Ru_cell[i * self.Nu: (i + 1) * self.Nu, i * self.Nu: (i + 1) * self.Nu] = self.Ru
        for i in range(self.Nc):
            self.Rdu_cell[i * self.Nu: (i + 1) * self.Nu, i * self.Nu: (i + 1) * self.Nu] = self.Rdu

        # y_ref
        for i in range(self.Np):
            # 自车状态预测
            self.x_predict[i] = x_current[0] + u_last[0] * np.cos(x_current[2]) * self.T * i
            self.y_predict[i] = x_current[1] + u_last[0] * np.sin(x_current[2]) * self.T * i
            self.phi_predict[i] = x_current[2] + u_last[1] * self.T * i
            # 参考路径信息
            self.x_ref[i] = ref[0][i]
            self.y_ref[i] = ref[1][i]
            self.phi_ref[i] = ref[2][i]
            self.x_ref_left[i] = ref_left[0][i]
            self.y_ref_left[i] = ref_left[1][i]
            self.phi_ref_left[i] = ref_left[2][i]
        for i in range(self.Np):
            self.x_ext[i * self.Ny: (i + 1) * self.Ny, 0] = [self.x_predict[i], self.y_predict[i], self.phi_predict[i]]
            self.y_ref_ext[i * self.Ny: (i + 1) * self.Ny, 0] = [self.x_ref[i], self.y_ref[i], self.phi_ref[i]]
            self.y_ref_left_ext[i * self.Ny: (i + 1) * self.Ny, 0] = [self.x_ref_left[i], self.y_ref_left[i], self.phi_ref_left[i]]
        self.y_ext = self.Cy_ext * self.x_ext
        self.y_error = self.y_ext - self.y_ref_ext
        self.y_error_left = self.y_ext - self.y_ref_ext
        for i in range(self.Np):
            self.Y[i] = np.transpose(self.y_error[i * self.Ny: (i + 1) * self.Ny, 0])*self.Q*self.y_error[i * self.Ny: (i + 1) * self.Ny, 0]
            self.Y_left[i] = np.transpose(self.y_error_left[i * self.Ny: (i + 1) * self.Ny, 0]) * self.Q * self.y_error_left[i * self.Ny: (i + 1) * self.Ny, 0]


        # 约束
        for i in range(self.Np):
            # 前车状态预测
            self.obj_x_ref[i] = obj[0] + obj[3] * np.cos(obj[2]) * self.T * i
            self.obj_y_ref[i] = obj[1] + obj[3] * np.sin(obj[2]) * self.T * i
            self.obj_phi_ref[i] = obj[2] + obj[4] * self.T * i

        for i in range(self.Np):
            # 状态量约束
            self.pos_x_min[i] = -1000
            self.pos_y_min[i] = self.obj_y_ref[i] - self.lanewidth * 1.5
            self.pos_phi_min[i] = -1000
            if x_frenet_current[1] <= -1.75:
                self.pos_x_max[i] = 1000
            else:
                self.pos_x_max[i] = self.obj_x_ref[i] - self.dstop
                self.pos_y_max[i] = self.obj_y_ref[i] + self.lanewidth * 2.5
                self.pos_phi_max[i] = 1000

            # 输出量约束
            self.y_x_max[i] = 1000
            self.y_y_max[i] = 1000
            self.y_phi_max[i] = 1000
            self.y_x_min[i] = -1000
            self.y_y_min[i] = -1000
            self.y_phi_min[i] = -1000

        for i in range(self.Np):
            self.x_max_ext[i * self.Nx: (i + 1) * self.Nx, :] = [[self.pos_x_max[i]], [self.pos_y_max[i]], [self.pos_phi_max[i]]]
            self.x_min_ext[i * self.Nx: (i + 1) * self.Nx, :] = [[self.pos_x_min[i]], [self.pos_y_min[i]], [self.pos_phi_min[i]]]
            self.y_max_ext[i * self.Ny: (i + 1) * self.Ny, :] = [[self.y_x_max[i]], [self.y_y_max[i]], [self.y_phi_max[i]]]
            self.y_min_ext[i * self.Ny: (i + 1) * self.Ny, :] = [[self.y_x_min[i]], [self.y_y_min[i]], [self.y_phi_min[i]]]


        # 根据U矩阵得到deltaU矩阵，deltaU = Cdu * U + Du
        # Cdu矩阵
        Cdu1 = np.empty((2, 2), dtype=object)
        Cdu1[0, 0] = np.zeros([self.Nu, self.Nu * (self.Nc - 1)])
        Cdu1[0, 1] = np.zeros([self.Nu, self.Nu])
        Cdu1[1, 0] = -1 * np.eye(self.Nu * (self.Nc - 1))
        Cdu1[1, 1] = np.zeros([self.Nu * (self.Nc - 1), self.Nu])
        Cdu1 = np.vstack([np.hstack(Mat_size) for Mat_size in Cdu1])
        Cdu1 = Cdu1 + np.eye(self.Nu * self.Nc)

        # Du矩阵（列向量）
        Cdu2 = np.zeros([self.Nu * self.Nc, 1])
        for i in range(self.Nu):
            Cdu2[i, 0] = -1 * u_last[i]



        model = pyo.AbstractModel()

        model.m = pyo.Param(within=self.Np)
        model.n = pyo.Param(within=self.Nc)

        # def y_init(model, m):
        #     for i in range(m):
        #         return self.Y[i]

        model.y = Param(self.Np, 1, initialize=self.Y)


        model.I = pyo.RangeSet(1, self.Np)
        model.J = pyo.RangeSet(0, self.Nc-1)

        model.a = pyo.Param(model.I,model)
        model.b = pyo.Param(model.I)

        # def fb1(model, n,):
        #     maxU = pyo.RangeSet(0,self.Nc*self.Nu)
        #     for j in range(n):
        #         return (self.u_min_ext[j*self.Nu:j*self.Nu+1], self.u_max_ext[j*self.Nu:j*self.Nu+1])
        model.x1 = pyo.Var([model.J*self.Nu,1], within=pyo.Reals, bounds=(self.u_min_ext,self.u_max_ext)) # U
        model.x2 = Var(within=pyo.Reals, bounds=(self.e_min,self.e_max)) #yibuxilu

        def obj_expression(model):
            return pyo.summation(model.y)+pyo.summation(np.transpose(model.x1)*self.Ru*model.x1) + \
            pyo.summation(np.transpose(Cdu1*model.x1+Cdu2)*self.Ru*(Cdu1*model.x1+Cdu2)) + self.rou * model.x2

        model.OBJ = pyo.Objective(rule=obj_expression, sense=minimize)

        instance = model.create_instance()
        # use 'pprint' to print the model information
        model.pprint()
        SolverFactory('ipopt', executable=path).solve(instance).write()
        print('optimal f: {:.4f}'.format(model.OBJ()))
        print('optimal x: [{:.4f}, {:.4f}]'.format(model.x1(), model.x2()))