import numpy as np
from matplotlib import pyplot as plt
from cvxopt import matrix, solvers
from MPC.qpoases import PyOptions as Options
from MPC.qpoases import PyQProblem as QProblem
from MPC.qpoases import PyPrintLevel as PrintLevel

'''
横纵向耦合运动模型
https://blog.csdn.net/m0_54639819/article/details/120275819?spm=1001.2014.3001.5502
'''


class MPC_controller_kinematics:

    def __init__(self, param):
        self.Matrix_Ap = None
        self.Matrix_Bp = None
        self.Matrix_ub = None
        self.param = param
        self.T = self.param.T
        self.N = self.param.N
        self.L = self.param.L
        self.Nx = self.param.mpc_Nx
        self.Nu = self.param.mpc_Nu
        self.Ny = self.param.mpc_Ny
        self.Nw = self.param.mpc_Nw
        self.Np = self.param.mpc_Np
        self.Nc = self.param.mpc_Nc
        self.eps = self.param.eps
        self.e_min = 0
        self.e_max = 10  # 松弛因子
        self.rou = 10

        self.q = 100
        self.ru = 80
        self.Matrix_Q = self.q * np.eye(self.Nx * self.Np)
        self.Matrix_Ru = self.ru * np.eye(self.Nu * self.Nc)

        self.Matrix_Cy = np.zeros([self.Ny, self.Nx + self.Nu])
        self.Matrix_Cy[0, 0] = 1
        self.Matrix_Cy[1, 1] = 1
        self.Matrix_Cy[2, 2] = 1


        self.U_min = [[0], [-0.194]]
        self.U_max = [[50/3.6], [0.194]]
        self.ext_U_min = [[-5],[-0.08]]
        self.ext_U_max = [[3],[0.08]]
        self.delta_ext_U_min = [[-2],[-0.005]]
        self.delta_ext_U_max = [[1], [0.005]]
        self.Matrix_U_max = np.zeros([self.Nc * self.Nu, 1])
        self.Matrix_U_min = np.zeros([self.Nc * self.Nu, 1])
        self.Matrix_delta_U_max = np.zeros([self.Nc * self.Nu, 1])
        self.Matrix_delta_U_min = np.zeros([self.Nc * self.Nu, 1])
        for i in range(self.Nc):
            self.Matrix_U_max[i * self.Nu:(i + 1) * self.Nu] = self.ext_U_max
            self.Matrix_U_min[i * self.Nu:(i + 1) * self.Nu] = self.ext_U_min
            self.Matrix_delta_U_max[i * self.Nu:(i + 1) * self.Nu] = self.delta_ext_U_max
            self.Matrix_delta_U_min[i * self.Nu:(i + 1) * self.Nu] = self.delta_ext_U_min

        self.Matrix_AI = np.zeros([self.Nc * self.Nu, self.Nc * self.Nu])
        self.Matrix_At = np.eye(self.Nu)
        for i in range(self.Nc):
            for j in range(self.Nc):
                if i >= j:
                    self.Matrix_AI[i * self.Nu: (i + 1) * self.Nu, j * self.Nu: (j + 1) * self.Nu] = self.Matrix_At

        self.x = np.zeros([3, 1])
        self.U = np.zeros([2, 1])
        self.ref = np.zeros([6, 1])
        self.output = np.zeros([2, 1])  # delta_(v-vr)/(delta-delta_r)
        self.Input = np.zeros([2, 1])  # v/delta_f

        self.Matrix_dU = np.zeros([self.Nc * self.Nu, 1])
        self.Matrix_state_new = np.zeros([self.Nx + self.Nu, 1])
        self.Matrix_Ak = np.zeros([self.Nx, self.Nx])
        self.Matrix_Bk = np.zeros([self.Nx, self.Nu])
        self.Matrix_a = np.zeros([self.Nx, self.Nx])
        self.Matrix_b = np.zeros([self.Nx, self.Nu])
        self.Matrix_yita = np.zeros([self.Ny * self.Np, self.Nx + self.Nu])
        self.Matrix_theta = np.zeros([self.Ny * self.Np, self.Nu * self.Nc])

    def calc_input(self, x, ref):

        self.ref = ref
        self.x = x

        for i in range(self.Nc):
            self.Matrix_dU[i * self.Nu:(i + 1) * self.Nu] = self.U

        self.Matrix_state_new[0] = self.x[0] - self.ref[0]
        self.Matrix_state_new[1] = self.x[1] - self.ref[1]  # Y
        self.Matrix_state_new[2] = self.x[2] - self.ref[2]  # PHI
        self.Matrix_state_new[3] = self.U[0]  # da??
        self.Matrix_state_new[4] = self.U[1]

        self.Matrix_Ak[0, 2] = -self.ref[3, 0] * np.sin(self.ref[2, 0])
        self.Matrix_Ak[1, 2] = self.ref[3, 0] * np.cos(self.ref[2, 0])

        self.Matrix_Bk[0, 0] = np.cos(self.ref[2, 0])
        self.Matrix_Bk[1, 0] = np.sin(self.ref[2, 0])
        self.Matrix_Bk[2, 0] = np.tan(self.ref[4, 0]) / self.L
        self.Matrix_Bk[2, 1] = self.ref[3, 0] / (self.L * np.cos(self.ref[4, 0] ** 2))

        self.Matrix_a = np.eye(self.Nx) + self.T * self.Matrix_Ak
        self.Matrix_b = self.T * self.Matrix_Bk

        self.Matrix_Ap = np.empty((2, 2), dtype=object)
        self.Matrix_Ap[0, 0] = self.Matrix_a
        self.Matrix_Ap[0, 1] = self.Matrix_b
        self.Matrix_Ap[1, 0] = np.zeros([self.Nu, self.Nx])
        self.Matrix_Ap[1, 1] = np.eye(self.Nu)
        self.Matrix_Ap = np.vstack([np.hstack(Mat_size) for Mat_size in self.Matrix_Ap])

        self.Matrix_Bp = np.empty((2, 1), dtype=object)
        self.Matrix_Bp[0, 0] = self.Matrix_b
        self.Matrix_Bp[1, 0] = np.eye(self.Nu)
        self.Matrix_Bp = np.vstack([np.hstack(Mat_size) for Mat_size in self.Matrix_Bp])

        for i in range(self.Np):
            if i == 0:
                self.Matrix_yita[i * self.Ny: (i + 1) * self.Ny, :] = self.Matrix_Cy @ self.Matrix_Ap
            else:
                self.Matrix_yita[i * self.Ny: (i + 1) * self.Ny, :] = self.Matrix_yita[(i - 1) * self.Ny: i * self.Ny,
                                                                      :] @ self.Matrix_Ap

        for i in range(self.Np):
            if i < self.Nc:  # 对角元
                self.Matrix_theta[i * self.Ny: (i + 1) * self.Ny, i * self.Nu: (i + 1) * self.Nu] = \
                    self.Matrix_Cy @ self.Matrix_Bp
                for j in range(self.Nc):
                    if i > j:
                        self.Matrix_theta[i * self.Ny: (i + 1) * self.Ny, j * self.Nu: (j + 1) * self.Nu] = \
                            self.Matrix_yita[(i - 1 - j) * self.Ny: (i - j) * self.Ny, :] @ self.Matrix_Bp
            else:
                for j in range(self.Nc):
                    self.Matrix_theta[i * self.Ny: (i + 1) * self.Ny, j * self.Nu: (j + 1) * self.Nu] = \
                        self.Matrix_yita[(i - 1 - j) * self.Ny: (i - j) * self.Ny, :] @ self.Matrix_Bp

        self.Matrix_Error = np.zeros([self.Ny * self.Np, 1])
        self.Matrix_Error = self.Matrix_yita @ self.Matrix_state_new

        self.Matrix_H_QP_du = np.empty((2, 2), dtype=object)
        self.Matrix_H_QP_du[0, 0] = np.transpose(self.Matrix_theta) @ self.Matrix_Q @ self.Matrix_theta + self.Matrix_Ru
        self.Matrix_H_QP_du[0, 1] = np.zeros([self.Nc * self.Nu, 1])
        self.Matrix_H_QP_du[1, 0] = np.zeros([1, self.Nc * self.Nu])
        self.Matrix_H_QP_du[1, 1] = self.rou * np.eye(1)
        self.Matrix_H_QP_du = np.vstack([np.hstack(Mat_size) for Mat_size in self.Matrix_H_QP_du])
        self.Matrix_H_QP_du = 2 * self.Matrix_H_QP_du

        self.Matrix_f_QP_du = np.empty((1, 2), dtype=object)
        self.Matrix_f_QP_du[0, 0] = 2 * np.transpose(self.Matrix_Error) @ self.Matrix_Q @ self.Matrix_theta
        self.Matrix_f_QP_du[0, 1] = np.zeros([1, 1])
        self.Matrix_f_QP_du = np.vstack([np.hstack(Mat_size) for Mat_size in self.Matrix_f_QP_du])
        self.Matrix_f_QP_du = np.transpose(self.Matrix_f_QP_du)

        self.Matrix_A_cons = np.empty((2, 2), dtype=object)
        self.Matrix_A_cons[0, 0] = self.Matrix_AI
        self.Matrix_A_cons[0, 1] = np.zeros([self.Nc * self.Nu, 1])
        self.Matrix_A_cons[1, 0] = -self.Matrix_AI
        self.Matrix_A_cons[1, 1] = np.zeros([self.Nc * self.Nu, 1])
        self.Matrix_A_cons = np.vstack([np.hstack(Mat_size) for Mat_size in self.Matrix_A_cons])

        self.Matrix_B_cons = np.empty((2, 1), dtype=object)
        self.Matrix_B_cons[0, 0] = self.Matrix_U_max - self.Matrix_dU
        self.Matrix_B_cons[1, 0] = -self.Matrix_U_min + self.Matrix_dU
        self.Matrix_B_cons = np.vstack([np.hstack(Mat_size) for Mat_size in self.Matrix_B_cons])

        self.Matrix_lb = np.empty((2, 1), dtype=object)
        self.Matrix_lb[0, 0] = self.Matrix_delta_U_min
        self.Matrix_lb[1, 0] = self.e_min
        self.Matrix_lb = np.vstack([np.hstack(Mat_size) for Mat_size in self.Matrix_lb])

        self.Matrix_ub = np.empty((2, 1), dtype=object)
        self.Matrix_ub[0, 0] = self.Matrix_delta_U_max
        self.Matrix_ub[1, 0] = self.e_max
        self.Matrix_ub = np.vstack([np.hstack(Mat_size) for Mat_size in self.Matrix_ub])

        # # cvxopt求解过程
        # P = matrix(self.Matrix_H_QP_du)
        # q = matrix(self.Matrix_f_QP_du.astype(np.double))
        # G = matrix(np.vstack((self.Matrix_A_cons, np.eye(self.Nc * self.Nu + 1), -np.eye(self.Nc * self.Nu + 1))))
        # h = matrix(np.vstack((self.Matrix_B_cons,self.Matrix_ub, -self.Matrix_lb)))
        #
        # result = solvers.qp(P, q, G, h)  # 1/2x'Px+q'x   Gx<=h  Ax=b 注意使用qp时，每个参数要换成matrix
        # X = result['x']  # 'x'为result中的解，'status'表示是否找到最优解。
        # self.output[0, 0] = X[0]  # U_bar
        # self.output[1, 0] = X[1]
        #
        # self.U[0] = self.U[0] + self.output[0, 0]
        # self.U[1] = self.U[1] + self.output[1, 0]
        #
        # self.Input[0] = self.U[0] + self.ref[3]
        # self.Input[1] = self.U[1] + self.ref[4]
        #
        # return self.Input

        qp = QProblem(61, 120)
        options = Options()
        options.printLevel = PrintLevel.NONE
        qp.setOptions(options)

        H = self.Matrix_H_QP_du
        g = self.Matrix_f_QP_du.astype(np.double).T[0, :]
        A = self.Matrix_A_cons
        lb = self.Matrix_lb.T[0, :]
        ub = self.Matrix_ub.T[0, :]
        lbA = -1e8 * np.ones(120)
        ubA = self.Matrix_B_cons.T[0, :]

        # Solve first QP.
        nWSR = np.array([200])
        qp.init(H, g, A, lb, ub, lbA,
                ubA, nWSR)

        result = np.zeros(60)
        qp.getPrimalSolution(result)
        X = result.reshape(60, 1)
        self.output[0, 0] = X[0][0]  # U_bar
        self.output[1, 0] = X[1][0]

        self.U[0] = self.U[0] + self.output[0, 0]
        self.U[1] = self.U[1] + self.output[1, 0]

        self.Input[0] = self.U[0] + self.ref[3]
        self.Input[1] = self.U[1] + self.ref[4]

        return self.Input

    def path(self, time_counts):
        self.t = self.T * time_counts
        self.ref[0, 0] = 406 + self.t * 5  # 参考的X
        self.ref[1, 0] = -100  # 参考的Y
        self.ref[2, 0] = 0  # 参考的横摆角，0.2怎么算出来的？ 答案可以根据书中p106 公式推出,根据车辆运动学的公式带入速度，前轮转角和轴距即可算出
        self.ref[3, 0] = 5  # 速度5m/s，这是参考速度
        self.ref[4, 0] = 0  # 前轮偏角=tan(轴距/半径）=tan(2.6/25)=0.104，这是参考前轮转角
        print(self.ref)
        plt.scatter(self.ref[0, 0], self.ref[1, 0], marker='o', facecolor='none',
                    edgecolors='r')

        plt.axis([-100, 100, -100, 100])
        plt.pause(0.05)
        return self.ref
