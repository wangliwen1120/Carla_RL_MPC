import numpy as np
from cvxopt import matrix, solvers
from MPC.qpoases import PyOptions as Options
from MPC.qpoases import PyQProblem as QProblem
from MPC.qpoases import PyPrintLevel as PrintLevel
from cvxopt import matrix


class MPC_controller_lon_lat:

    def __init__(self, param):

        self.rou = None
        self.ru = None
        self.rdu = None
        self.q = None
        self.u_last = None
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

        self.dstop = self.param.dstop
        self.v_x_ref = self.param.v_x_ref
        self.v_x_0 = self.param.v_x_0
        self.Pos_x_0 = self.param.Pos_x_0

        # 权重矩阵
        # self.q = 1 * np.diag([1 / (1 ** 2), 0 / (1 ** 2)])
        # self.ru = 10 * np.diag([1 / (1 ** 2)])  # 控制量的权重矩阵
        # self.rdu = 50 * np.diag([1 / self.delta_aS_max ** 2])
        # self.rou = 0 * 0.005 * 1  # rho的值

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
        self.e_max = 0.02

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
        self.x_max = np.zeros((self.Np, 1))
        self.x_min = np.zeros((self.Np, 1))
        self.x_ref = np.zeros((self.Np, 1))
        self.y_max = np.zeros((self.Np, 1))
        self.y_min = np.zeros((self.Np, 1))
        self.y_ref = np.zeros((self.Np, 1))
        self.phi_max = np.zeros((self.Np, 1))
        self.phi_min = np.zeros((self.Np, 1))
        self.phi_ref = np.zeros((self.Np, 1))
        self.y_ref_ext = np.zeros([self.Np * self.Ny, 1])

    def calc_input(self, x_current,ref, u_last, q, ru, rdu):
        self.u_last = u_last
        # ref_x = ref[0]
        # ref_y = ref[1]
        # ref_phi = ref[2]
        # ref_vx = ref_v * np.cos(ref_phi)
        # ref_vy = ref_v * np.sin(ref_phi)

        ref_v = np.zeros([self.Np, 1])
        ref_delta_f = np.zeros([self.Np, 1])
        ref_x = np.zeros([self.Np, 1])
        ref_y = np.zeros([self.Np, 1])
        ref_phi = np.zeros([self.Np, 1])
        ref_vx = np.zeros([self.Np, 1])
        ref_vy = np.zeros([self.Np, 1])
        for i in range(self.Np):
            if i == 0:
                ref_x[i] = x_current[0]
                ref_y[i] = x_current[1]
                ref_phi[i] = x_current[2]
            else:
                ref_x[i] = ref_x[i - 1] + u_last[0] * np.cos(ref_phi[i - 1]) * self.T
                ref_y[i] = ref_y[i - 1] + u_last[0] * np.sin(ref_phi[i - 1]) * self.T
                ref_phi[i] = ref_phi[i - 1] + u_last[1] * self.T
            ref_vx[i] = u_last[0] * np.cos(ref_phi[i])
            ref_vy[i] = u_last[0] * np.sin(ref_phi[i])
            ref_v[i] = u_last[0][0]
            ref_delta_f[i] = u_last[1][0]

            # for i in range(self.Np):
            #     if i == 0:
            #         ref_x[i] = ref[0]
            #         ref_y[i] = ref[1]
            #         ref_phi[i] = ref[2]
            #     else:
            #         ref_x[i] = ref_x[i - 1] + ref[3] * np.cos(ref_phi[i - 1]) * self.T
            #         ref_y[i] = ref_y[i - 1] + ref[3] * np.sin(ref_phi[i - 1]) * self.T
            #         ref_phi[i] = ref_phi[i - 1] + ref[4] * self.T
            #     ref_vx[i] = ref[3] * np.cos(ref_phi[i])
            #     ref_vy[i] = ref[4] * np.sin(ref_phi[i])
            #     ref_v[i] = ref[3]
            #     ref_delta_f[i] = ref[4]
            # for i in range(self.Np):
            #     self.x_max[i] = ref_x[i]
            #     self.y_max[i] = ref_y[i]
            #     self.phi_max[i] = ref_phi[i]
            #     self.x_min[i] = x_current[0]
            #     self.y_min[i] = x_current[1]
            #     self.phi_min[i] = x_current[2]
            #     self.x_ref[i] = self.x_max[i] - self.dstop
            #     self.y_ref[i] = self.y_max[i]
            #     self.phi_ref[i] = self.phi_max[i]

        # 权重矩阵
        self.q = 100
        self.ru = 80
        self.rdu = 80
        self.rou = 1  # rho的值
        self.Q = self.q * np.eye(self.Nx)
        self.Ru = self.ru * np.eye(self.Nu)
        self.Rdu = self.rdu * np.eye(self.Nu)

        for i in range(self.Np):
            self.x_max[i] = ref_x[i] + self.dstop
            self.y_max[i] = ref_y[i]
            self.phi_max[i] = ref_phi[i]
            self.x_min[i] = x_current[0]
            self.y_min[i] = x_current[1]
            self.phi_min[i] = x_current[2]
            self.x_ref[i] = ref_x[i]
            self.y_ref[i] = ref_y[i]
            self.phi_ref[i] = ref_phi[i]

        for i in range(self.Np):
            self.y_ref_ext[i * self.Ny: (i + 1) * self.Ny, :] = \
                np.array([[self.x_ref[i, 0]], [self.y_ref[i, 0]], [self.phi_ref[i, 0]]])

        for i in range(self.Np):
            self.x_max_ext[i * self.Nx: (i + 1) * self.Nx, :] = np.array(
                [[self.x_max[i, 0]], [self.y_max[i, 0]], [self.phi_max[i, 0]]])
            self.x_min_ext[i * self.Nx: (i + 1) * self.Nx, :] = np.array(
                [[self.x_min[i, 0]], [self.y_min[i, 0]], [self.phi_min[i, 0]]])
            self.y_max_ext[i * self.Ny: (i + 1) * self.Ny, :] = np.array(
                [[self.x_max[i, 0]], [self.y_max[i, 0]], [self.phi_max[i, 0]]])
            self.y_min_ext[i * self.Ny: (i + 1) * self.Ny, :] = np.array(
                [[self.x_min[i, 0]], [self.y_min[i, 0]], [self.phi_min[i, 0]]])

            if i < self.Nc:
                self.u_max_ext[i * self.Nu:(i + 1) * self.Nu] = np.array([[self.v_max], [self.delta_f_max]])
                self.u_min_ext[i * self.Nu:(i + 1) * self.Nu] = np.array([[self.v_min], [self.delta_f_min]])
                self.du_max_ext[i * self.Nu:(i + 1) * self.Nu] = np.array([[self.d_v_max], [self.d_delta_f_max]])
                self.du_min_ext[i * self.Nu:(i + 1) * self.Nu] = np.array([[self.d_v_min], [self.d_delta_f_min]])

        # model linearization

        A_cell = np.zeros([self.Nx, self.Nx, self.Np])  # 2 * 2 * Np的矩阵，第三个维度为每个时刻的对应的A矩阵
        B_cell = np.zeros([self.Nx, self.Nu, self.Np])  # 2 * 1 * Np的矩阵
        C_cell = np.zeros([self.Nx, 1,self.Np])  # 2 * Np的矩阵

        for i in range(self.Np):  # 保存每个预测时间步的Ak，Bk，Ck矩阵
            A_cell[:, :, i] = np.eye(self.Nx) + self.T*(np.array(
                [[0, 0, -ref_v[i] * np.sin(ref_phi[i])], [0, 0, ref_v[i] * np.cos(ref_phi[i])], [0, 0, 0]]))
            B_cell[:, :, i] = self.T * (np.array([[np.cos(ref_phi[i]), 0], [np.sin(ref_phi[i]), 0],
                           [np.tan(ref_delta_f[i]) / self.L, ref_v[i] / (self.L * (np.cos(ref_delta_f[i]) ** 2))]]))
            C_cell[:, :,i] = self.T * (np.array([ref_v[i] * np.cos(ref_phi[i]),ref_v[i] * np.sin(ref_phi[i]),
                            ref_v[i] * np.tan(ref_delta_f[i]) / self.L]) - A_cell[:, :, i] @
                            [ref_x[i], ref_y[i], ref_phi[i]] - B_cell[:, :, i] @ [ref_v[i], ref_delta_f[i]])

        # dynamicmatrix:
        A_ext = np.zeros([self.Nx * self.Np, self.Nx])  # 2Np * 2的分块列矩阵
        B_ext = np.zeros([self.Nx * self.Np, self.Nu * self.Nc])  # 2Np * Nc的分块矩阵
        for i in range(self.Np):  # 递推形式下的A_bar矩阵
            if i == 0:
                A_ext[i * self.Nx: (i + 1) * self.Nx, :] = A_cell[:, :, i]
            else:
                A_ext[i * self.Nx: (i + 1) * self.Nx, :] = A_cell[:, :, i] @ A_ext[(i - 1) * self.Nx: i * self.Nx, :]

        for i in range(self.Np):
            if i < self.Nc:  # 对角元
                B_ext[i * self.Nx: (i + 1) * self.Nx, i * self.Nu: (i + 1) * self.Nu] = B_cell[:, :,
                                                                                        i]  # 维数相同，每个小块是Nx*Nu维度
            else:  # Nc行之后的最后一列元素，形式改变了，先给了一个Bk(k > Nc)
                B_ext[i * self.Nx: (i + 1) * self.Nx, (self.Nc - 1) * self.Nu: self.Nc * self.Nu] = B_cell[:, :, i]

        for i in range(1, self.Np):  # 次对角元素，每一行都是上一行乘新的Ai，再加上上一步已经加好的每行最后一个元素，就得到B_bar
            B_ext[i * self.Nx: (i + 1) * self.Nx, :] = A_cell[:, :, i] @ B_ext[(i - 1) * self.Nx: i * self.Nx,
                                                                         :] + B_ext[i * self.Nx: (i + 1) * self.Nx, :]

        C_ext = np.zeros([self.Nx * self.Np, 1])  # 2Np * 1的矩阵
        for i in range(self.Np):
            if i == 0:
                C_ext[i * self.Nx: (i + 1) * self.Nx,:] = C_cell[:, :,i]
            else:  # C的上一行乘以新的Ak再加上新的Ck得到C_bar
                C_ext[i * self.Nx: (i + 1) * self.Nx,:] = A_cell[:, :, i] @ C_ext[(i - 1) * self.Nx: i * self.Nx,:] + C_cell[:,:, i]

        # 预测时域和控制时域内的分块权重矩阵
        Cy_ext = np.zeros([self.Np * self.Ny, self.Np * self.Nx])
        Q_cell = np.zeros([self.Np * self.Ny, self.Np * self.Ny])
        Ru_cell = np.zeros([self.Nc * self.Nu, self.Nc * self.Nu])
        Rdu_cell = np.zeros([self.Nc * self.Nu, self.Nc * self.Nu])
        for i in range(self.Np):
            Cy_ext[i * self.Ny:(i + 1) * self.Ny, i * self.Ny: (i + 1) * self.Ny] = self.Cy
        for i in range(self.Np - 2):
            Q_cell[i * self.Ny:(i + 1) * self.Ny, i * self.Ny: (i + 1) * self.Ny] = self.Q
        for i in range(self.Np - 2, self.Np):
            Q_cell[i * self.Ny:(i + 1) * self.Ny, i * self.Ny: (i + 1) * self.Ny] = self.Q
        for i in range(self.Nc - 1):
            Ru_cell[i * self.Nu: (i + 1) * self.Nu, i * self.Nu: (i + 1) * self.Nu] = self.Ru
        for i in range(self.Nc - 1 + 1, self.Nc):
            Ru_cell[i * self.Nu: (i + 1) * self.Nu, i * self.Nu: (i + 1) * self.Nu] = self.Ru
        for i in range(self.Nc):
            Rdu_cell[i * self.Nu: (i + 1) * self.Nu, i * self.Nu: (i + 1) * self.Nu] = self.Rdu

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
            Cdu2[i, 0] = -1 * self.u_last[i, 0]

        # 标准形式min (1/2x'Px+q'x)   s.t. Gx<=h
        H_QP_du_e = np.empty((2, 2), dtype=object)
        H_QP_du_e[0, 0] = np.transpose(Cy_ext @ B_ext @ np.linalg.inv(Cdu1)) @ Q_cell @ (
                Cy_ext @ B_ext @ np.linalg.inv(Cdu1)) + np.transpose(
            np.linalg.inv(Cdu1)) @ Ru_cell @ np.linalg.inv(Cdu1) + Rdu_cell

        H_QP_du_e[0, 1] = np.zeros([self.Nc * self.Nu, 1])
        H_QP_du_e[1, 0] = np.zeros([1, self.Nc * self.Nu])
        H_QP_du_e[1, 1] = self.rou * np.eye(1)
        H_QP_du_e = np.vstack([np.hstack(Mat_size) for Mat_size in H_QP_du_e])
        H_QP_du_e = 2 * H_QP_du_e

        # q为列向量
        f_QP_du_e = np.empty((1, 2), dtype=object)
        f_QP_du_e[0, 0] = 2 * np.transpose(Cy_ext @ (A_ext @ x_current - B_ext @ np.linalg.inv(
            Cdu1) @ Cdu2 + C_ext) - self.y_ref_ext) @ Q_cell @ Cy_ext @ B_ext @ np.linalg.inv(
            Cdu1) + 2 * np.transpose(np.linalg.inv(Cdu1) @ (-Cdu2)) @ Ru_cell @ np.linalg.inv(Cdu1)
        f_QP_du_e[0, 1] = np.zeros([1, 1])
        f_QP_du_e = np.vstack([np.hstack(Mat_size) for Mat_size in f_QP_du_e])
        f_QP_du_e = np.transpose(f_QP_du_e)

        lb = np.vstack((self.du_min_ext, self.e_min))
        ub = np.vstack((self.du_max_ext, self.e_max))

        A_du_eCons = np.empty([6, 2], dtype=object)
        ubA_du_eCons = np.empty([6, 1], dtype=object)
        A_du_eCons[0, 0] = np.linalg.inv(Cdu1)
        A_du_eCons[0, 1] = np.zeros([self.Nc * self.Nu, 1])
        A_du_eCons[1, 0] = B_ext @ np.linalg.inv(Cdu1)
        A_du_eCons[1, 1] = np.zeros([self.Nx * self.Np, 1])
        A_du_eCons[2, 0] = Cy_ext @ B_ext @ np.linalg.inv(Cdu1)
        A_du_eCons[2, 1] = np.zeros([self.Np * self.Ny, 1])
        A_du_eCons[3, 0] = -np.linalg.inv(Cdu1)
        A_du_eCons[3, 1] = np.zeros([self.Nc * self.Nu, 1])
        A_du_eCons[4, 0] = -B_ext @ np.linalg.inv(Cdu1)
        A_du_eCons[4, 1] = np.zeros([self.Nx * self.Np, 1])
        A_du_eCons[5, 0] = -Cy_ext @ B_ext @ np.linalg.inv(Cdu1)
        A_du_eCons[5, 1] = np.zeros([self.Np * self.Ny, 1])
        A_du_eCons = np.vstack([np.hstack(Mat_size) for Mat_size in A_du_eCons])

        ubA_du_eCons[0, 0] = self.u_max_ext + Cdu1 @ Cdu2
        ubA_du_eCons[1, 0] = self.x_max_ext - A_ext @ x_current - C_ext - B_ext @ np.linalg.inv(Cdu1) @ (-Cdu2)
        ubA_du_eCons[2, 0] = self.y_max_ext - Cy_ext @ (
                A_ext @ x_current + C_ext + B_ext @ np.linalg.inv(Cdu1) @ (-Cdu2))
        ubA_du_eCons[3, 0] = -self.u_min_ext - np.linalg.inv(Cdu1) @ Cdu2
        ubA_du_eCons[4, 0] = -self.x_min_ext + A_ext @ x_current + C_ext - B_ext @ np.linalg.inv(Cdu1) @ Cdu2
        ubA_du_eCons[5, 0] = -self.y_min_ext + Cy_ext @ (A_ext @ x_current + C_ext - B_ext @ np.linalg.inv(Cdu1) @ Cdu2)
        ubA_du_eCons = np.vstack([np.hstack(Mat_size) for Mat_size in ubA_du_eCons])

        # # cvxopt求解过程
        P = matrix(H_QP_du_e)
        q0 = f_QP_du_e.astype(np.double)
        q = matrix(q0)
        G = matrix(np.vstack((A_du_eCons, np.eye(self.Nc * self.Nu + 1), -np.eye(self.Nc * self.Nu + 1))))
        h = matrix(np.vstack((ubA_du_eCons, ub, -lb)))
        result = solvers.qp(P, q, G, h)  # 1/2x'Px+q'x   Gx<=h  Ax=b 注意使用qp时，每个参数要换成matrix
        # 重要：print可被关闭
        X = result['x']

        # # # qpoases求解过程
        # # Setting up QProblem object.
        # qp = QProblem(21, 400)
        # options = Options()
        # options.printLevel = PrintLevel.NONE
        # qp.setOptions(options)
        #
        # H = H_QP_du_e
        # g = f_QP_du_e.astype(np.double)[:, 0]
        # A = A_du_eCons
        # lb = lb[:, 0]
        # ub = ub[:, 0]
        # lbA = -1e8 * np.ones(400)
        # ubA = ubA_du_eCons[:, 0]
        #
        # # Solve first QP.
        # nWSR = np.array([200])
        # qp.init(H, g, A, lb, ub, lbA,
        #         ubA, nWSR)
        #
        # # X = qp.getObjval()
        # # print(nWSR)
        # result = np.zeros(21)
        # qp.getPrimalSolution(result)
        # # print("\nxOpt = [ %e, %e ];  objVal = %e\n\n" % (result[0], result[1], qp.getObjVal()))
        # # qp.printOptions()
        # # SolutionAnalysis.getKktViolation(qp,np.ndarray[0], np.ndarray[0], np.ndarray[0],0,0 )
        #
        # X = result.reshape(21, 1)

        Input = np.hstack([np.linalg.inv(Cdu1), np.zeros([self.Nu * self.Nc, 1])]) @ np.array(X) + np.linalg.inv(
            Cdu1) @ (-Cdu2)

        MPC_no_answer = False

        return Input, MPC_no_answer
