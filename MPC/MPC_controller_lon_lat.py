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

        # y_ref
        for i in range(self.Np):
            self.x_ref[i] = ref[0][i]
            self.y_ref[i] = ref[1][i]
            self.phi_ref[i] = ref[2][i]
            self.x_ref_left[i] = ref_left[0][i]
            self.y_ref_left[i] = ref_left[1][i]
            self.phi_ref_left[i] = ref_left[2][i]
        for i in range(self.Np):
            self.y_ref_ext[i * self.Ny: (i + 1) * self.Ny, 0] = [self.x_ref[i], self.y_ref[i], self.phi_ref[i]]
            self.y_ref_left_ext[i * self.Ny: (i + 1) * self.Ny, 0] = [self.x_ref_left[i], self.y_ref_left[i], self.phi_ref_left[i]]

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

        # 预测时域和控制时域内的分块权重矩阵
        # 权重矩阵
        self.q = q
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

        # model linearization
        A_cell = np.zeros([self.Nx, self.Nx, self.Np])  # 2 * 2 * Np的矩阵，第三个维度为每个时刻的对应的A矩阵
        B_cell = np.zeros([self.Nx, self.Nu, self.Np])  # 2 * 1 * Np的矩阵
        C_cell = np.zeros([self.Nx, 1, self.Np])  # 2 * Np的矩阵

        for i in range(self.Np):  # 保存每个预测时间步的Ak，Bk，Ck矩阵
            A_cell[:, :, i] = np.eye(self.Nx) + self.T * (np.array(
                [[0, 0, -u_last[0] * np.sin(self.phi_ref[i])], [0, 0, u_last[0] * np.cos(self.phi_ref[i])], [0, 0, 0]]))
            B_cell[:, :, i] = self.T * (np.array([[np.cos(self.phi_ref[i]), 0], [np.sin(self.phi_ref[i]), 0],
                                                  [np.tan(u_last[1]) / self.L,
                                                   u_last[0] / (self.L * (np.cos(u_last[1]) ** 2))]]))
            C_cell[:, :, i] = self.T * (
                    np.array([u_last[0] * np.cos(self.phi_ref[i]), u_last[0] * np.sin(self.phi_ref[i]),
                              u_last[0] * np.tan(u_last[1]) / self.L]) - np.array(
                [[0, 0, -u_last[0] * np.sin(self.phi_ref[i])],
                 [0, 0, u_last[0] * np.cos(self.phi_ref[i])], [0, 0, 0]]) @ np.array([self.x_ref[i], self.y_ref[i],
                                                                                      self.phi_ref[i]]) - np.array(
                [[np.cos(self.phi_ref[i]), 0], [np.sin(self.phi_ref[i]), 0],
                 [np.tan(u_last[1]) / self.L, u_last[0] / (self.L * (np.cos(u_last[1]) ** 2))]]) @ np.array(
                [u_last[0], u_last[1]]))

        # dynamicmatrix:
        A_ext = np.zeros([self.Nx * self.Np, self.Nx])  # 3Np * 3的分块列矩阵
        B_ext = np.zeros([self.Nx * self.Np, self.Nu * self.Nc])  # 3Np * Nc的分块矩阵
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
            B_ext[i * self.Nx: (i + 1) * self.Nx, :] = A_cell[:, :, i] @ B_ext[(i - 1) * self.Nx: i * self.Nx, :] \
                                                       + B_ext[i * self.Nx: (i + 1) * self.Nx, :]

        C_ext = np.zeros([self.Nx * self.Np, 1])  # 2Np * 1的矩阵
        for i in range(self.Np):
            if i == 0:
                C_ext[i * self.Nx: (i + 1) * self.Nx, :] = C_cell[:, :, i]
            else:  # C的上一行乘以新的Ak再加上新的Ck得到C_bar
                C_ext[i * self.Nx: (i + 1) * self.Nx, :] = A_cell[:, :, i] @ C_ext[(i - 1) * self.Nx: i * self.Nx,
                                                                             :] + C_cell[:, :, i]

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

        # 标准形式min (1/2x'Px+q'x)   s.t. Gx<=h
        H_QP_du_e = np.empty((2, 2), dtype=object)
        # H_QP_du_e[0, 0] = np.transpose(self.Cy_ext @ B_ext) @ self.Q_cell @ (
        #         self.Cy_ext @ B_ext) + np.transpose(
        #     Cdu1) @ self.Rdu_cell @ Cdu1 + self.Ru_cell
        H_QP_du_e[0, 0] = np.transpose(self.Cy_ext @ B_ext)@ (
                self.Cy_ext @ B_ext) + np.transpose(
            Cdu1) @ self.Rdu_cell @ Cdu1 + self.Ru_cell
        H_QP_du_e[0, 1] = np.zeros([self.Nc * self.Nu, 1])
        H_QP_du_e[1, 0] = np.zeros([1, self.Nc * self.Nu])
        H_QP_du_e[1, 1] = self.rou * np.eye(1)
        H_QP_du_e = np.vstack([np.hstack(Mat_size) for Mat_size in H_QP_du_e])
        H_QP_du_e = 2 * H_QP_du_e

        # q为列向量
        f_QP_du_e = np.empty((1, 2), dtype=object)
        # f_QP_du_e[0, 0] = 2 * np.transpose(self.Cy_ext @ (
        #         A_ext @ x_current + C_ext) - self.y_ref_ext) @ self.Q_cell @ self.Cy_ext @ B_ext + 2 * np.transpose(
        #     Cdu2) @ self.Rdu_cell @ Cdu1

        f_QP_du_e[0, 0] = 2 * np.transpose(self.y_ref_left_ext - self.y_ref_ext) @ self.Q_cell @ self.Cy_ext @ B_ext\
        + 2 * np.transpose(self.Cy_ext @ (A_ext @ x_current + C_ext) - self.y_ref_left_ext) @ self.Cy_ext @ B_ext \
        + 2 * np.transpose(Cdu2) @ self.Rdu_cell @ Cdu1

        f_QP_du_e[0, 1] = np.zeros([1, 1])
        f_QP_du_e = np.vstack([np.hstack(Mat_size) for Mat_size in f_QP_du_e])
        f_QP_du_e = np.transpose(f_QP_du_e)

        lb = np.vstack((self.u_min_ext, self.e_min))
        ub = np.vstack((self.u_max_ext, self.e_max))

        H_QP_du_e[0, 1] = H_QP_du_e[0, 1] * 1
        H_QP_du_e[1, 0] = H_QP_du_e[1, 0] * 1
        H_QP_du_e[1, 1] = H_QP_du_e[1, 1] * 1

        f_QP_du_e[0, 0] = f_QP_du_e[0, 0] * 1
        f_QP_du_e[1, 0] = f_QP_du_e[1, 0] * 1

        A_du_eCons = np.empty([6, 2], dtype=object)
        ubA_du_eCons = np.empty([6, 1], dtype=object)
        A_du_eCons[0, 0] = Cdu1
        A_du_eCons[0, 1] = np.zeros([self.Nc * self.Nu, 1])
        A_du_eCons[1, 0] = - Cdu1
        A_du_eCons[1, 1] = np.zeros([self.Nc * self.Nu, 1])
        A_du_eCons[2, 0] = B_ext
        A_du_eCons[2, 1] = -np.ones([self.Np * self.Ny, 1])
        A_du_eCons[3, 0] = - B_ext
        A_du_eCons[3, 1] = -np.ones([self.Np * self.Ny, 1])
        A_du_eCons[4, 0] = self.Cy_ext @ B_ext
        A_du_eCons[4, 1] = -np.ones([self.Np * self.Ny, 1])
        A_du_eCons[5, 0] = -self.Cy_ext @ B_ext
        A_du_eCons[5, 1] = -np.ones([self.Np * self.Ny, 1])
        A_du_eCons = np.vstack([np.hstack(Mat_size) for Mat_size in A_du_eCons])

        ubA_du_eCons[0, 0] = self.du_max_ext - Cdu2
        ubA_du_eCons[1, 0] = -self.du_min_ext + Cdu2
        ubA_du_eCons[2, 0] = self.x_max_ext - (A_ext @ x_current + C_ext)
        ubA_du_eCons[3, 0] = -self.x_min_ext + (A_ext @ x_current + C_ext)
        ubA_du_eCons[4, 0] = self.y_max_ext - self.Cy_ext @ (A_ext @ x_current + C_ext)
        ubA_du_eCons[5, 0] = -self.y_min_ext + self.Cy_ext @ (A_ext @ x_current + C_ext)
        ubA_du_eCons = np.vstack([np.hstack(Mat_size) for Mat_size in ubA_du_eCons])


        # # cvxopt求解过程
        # P = matrix(H_QP_du_e)
        # q0 = f_QP_du_e.astype(np.double)
        # q = matrix(q0)
        # G = matrix(np.vstack((A_du_eCons, np.eye(self.Nc * self.Nu + 1), -np.eye(self.Nc * self.Nu + 1))))
        # h = matrix(np.vstack((ubA_du_eCons, ub, -lb)))
        # result = solvers.qp(P, q, G, h)  # 1/2x'Px+q'x   Gx<=h  Ax=b 注意使用qp时，每个参数要换成matrix
        # # 重要：print可被关闭
        # X = result['x']

        # # # qpoases求解过程
        # # Setting up QProblem object.
        H = H_QP_du_e
        g = f_QP_du_e.astype(np.double)[:, 0]
        A = A_du_eCons
        lb = lb[:, 0]
        ub = ub[:, 0]

        ubA = ubA_du_eCons[:, 0]
        lbA = -1e8 * np.ones(np.size(ubA))
        # ubA = 1e8 * np.ones(np.size(ubA))
        # Solve first QP.
        nWSR = np.array([200])

        # qp = QProblem(self.Nc * self.Nu + 1, self.Nc * self.Nu * self.Nc * self.Nu)
        qp = QProblem(self.Nc * self.Nu + 1, np.size(ubA))
        options = Options()

        options.printLevel = PrintLevel.NONE
        qp.setOptions(options)

        return_flag = qp.init(H, g, A, lb, ub, lbA,
                              ubA, nWSR)

        # X = qp.getObjval()
        # print(nWSR)
        result = np.zeros(self.Nc * self.Nu + 1)
        qp.getPrimalSolution(result)

        # qp.printOptions()
        # SolutionAnalysis.getKktViolation(qp,np.ndarray[0], np.ndarray[0], np.ndarray[0],0,0 )

        X = result.reshape(self.Nc * self.Nu + 1, 1)

        # Input = np.hstack([np.linalg.inv(Cdu1), np.zeros([self.Nu * self.Nc, 1])]) @ np.array(X) + np.linalg.inv(
        #     Cdu1) @ (-Cdu2)

        Input = X * 1.0
        MPC_unsolved = False
        if return_flag == 37:  # qp.init返回37说明无解，返回0说明有解
            MPC_unsolved = True

        # print(MPC_no_answer)
        return Input, MPC_unsolved
