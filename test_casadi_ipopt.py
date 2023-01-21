# from sys import path
# from casadi import *  # 不要用 import casadi.* 的方法
#
# # x = MX.sym("x")
# # print(jacobian(sin(x), x))
#
# # --------------- python --------------
# opti = casadi.Opti()  # 实例化一个 opti
#
#
# x = opti.variable(2, 1)  # 声明变量
# y = opti.variable()
# x = SX.sym("x")
# y = SX.sym("y")

# print(x)

# opti.minimize((x[1] - x[0] ** 2) ** 2)  # 优化目标
# opti.subject_to(x[0] ** 2 + x[1] ** 2 == 1)  # 约束1
# opti.subject_to(x[0] + x[1] >= 1)  # 约束2


# opti.solver('ipopt')  # 设置求解器
#
# sol = opti.solve()  # 求解
#
# print(sol.value(x[0]))
# print(sol.value(x[1]))

from sys import path
from casadi import *  # 不要用 import casadi.* 的方法

# x = MX.sym("x")
# print(jacobian(sin(x), x))

# --------------- python --------------
opti = casadi.Opti()  # 实例化一个 opti

x = opti.variable(2, 1)  # 声明变量
# y = opti.variable()
# x = SX.sym("x")
# y = SX.sym("y")
A = MX(np.zeros([2,1]))
A[0] = x[0]
A[1] = x[1]
print(x)
# m = np.array([[1, 2]])
Q = np.array([[1,0],[0,1]])
# expr = x.T @ Q @ x + 0.8
expr = 2*A[0]**2 + 3*A[1]**2
opti.minimize(expr)  # 优化目标
# opti.subject_to(x[0] ** 2 + x[1] ** 2 == 1)  # 约束1
opti.subject_to(x[0] + x[1] >= 1)  # 约束2

opti.solver('ipopt')  # 设置求解器

sol = opti.solve()  # 求解

print(sol.value(x[0]))
print(sol.value(x[1]))
