from sys import path
from casadi import *    # 不要用 import casadi.* 的方法
x = MX.sym("x")
print(jacobian(sin(x),x))

# --------------- python --------------
opti = casadi.Opti()    # 实例化一个 opti

x = opti.variable()     # 声明变量
y = opti.variable()

opti.minimize(  (y-x**2)**2   )     # 优化目标
opti.subject_to( x**2+y**2==1 )     # 约束1
opti.subject_to(       x+y>=1 )     # 约束2

opti.solver('ipopt')    # 设置求解器

sol = opti.solve()      # 求解

print(sol.value(x))
print(sol.value(y))


