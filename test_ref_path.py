from pyomo.environ import * 
path = '/home/wangliwen/Package/ipopt-linux64/ipopt'# 这里的 path 指的就是刚刚 ipopt 执行文件的路径
# from ipopt import minimize_unconstrained
#
# def objective_function(x):
#     return (x[0] - 1) ** 2 + (x[1] - 2) ** 2
#
# x_opt = minimize_unconstrained(objective_function, [0, 0])
# print(x_opt)
model = ConcreteModel()
model.x1 = Var(domain=Reals)
model.x2 = Var(domain=Reals)
# define objective function
# sense = minimize(Default) / maximize
model.f = Objective(expr = model.x1**2 + model.x2**2, sense=minimize)
# define constraints, equations or inequations
model.c1 = Constraint(expr = -model.x1**2 + model.x2 <= 0)
model.ceq1 = Constraint(expr = model.x1 + model.x2**2 == 2)
# use 'pprint' to print the model information
model.pprint()
SolverFactory('ipopt', executable=path).solve(model).write()
print('optimal f: {:.4f}'.format(model.f()))
print('optimal x: [{:.4f}, {:.4f}]'.format(model.x1(), model.x2()))

