# wildcard import is not advisable in general
# however current fenics tutorials are written in this way
from fenics import *
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
f = 1.0
a = inner(grad(u), grad(v)) * dx # left-hand-side of the equation (1)
L = f * v * dx # right-hand-side of the equation (1)
bc = DirichletBC(V, 0.0, "on_boundary")
sol = Function(V)
solve(a==L, sol, bc)

import matplotlib.pyplot as plt
contours = plot(sol)
plt.colorbar(contours)
plt.savefig("poisson_solution.png", dpi=150)
