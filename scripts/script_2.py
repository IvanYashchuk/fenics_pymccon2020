import fenics
nx, ny = 16, 16
mesh = fenics.UnitSquareMesh(nx, ny)

fenics.plot(mesh)

import matplotlib.pyplot as plt
plt.savefig("mesh.png", dpi=150)

V = fenics.FunctionSpace(mesh, "CG", 1)
x, y = fenics.SpatialCoordinate(mesh)
a, b = 0.5, 10.0
expr = (a - x)**2 + b*(y - x**2)**2
rosenbrock_field = fenics.project(expr, V)

plt.clf()
contours = fenics.plot(rosenbrock_field)
plt.colorbar(contours)
plt.savefig("rosenbrock.png", dpi=150)

point = (0.5, 0.5)
print(f"Value at {point} is {rosenbrock_field(point)}")

from fenics import dx, assemble

print(f"Value of the integral is {assemble(x*y*dx)}")
print(f"Value of the integral is {assemble(rosenbrock_field*dx)}")
