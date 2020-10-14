import fenics
L = 25.0
H = 1.0
Nx = 25
Ny = 1
mesh = fenics.RectangleMesh(fenics.Point(0., 0.), fenics.Point(L, H), Nx, Ny, "crossed")
fenics.plot(mesh)

import matplotlib.pyplot as plt
plt.savefig("beam_mesh.png", dpi=150)
plt.close()

from fenics import sym, grad
def eps(v):
    return sym(grad(v))

from fenics import Constant, tr, Identity
E = Constant(1e5)
nu = Constant(0.3)

mu = E/2/(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)

def sigma(v):
    return lmbda*tr(eps(v))*Identity(2) + 2.0*mu*eps(v)

rho_g = 1e-3
f = Constant((0, -rho_g))

from fenics import inner, dx

V = fenics.VectorFunctionSpace(mesh, 'CG', degree=2)
du = fenics.TrialFunction(V)
u_ = fenics.TestFunction(V)
a = inner(sigma(du), eps(u_))*dx
l = inner(f, u_)*dx

def left(x, on_boundary):
    return fenics.near(x[0], 0.)

bc = fenics.DirichletBC(V, Constant((0.,0.)), left)

u = fenics.Function(V, name="Displacement")
fenics.solve(a == l, u, bc)

fenics.plot(1e3*u, mode="displacement") # the solution is amplified with 1e3 for the visuals
plt.savefig("beam_displacement.png", dpi=150)
plt.close()

print("Maximal deflection:", -u(L, H/2.)[1])
print("Beam theory deflection:", float(rho_g * L**4 / (2/3  * E * H**3)))
