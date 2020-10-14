import fenics, fenics_adjoint
from fenics import tr, Identity, sym, grad, inner, dx
L = 25.0
H = 1.0
Nx = 25
Ny = 1
mesh = fenics_adjoint.RectangleMesh(fenics.Point(0., 0.), fenics.Point(L, H), Nx, Ny, "crossed")
V = fenics.VectorFunctionSpace(mesh, 'CG', degree=2)

def left(x, on_boundary):
    return fenics.near(x[0], 0.)

bc = fenics_adjoint.DirichletBC(V, fenics_adjoint.Constant((0., 0.)), left)

def solve_elasticity(E, ρ_g):
    f = fenics.as_vector([fenics_adjoint.Constant(0), -ρ_g])
    nu = fenics_adjoint.Constant(0.3)
    mu = 0.5 * E / (1 + nu)
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    def sigma(v):
        return lmbda * tr(eps(v)) * Identity(2) + 2.0 * mu * eps(v)
    def eps(v):
        return sym(grad(v))
    u = fenics.TrialFunction(V)
    v = fenics.TestFunction(V)
    a = inner(sigma(u), eps(v))*dx
    l = inner(f, v)*dx
    w = fenics_adjoint.Function(V)
    fenics_adjoint.solve(a == l, w, bc)
    return w

E = fenics_adjoint.Constant(1e5)
ρ_g = fenics_adjoint.Constant(1e-3)

w = solve_elasticity(E, ρ_g)

import pyadjoint

J = fenics_adjoint.assemble(inner(w, w)*dx)
dJdρg = pyadjoint.compute_gradient(J, pyadjoint.Control(ρ_g))
print(f"dJ/dρg = {float(dJdρg)}")
