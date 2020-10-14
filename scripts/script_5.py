import fenics, fenics_adjoint
fenics.set_log_level(fenics.LogLevel.ERROR)
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

from fenics_pymc3 import create_fenics_theano_op
# Define FEniCS template representation of Theano/NumPy input
# that is we promise that our arguments are of the following types
# the choice is between Constant and Function
templates = (fenics_adjoint.Constant(0.0), fenics_adjoint.Constant(0.0))
theano_fem_solver = create_fenics_theano_op(templates)(solve_elasticity)

import pymc3 as pm
import theano.tensor as tt

loads = [[1.e-2], [2.5e-2], [5.e-2]]
measurements = [0.04254, 0.10635, 0.21270]

with pm.Model() as model:

    E = pm.Normal("E", mu=1.1e5, sigma=0.3e5, shape=(1,))

    maximum_deflections = []
    for i in range(len(measurements)):
        ρ_g = loads[i]
        predicted_displacement = theano_fem_solver(E, ρ_g)
        maximum_deflection = tt.max(-predicted_displacement)
        maximum_deflections.append(maximum_deflection)
    maximum_deflections = tt.stack(maximum_deflections)

    d = pm.Normal("d", mu=maximum_deflections, sd=1e-3, observed=measurements)

map_estimate = pm.find_MAP(model=model)
print(f"MAP estimate of E is {map_estimate['E']}")
print(f"Analytical estimate of E is {loads[0][0] * L**4 / (2/3  * H**3 * measurements[0])}")

with model:
    trace = pm.sample(100, chains=1, cores=1, tune=100)

pm.summary(trace)

with model:
    advi_fit = pm.fit(method='advi')

