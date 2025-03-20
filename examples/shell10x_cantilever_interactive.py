from __future__ import absolute_import

import sys

sys.path.append(".")

import numpy as nm
import numpy as np
import sfepy.mechanics.shell10x as sh
from sfepy.base.base import IndexedStruct
from sfepy.discrete import Equation, Equations, FieldVariable, Integral, Material, Problem
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.discrete.fem import FEDomain, Field, Mesh
from sfepy.mesh.mesh_generators import gen_block_mesh
from sfepy.solvers.auto_fallback import AutoDirect
from sfepy.solvers.nls import Newton
from sfepy.terms import Term


# Crear el tensor elástico para flexión (Cb)
def create_Cb(young, poisson, thickness):
    factor = young * thickness**3 / (12 * (1 - poisson**2))
    Cb = factor * np.array([[1, poisson, 0], [poisson, 1, 0], [0, 0, (1 - poisson) / 2]])
    return Cb


# Crear el tensor elástico para cortante (Cs)
def create_Cs(young, poisson, thickness):
    k = 5 / 6  # Factor de corrección por cortante
    factor = young * thickness * k / (2 * (1 + poisson))
    Cs = factor * np.array([[1, 0], [0, 1]])
    return Cs


# Crear el tensor elástico para membrana (Cm)
def create_Cm(young, poisson):
    Ex = young
    Ey = young
    nu_xy = poisson
    nu_yx = poisson
    G = young / (2 * (1 + poisson))
    factor = 1 / (1 - nu_xy * nu_yx)
    Cm = factor * np.array(
        [[Ex, nu_yx * Ex, 0], [nu_xy * Ey, Ey, 0], [0, 0, (1 - nu_xy * nu_yx) * G]]
    )
    return Cm


# Combinar las matrices en un tensor elástico D
def create_elastic_tensor(young, poisson, thickness):
    Cb = create_Cb(young, poisson, thickness)
    Cs = create_Cs(young, poisson, thickness)
    Cm = create_Cm(young, poisson)

    # Construir el tensor elástico D de tamaño (6, 6)
    D = np.zeros((6, 6))

    # Asignar Cb a las primeras 3 filas y columnas (flexión)
    D[:3, :3] += Cb

    # Asignar Cs a las siguientes 2 filas y columnas (cortante)
    D[3:5, 3:5] += Cs

    # Asignar Cm a las últimas 3 filas y columnas (membrana)
    D[3:, 3:] += Cm

    return D


def make_mesh(dims, shape, transform=None):
    """
    Generate a 2D rectangle mesh in 3D space, and optionally apply a coordinate
    transform.
    """
    _mesh = gen_block_mesh(dims, shape, [0, 0], name="shell10x", verbose=False)

    coors = nm.c_[_mesh.coors, nm.zeros(_mesh.n_nod, dtype=nm.float64)]
    coors = nm.ascontiguousarray(coors)

    conns = [_mesh.get_conn(_mesh.descs[0])]

    mesh = Mesh.from_data(
        _mesh.name, coors, _mesh.cmesh.vertex_groups, conns, [_mesh.cmesh.cell_groups], _mesh.descs
    )

    return mesh


def make_domain(dims, shape, transform=None):
    """
    Generate a 2D rectangle domain in 3D space, define regions.
    """
    xmin = (-0.5 + 1e-12) * dims[0]
    xmax = (0.5 - 1e-12) * dims[0]

    mesh = make_mesh(dims, shape, transform=transform)
    domain = FEDomain("domain", mesh)
    domain.create_region("Omega", "all")
    domain.create_region("Gamma1", "vertices in (x < %.14f)" % xmin, "facet")
    domain.create_region("Gamma2", "vertices in (x > %.14f)" % xmax, "facet")

    return domain


def solve_problem(shape, dims, young, poisson, force, transform=None):
    domain = make_domain(dims[:2], shape, transform=transform)

    omega = domain.regions["Omega"]
    gamma1 = domain.regions["Gamma1"]
    gamma2 = domain.regions["Gamma2"]

    # 1. Definir el campo y las variables
    field = Field.from_args("fu", np.float64, 6, omega, approx_order=1, poly_space_basis="shell10x")
    u = FieldVariable("u", "unknown", field)
    v = FieldVariable("v", "test", field, primary_var_name="u")

    # 2. Configurar el espesor y la carga
    thickness = dims[2]
    pload = [[0.0, 0.0, force / shape[1], 0.0, 0.0, 0.0]] * shape[1]

    # 3. Definir materiales
    # Crear las matrices de rigidez
    D = create_elastic_tensor(young, poisson, thickness)
    D1 = sh.create_elastic_tensor(young=young, poisson=poisson)

    m = Material("m", D=D1, values={".drill": 1e-7})
    load = Material("load", values={".val": pload})

    # 4. Configurar términos e ecuaciones
    aux = Integral("i", order=2)
    qp_coors, qp_weights = aux.get_qp("3_8")
    qp_coors[:, 2] = thickness * (qp_coors[:, 2] - 0.5)
    qp_weights *= thickness

    integral = Integral("i", coors=qp_coors, weights=qp_weights, order="custom")

    t1 = Term.new("dw_shell10x(m.D, m.drill, v, u)", integral, omega, m=m, v=v, u=u)
    t2 = Term.new("dw_point_load(load.val, v)", integral, gamma2, load=load, v=v)
    eq = Equation("balance", t1 - t2)
    eqs = Equations([eq])

    # 5. Configurar el problema
    pb = Problem("elasticity with shell10x", equations=eqs)

    fix_u = EssentialBC("fix_u", gamma1, {"u.all": 0.0})
    pb.set_bcs(ebcs=Conditions([fix_u]))

    # 6. Configurar solvers
    ls = AutoDirect({})
    nls_status = IndexedStruct()
    nls = Newton({}, lin_solver=ls, status=nls_status)
    pb.set_solver(nls)

    # 7. Actualizar materiales con time stepper y equations
    ts = pb.get_timestepper()
    m.time_update(ts, pb.equations)
    load.time_update(ts, pb.equations)
    pb.update_materials()

    # 11. Resolver el problema
    state = pb.solve()

    K_full = pb.equations.evaluate(mode="weak", dw_mode="matrix", asm_obj=pb.mtx_a)
    K_array = K_full.toarray()
    print(K_array.shape)
    # print_matrix(K_array, max_size=24)

    return pb, state, u, gamma2


def get_analytical_displacement(dims, young, force, transform=None):
    """
    Returns the analytical value of the max. displacement according to
    Euler-Bernoulli theory.
    """
    l, b, h = dims

    if transform is None:
        moment = b * h**3 / 12.0
        u = force * l**3 / (3 * young * moment)

    elif transform == "bend":
        u = force * 3.0 * nm.pi * l**3 / (young * b * h**3)

    elif transform == "twist":
        u = None

    return u


def main():
    silent = False

    w = 0.2
    h = 0.01
    t = 0.001
    nx = 20
    ny = 2

    E = 210e9
    nu = 0.3

    dims = nm.array([w, h, t], dtype=nm.float64)
    young = E
    poisson = nu
    force = 1

    shape = (nx, ny)

    transform = "twist"

    if transform is None:
        ilog = 2
        labels = ["u_3"]

    elif transform == "bend":
        ilog = 0
        labels = ["u_1"]

    u_exact = get_analytical_displacement(dims, young, force, transform=transform)
    pb, state, u, _ = solve_problem(shape, dims, young, poisson, force, transform=transform)

    pb.save_state("shell10x_cantilever.vtk", state)

    print(f"max. displacement {u.data[0].max()}")
    print("analytical value:", u_exact)


if __name__ == "__main__":
    main()
