r"""Heat equation in a rectangle with several point sources.

Solves

    du/dt = kappa * Laplace(u) + sum_k q_k * delta(x - x_k)    in  Omega = (-a, a) x (-b, b)
        u = 0                                                   on  dOmega
    u(x, 0) = cos(pi x / 2a) cos(pi y / 2b)

with Q2 finite elements in space and the theta-method (Crank-Nicolson) in time:

    (M + theta dt L) U^{n+1} = (M - (1 - theta) dt L) U^n
                               + dt [theta F(t_{n+1}) + (1 - theta) F(t_n)]

where the point-source load vector is F_i(t) = sum_k q_k(t) phi_i(x_k).

The result is checked against the exact eigenfunction-series solution
at probe points that are not on top of a source.

Usage:
    python heat_point_sources.py          # animate on screen
    python heat_point_sources.py --gif    # write heat_point_sources.gif
"""
from math import ceil
from typing import Callable, Iterator, Tuple

import numpy as np
from scipy.sparse.linalg import splu

from skfem import *
from skfem.models.poisson import laplace, mass

# ---------------------------------------------------------------------------
# Problem data
# ---------------------------------------------------------------------------
halfwidth = np.array([2., 3.])     # a, b
ncells = 2**5
diffusivity = 5.                   # kappa


h = 2.0 
u_ext = 0.0

# Point sources: positions (must be inside Omega) and strengths q_k.
# Make a strength time-dependent by replacing it with a function of t.
source_points = np.array([[0,  0],
                          [ 1.0, -1.0],
                          [ 0.5,  2.0],
                          [-1.2, -2.0]])
source_strengths: list = [100., 80., -60, 70.]   # negative = sink

dt = .01
theta = 0.5                        # 0.5 = Crank-Nicolson, 1 = backward Euler
t_final = 10.0                      # stop here, or earlier once steady
steady_tol = 1e-6                  # stop when ||U^{n+1} - U^n||_inf < tol

# Points where the solution is printed and compared with the exact series
probe_points = np.array([[0.0, 0.0],
                         [1.0, 1.0]])

# ---------------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------------
assert np.all(np.abs(source_points) < halfwidth), 'sources must lie inside the domain'

mesh = MeshQuad.init_tensor(
    np.linspace(-1, 1, 2 * ncells) * halfwidth[0],
    # true division here, so the cells are roughly square
    np.linspace(-1, 1, 2 * ncells * ceil(halfwidth[1] / halfwidth[0])) * halfwidth[1])

element = ElementQuad2()
basis = Basis(mesh, element)
fbasis = FacetBasis(mesh, element) 

@BilinearForm
def robin(u, v, w): 
    return h * u * v

@LinearForm
def ambient(v,w):
    return h* u_ext * v

L = diffusivity * asm(laplace, basis) + asm(robin, fbasis)
M = asm(mass, basis)
G = asm(ambient, fbasis)

A = M + theta * L * dt
B = M - (1 - theta) * L * dt

backsolve = splu(A.T).solve

# Load from point sources: delta(x - x_k) tested with phi_i gives phi_i(x_k),
# so P[k, i] = phi_i(x_k) and F = P^T q.
P = basis.probes(source_points.T)


def strengths(t: float) -> np.ndarray:
    return np.array([q(t) if callable(q) else q for q in source_strengths])


def load(t: float) -> np.ndarray:
    F = P.T @ strengths(t) + G
    return F


u_init = np.cos(np.pi * basis.doflocs / 2 / halfwidth[:, None]).prod(0)

# Steady state  L U = F  (only meaningful when the strengths are constant)
u_steady = splu(L.tocsc()).solve(load(0.))


def evolve(t: float, u: np.ndarray) -> Iterator[Tuple[float, np.ndarray]]:
    F_old = load(t)
    while t < t_final - 1e-12:
        F_new = load(t + dt)
        u_new = backsolve(B @ u + dt * (theta * F_new + (1 - theta) * F_old))
        change = np.linalg.norm(u_new - u, np.inf)
        t, u, F_old = t + dt, u_new, F_new
        yield t, u
        #if change < steady_tol:
            #break


# ---------------------------------------------------------------------------
# Exact solution (eigenfunction series), valid for constant strengths
#   psi_mn = X_m(x) Y_n(y),  X_m = sin(m pi (x + a) / 2a),  ||psi_mn||^2 = a b
#   lambda_mn = kappa pi^2 / 4 (m^2 / a^2 + n^2 / b^2)
#   c_mn(t) = c_mn(0) e^{-lambda t} + s_mn / lambda (1 - e^{-lambda t})
#   s_mn = sum_k q_k psi_mn(x_k) / (a b),   c_mn(0) = delta_{m1} delta_{n1}
# ---------------------------------------------------------------------------
def exact_at(points: np.ndarray, t: float, nmodes: int = 1500) -> np.ndarray:
    a, b = halfwidth
    k = np.arange(1, nmodes + 1)

    def X(x): return np.sin(np.outer(x + a, k) * np.pi / (2 * a))   # (npts, nmodes)
    def Y(y): return np.sin(np.outer(y + b, k) * np.pi / (2 * b))

    lam = diffusivity * np.pi**2 / 4 * (k[:, None]**2 / a**2 + k[None, :]**2 / b**2)
    q = strengths(0.)
    s = np.einsum('k,km,kn->mn', q, X(source_points[:, 0]), Y(source_points[:, 1])) / (a * b)
    decay = np.exp(-lam * t)
    c = s / lam * (1 - decay)
    c[0, 0] += decay[0, 0]
    return np.einsum('pm,mn,pn->p', X(points[:, 0]), c, Y(points[:, 1]))


probe = basis.probes(probe_points.T)

if __name__ == '__main__':

    from argparse import ArgumentParser
    from pathlib import Path

    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt

    from skfem.visuals.matplotlib import plot

    parser = ArgumentParser(description='heat equation in a rectangle with point sources')
    parser.add_argument('-g', '--gif', action='store_true', help='write animated GIF')
    args = parser.parse_args()

    ax = plot(mesh, u_init[basis.nodal_dofs.flatten()], shading='gouraud')
    title = ax.set_title('t = 0.00')
    field = ax.get_children()[0]
    vmin = min(u_init.min(), u_steady.min())
    vmax = max(u_init.max(), u_steady.max())
    field.set_clim(vmin, vmax)
    #ax.plot(*source_points.T, 'k+', ms=10, mew=2, label='sources')
    #ax.plot(*probe_points.T, 'wo', mfc='none', label='probes')
    ax.legend(loc='upper right', fontsize='small')
    fig = ax.get_figure()
    fig.colorbar(field)

    header = '   t   ' + ''.join(
        f'  u_h{tuple(p)}  err   ' for p in probe_points.tolist())
    print(header)

    def update(event):
        t, u = event
        uh = probe @ u
        ue = exact_at(probe_points, t)
        print(f'{t:5.2f} ' + ''.join(f'  {v:8.4f} {v - w:+8.4f}' for v, w in zip(uh, ue)))
        title.set_text(f'$t$ = {t:.2f}')
        field.set_array(u[basis.nodal_dofs.flatten()])

    animation = FuncAnimation(fig, update, evolve(0., u_init),
                              repeat=False, interval=50, cache_frame_data=False)
    if args.gif:
        animation.save(Path(__file__).with_suffix('.gif'), 'pillow')
    else:
        plt.show()
