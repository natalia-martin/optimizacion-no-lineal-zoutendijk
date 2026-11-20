"""
Laboratorio: Algoritmo de Zoutendijk
Universidad Complutense de Madrid — Optimización (3º Matemáticas)

ENTORNO DE PRUEBAS
Ejecutad este fichero con vuestra implementación completada.
Compara Zoutendijk con penalización cuadrática y barrera logarítmica.

Uso:
    python test_env.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
import time

# ─── Importar implementación del alumno ───────────────────────────
try:
    from lab_template import zoutendijk
    STUDENT_IMPL = True
except Exception as e:
    print(f"[!] No se pudo cargar lab_template.py: {e}")
    STUDENT_IMPL = False


# ══════════════════════════════════════════════════════════════════
# PROBLEMAS DE PRUEBA
# ══════════════════════════════════════════════════════════════════

class Problem:
    """Contenedor de un problema de optimización con restricciones."""
    def __init__(self, name, f, grad_f, g_funcs, grad_g_funcs, x0, x_opt, f_opt, description):
        self.name = name
        self.f = f
        self.grad_f = grad_f
        self.g_funcs = g_funcs
        self.grad_g_funcs = grad_g_funcs
        self.x0 = np.array(x0, dtype=float)
        self.x_opt = np.array(x_opt, dtype=float)
        self.f_opt = f_opt
        self.description = description


def build_problems():
    problems = []

    # ─── Problema 1: Cuadrático con restricciones lineales ────────
    # min  (x1-4)² + (x2-4)²
    # s.t. x1 + x2 ≤ 5
    #      x1 ≥ 0  (→ -x1 ≤ 0)
    #      x2 ≥ 0  (→ -x2 ≤ 0)
    # Solución analítica: x* = (2.5, 2.5), f* = 4.5
    p1 = Problem(
        name="P1: Cuadrático con restricciones lineales",
        f=lambda x: (x[0]-4)**2 + (x[1]-4)**2,
        grad_f=lambda x: np.array([2*(x[0]-4), 2*(x[1]-4)]),
        g_funcs=[
            lambda x:  x[0] + x[1] - 5,   # x1+x2 ≤ 5
            lambda x: -x[0],               # x1 ≥ 0
            lambda x: -x[1],               # x2 ≥ 0
        ],
        grad_g_funcs=[
            lambda x: np.array([ 1.0,  1.0]),
            lambda x: np.array([-1.0,  0.0]),
            lambda x: np.array([ 0.0, -1.0]),
        ],
        x0=[1.0, 1.0],
        x_opt=[2.5, 2.5],
        f_opt=4.5,
        description="Cuadrático convexo con restricción lineal activa en el óptimo.\n"
                    "Caso favorable para Zoutendijk."
    )

    # ─── Problema 2: Disco — restricción cuadrática activa ────────
    # min  x1 + x2
    # s.t. x1² + x2² ≤ 2
    # Solución analítica: x* = (-1, -1), f* = -2
    p2 = Problem(
        name="P2: Lineal sobre disco",
        f=lambda x: x[0] + x[1],
        grad_f=lambda x: np.array([1.0, 1.0]),
        g_funcs=[
            lambda x: x[0]**2 + x[1]**2 - 2,
        ],
        grad_g_funcs=[
            lambda x: np.array([2*x[0], 2*x[1]]),
        ],
        x0=[0.5, 0.5],
        x_opt=[-1.0, -1.0],
        f_opt=-2.0,
        description="Restricción no lineal (círculo). El óptimo está en la frontera.\n"
                    "Comprueba cómo Zoutendijk navega hacia la frontera."
    )

    # ─── Problema 3: Zona acotada no convexa ─────────────────────
    # min  (x1-1)² + (x2-2.5)²
    # s.t. -x1 + 2*x2 ≤ 4
    #       x1 + 2*x2 ≤ 14
    #       x1 - 2*x2 ≤ 2
    #       -x1 ≤ 0, -x2 ≤ 0
    # Solución: x* = (1, 2.5), f* = 0 (interior factible)
    p3 = Problem(
        name="P3: Cuadrático interior (óptimo sin restricción activa)",
        f=lambda x: (x[0]-1)**2 + (x[1]-2.5)**2,
        grad_f=lambda x: np.array([2*(x[0]-1), 2*(x[1]-2.5)]),
        g_funcs=[
            lambda x: -x[0] + 2*x[1] - 4,
            lambda x:  x[0] + 2*x[1] - 14,
            lambda x:  x[0] - 2*x[1] - 2,
            lambda x: -x[0],
            lambda x: -x[1],
        ],
        grad_g_funcs=[
            lambda x: np.array([-1.0,  2.0]),
            lambda x: np.array([ 1.0,  2.0]),
            lambda x: np.array([ 1.0, -2.0]),
            lambda x: np.array([-1.0,  0.0]),
            lambda x: np.array([ 0.0, -1.0]),
        ],
        x0=[2.0, 0.5],
        x_opt=[1.0, 2.5],
        f_opt=0.0,
        description="El óptimo está en el interior. Ninguna restricción activa en x*.\n"
                    "Zoutendijk debe detectar que z* ≥ 0 cuando ∇f = 0."
    )

    return [p1, p2, p3]


# ══════════════════════════════════════════════════════════════════
# MÉTODOS DE REFERENCIA (penalización y barrera)
# ══════════════════════════════════════════════════════════════════

def solve_penalty(prob: Problem, mu_sequence=None, tol=1e-8):
    """Penalización cuadrática para restricciones de desigualdad
       usando max(0, gi(x))² como término de penalización."""
    if mu_sequence is None:
        mu_sequence = [1, 10, 100, 1000, 10000]

    x = prob.x0.copy()
    history = [x.copy()]
    n_evals = 0

    for mu in mu_sequence:
        def F(x):
            pen = sum(max(0.0, gi(x))**2 for gi in prob.g_funcs)
            return prob.f(x) + mu * pen

        result = minimize(F, x, method='L-BFGS-B', options={'ftol': tol/mu, 'maxiter': 1000})
        x = result.x
        n_evals += result.nfev
        history.append(x.copy())

    return x, prob.f(x), history, n_evals


def solve_barrier_log(prob: Problem, mu_sequence=None, tol=1e-8):
    """Barrera logarítmica — requiere punto estrictamente interior."""
    if mu_sequence is None:
        mu_sequence = [10, 1, 0.1, 0.01, 0.001]

    # Buscar punto interior
    x = prob.x0.copy()
    if any(gi(x) >= 0 for gi in prob.g_funcs):
        # Intentar perturbar hacia el interior
        for _ in range(1000):
            x_try = x - 0.01 * np.array([gi(x) for gi in prob.g_funcs] @ [1]*len(prob.g_funcs) * np.ones(len(x)))
            if all(gi(x_try) < 0 for gi in prob.g_funcs):
                x = x_try
                break
        else:
            return None, None, [], 0  # No se encontró interior

    history = [x.copy()]
    n_evals = 0

    for mu in mu_sequence:
        def F(xv):
            vals = [gi(xv) for gi in prob.g_funcs]
            if any(v >= 0 for v in vals):
                return 1e15
            barrier = -mu * sum(np.log(-v) for v in vals)
            return prob.f(xv) + barrier

        result = minimize(F, x, method='L-BFGS-B', options={'ftol': 1e-12, 'maxiter': 2000})
        x = result.x
        n_evals += result.nfev
        history.append(x.copy())

    return x, prob.f(x), history, n_evals


# ══════════════════════════════════════════════════════════════════
# VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════

def plot_problem_2d(prob: Problem, results: dict, ax: plt.Axes):
    """Dibuja curvas de nivel, región factible y trayectorias."""
    all_pts = []
    for res in results.values():
        if res.get('x_history'):
            all_pts.extend(res['x_history'])
    if all_pts:
        all_pts = np.array(all_pts)
        margin = 1.5
        xmin = min(all_pts[:,0].min(), prob.x_opt[0]) - margin
        xmax = max(all_pts[:,0].max(), prob.x_opt[0]) + margin
        ymin = min(all_pts[:,1].min(), prob.x_opt[1]) - margin
        ymax = max(all_pts[:,1].max(), prob.x_opt[1]) + margin
    else:
        xmin, xmax, ymin, ymax = -3, 7, -3, 7

    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 300),
                          np.linspace(ymin, ymax, 300))
    zz = np.array([[prob.f(np.array([xi, yi])) for xi in xx[0]] for yi in yy[:,0]])

    # Región factible
    feasible = np.ones_like(xx, dtype=bool)
    for gi in prob.g_funcs:
        feasible &= np.array([[gi(np.array([xi, yi])) <= 0
                               for xi in xx[0]] for yi in yy[:,0]])

    ax.contourf(xx, yy, feasible.astype(float), levels=[0.5, 1.5],
                colors=['#c8e6c9'], alpha=0.5)
    ax.contour(xx, yy, zz, levels=20, colors='gray', linewidths=0.6, alpha=0.5)

    colors = {'Zoutendijk': '#e53935', 'Penalización': '#1565c0', 'Barrera': '#2e7d32'}
    markers = {'Zoutendijk': 'o', 'Penalización': 's', 'Barrera': '^'}

    for name, res in results.items():
        if res.get('x_history') and len(res['x_history']) > 0:
            pts = np.array(res['x_history'])
            c = colors.get(name, 'black')
            ax.plot(pts[:,0], pts[:,1], '-', color=c, lw=1.5,
                    label=name, alpha=0.85)
            ax.plot(pts[0,0], pts[0,1], 'k*', ms=10, zorder=5)
            ax.plot(pts[-1,0], pts[-1,1], markers.get(name,'o'),
                    color=c, ms=8, zorder=5)

    ax.plot(*prob.x_opt, 'k+', ms=14, mew=2.5, zorder=6, label='Óptimo')
    ax.set_title(prob.name, fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)


def plot_convergence(prob: Problem, results: dict, ax: plt.Axes):
    """Dibuja error |f(xk) - f*| vs iteración."""
    colors = {'Zoutendijk': '#e53935', 'Penalización': '#1565c0', 'Barrera': '#2e7d32'}
    for name, res in results.items():
        if res.get('x_history'):
            errs = [abs(prob.f(np.array(x)) - prob.f_opt)
                    for x in res['x_history']]
            errs = [max(e, 1e-15) for e in errs]
            ax.semilogy(errs, color=colors.get(name, 'black'),
                        label=name, lw=1.8)
    ax.set_title('Convergencia', fontsize=9, fontweight='bold')
    ax.set_xlabel('Iteración / evaluación secuencial')
    ax.set_ylabel('|f(xk) − f*|')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


# ══════════════════════════════════════════════════════════════════
# TABLA DE RESULTADOS
# ══════════════════════════════════════════════════════════════════

def print_table(prob: Problem, results: dict):
    print(f"\n{'─'*65}")
    print(f"  {prob.name}")
    print(f"  {prob.description}")
    print(f"  Solución exacta: x* = {prob.x_opt},  f* = {prob.f_opt}")
    print(f"{'─'*65}")
    print(f"  {'Método':<18} {'x encontrada':<26} {'f(x)':<12} {'Error f':<12} {'Iters'}")
    print(f"  {'─'*60}")
    for name, res in results.items():
        if res.get('x') is None:
            print(f"  {name:<18} {'N/A':<26} {'N/A':<12} {'N/A':<12} {'N/A'}")
            continue
        x = res['x']
        fval = prob.f(np.array(x))
        err = abs(fval - prob.f_opt)
        nit = len(res.get('x_history', [])) - 1
        xstr = f"({x[0]:.4f}, {x[1]:.4f})"
        print(f"  {name:<18} {xstr:<26} {fval:<12.6f} {err:<12.2e} {nit}")
    print()


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run_tests():
    problems = build_problems()

    fig = plt.figure(figsize=(15, 4 * len(problems)))
    gs = gridspec.GridSpec(len(problems), 2, figure=fig, hspace=0.5, wspace=0.35)

    for row, prob in enumerate(problems):
        print(f"\n{'═'*65}")
        print(f"  Ejecutando: {prob.name}")
        print(f"{'═'*65}")

        results = {}

        # ── Zoutendijk (alumno) ──────────────────────────────────
        if STUDENT_IMPL:
            try:
                t0 = time.time()
                res = zoutendijk(
                    prob.f, prob.grad_f,
                    prob.g_funcs, prob.grad_g_funcs,
                    prob.x0, tol=1e-5, max_iter=300
                )
                elapsed = time.time() - t0
                results['Zoutendijk'] = {
                    'x': res['x'],
                    'x_history': res.get('x_history', []),
                    'time': elapsed
                }
                print(f"  Zoutendijk: {res['iterations']} iters, "
                      f"converged={res['converged']}, t={elapsed:.3f}s")
            except Exception as e:
                print(f"  [!] Error en Zoutendijk: {e}")
                results['Zoutendijk'] = {'x': None, 'x_history': []}
        else:
            print("  [!] Implementación de Zoutendijk no disponible.")

        # ── Penalización cuadrática ──────────────────────────────
        t0 = time.time()
        x_pen, f_pen, hist_pen, nev_pen = solve_penalty(prob)
        results['Penalización'] = {
            'x': x_pen,
            'x_history': hist_pen,
            'time': time.time() - t0
        }
        print(f"  Penalización: {len(hist_pen)-1} etapas μ, "
              f"f={f_pen:.6f}, t={results['Penalización']['time']:.3f}s")

        # ── Barrera logarítmica ──────────────────────────────────
        t0 = time.time()
        x_bar, f_bar, hist_bar, nev_bar = solve_barrier_log(prob)
        results['Barrera'] = {
            'x': x_bar,
            'x_history': hist_bar,
            'time': time.time() - t0
        }
        if x_bar is not None:
            print(f"  Barrera:     {len(hist_bar)-1} etapas μ, "
                  f"f={f_bar:.6f}, t={results['Barrera']['time']:.3f}s")
        else:
            print("  Barrera:     no se encontró punto interior")

        print_table(prob, results)

        # ── Gráficas ─────────────────────────────────────────────
        ax_tray = fig.add_subplot(gs[row, 0])
        ax_conv = fig.add_subplot(gs[row, 1])
        plot_problem_2d(prob, results, ax_tray)
        plot_convergence(prob, results, ax_conv)

    fig.suptitle("Laboratorio Zoutendijk — Comparativa de métodos",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig('resultados_lab.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('resultados_lab.png', bbox_inches='tight', dpi=150)
    print("\n[✓] Gráficas guardadas en resultados_lab.pdf y resultados_lab.png")
    plt.show()


if __name__ == '__main__':
    run_tests()
