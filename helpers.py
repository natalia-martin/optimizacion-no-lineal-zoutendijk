"""
Laboratorio: Algoritmo de Zoutendijk
Universidad Complutense de Madrid — Optimización (3º Matemáticas)

FUNCIONES BASE — NO MODIFICAR
Estas funciones están implementadas. Las usaréis en vuestra implementación.
"""

import numpy as np
from scipy.optimize import linprog, minimize_scalar
from typing import Callable, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────
# 1. SUBPROBLEMA LP — encontrar la dirección de descenso factible
# ─────────────────────────────────────────────────────────────────

def solve_direction_lp(
    grad_f: np.ndarray,
    active_grads: List[np.ndarray]
) -> Tuple[float, np.ndarray]:
    """
    Resuelve el subproblema lineal de Zoutendijk para encontrar la dirección dk.

    Problema:
        min_{z, d}  z
        s.t.
            ∇f(xk)ᵀ d  ≤  z
            ∇gi(xk)ᵀ d ≤  z,   para cada restricción activa i ∈ I
            -1 ≤ dj ≤ 1,        para cada componente j

    Parámetros
    ----------
    grad_f : array (n,)
        Gradiente de f en el punto actual xk.
    active_grads : lista de arrays (n,)
        Gradientes de las restricciones activas gi en xk.
        Pasar lista vacía [] si no hay restricciones activas.

    Devuelve
    --------
    z_opt : float
        Valor óptimo de z. Si z_opt < 0, existe dirección de descenso factible.
    d_opt : array (n,)
        Dirección óptima. Solo es válida si z_opt < 0.

    Notas
    -----
    Se usa scipy.optimize.linprog internamente (método simplex revisado).
    Las variables del LP son [z, d1, d2, ..., dn], es decir, n+1 variables.
    """
    n = len(grad_f)
    num_active = len(active_grads)

    # Variables del LP: v = [z, d1, ..., dn]  (n+1 variables)
    # Objetivo: minimizar z  →  c = [1, 0, ..., 0]
    c = np.zeros(n + 1)
    c[0] = 1.0

    # Restricciones de desigualdad: A_ub @ v <= 0
    # Cada fila: [-z + ∇f·d <= 0]  →  [-1 | ∇f^T]
    # Para activas: [-z + ∇gi·d <= 0]  →  [-1 | ∇gi^T]
    num_ineq = 1 + num_active
    A_ub = np.zeros((num_ineq, n + 1))
    b_ub = np.zeros(num_ineq)

    A_ub[0, 0] = -1.0          # coef de z
    A_ub[0, 1:] = grad_f       # coef de d

    for k, g_grad in enumerate(active_grads):
        A_ub[k + 1, 0] = -1.0
        A_ub[k + 1, 1:] = g_grad

    # Cotas: z libre, d ∈ [-1, 1]
    bounds = [(None, None)] + [(-1.0, 1.0)] * n

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.status not in (0, 1):
        # Si el LP no converge, devolvemos z=0 (sin dirección útil)
        return 0.0, np.zeros(n)

    z_opt = result.fun
    d_opt = result.x[1:]
    return float(z_opt), d_opt


# ─────────────────────────────────────────────────────────────────
# 2. BÚSQUEDA LINEAL — mejor λ dentro de (0, λ_max)
# ─────────────────────────────────────────────────────────────────

def line_search(
    f: Callable,
    x: np.ndarray,
    d: np.ndarray,
    lambda_max: float,
    method: str = 'golden'
) -> float:
    """
    Minimiza φ(λ) = f(x + λ·d) en el intervalo (0, λ_max).

    Parámetros
    ----------
    f : callable         Función objetivo.
    x : array (n,)       Punto actual.
    d : array (n,)       Dirección de descenso.
    lambda_max : float   Paso máximo permitido (no incluido).
    method : str         'golden' (sección áurea) o 'brent'.

    Devuelve
    --------
    lambda_opt : float   Paso óptimo en (0, lambda_max).
    """
    if lambda_max <= 1e-14:
        return 0.0

    phi = lambda lam: f(x + lam * d)

    result = minimize_scalar(
        phi,
        bounds=(1e-12, lambda_max * 0.9999),
        method='bounded',
        options={'xatol': 1e-10}
    )
    return float(result.x)


# ─────────────────────────────────────────────────────────────────
# 3. UTILIDADES — para visualización y comparativa
# ─────────────────────────────────────────────────────────────────

def check_feasibility(x: np.ndarray, g_funcs: List[Callable], tol: float = 1e-8) -> bool:
    """Devuelve True si x es factible (gi(x) <= tol para todo i)."""
    return all(gi(x) <= tol for gi in g_funcs)


def constraint_violation(x: np.ndarray, g_funcs: List[Callable]) -> float:
    """Suma de violaciones max(0, gi(x)) — útil para diagnóstico."""
    return sum(max(0.0, gi(x)) for gi in g_funcs)
