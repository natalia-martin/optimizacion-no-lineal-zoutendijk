"""
Laboratorio: Algoritmo de Zoutendijk — UCM Optimización
Completad las cuatro funciones. No modificar helpers.py.
"""

import numpy as np
from typing import Callable, List, Tuple
from helpers import solve_direction_lp, line_search, check_feasibility


# ─────────────────────────────────────────────────────────────────
# FUNCIÓN 1
# ─────────────────────────────────────────────────────────────────

def compute_lambda_max(
    x: np.ndarray,
    d: np.ndarray,
    g_funcs: List[Callable],
    tol: float = 1e-10
) -> float:
    """
    Calcula el mayor λ ≥ 0 tal que gi(x + λ·d) ≤ 0 para todo i.

    Parámetros
    ----------
    x       : punto factible actual
    d       : dirección de descenso
    g_funcs : lista de callables; gi(x) -> escalar
    tol     : tolerancia de la bisección

    Devuelve
    --------
    lambda_max : float
    """
    lambda_max=np.inf #Comenzamos poniendo que el paso máximo λ es infinito.
    for gi in g_funcs: #Recorremos todas las restricciones.
        if gi(x)>1e-8:#Comprobamos que el punto x actual es factible.
            raise ValueError("El punto actual no es factible")
        #Fase 1: Búsqueda exponencial. Buscamos un intervalo [lo,hi] tal 
        #que gi(x+lo*d)<=0 (factible) y gi(x+hi*d)>0 (no factible).
        lo=0.0 # ya que x+0*d=x es factible.
        hi=1.0 #Se empieza con λ=1, el primer valor que probamos.
        limite=False #Esta nueva variable sirve para detectar cuando la restricción
        #no limita el paso en esta dirección, esto ocurre si hi crece mucho y sigue siendo factible.
        while gi(x+hi*d)<=0: #Mientras gi(x+hi*d)<=0, sigue siendo factible.
            lo=hi #Guardamos hi como el último valor factible que conocemos.
            hi=2.0*hi #Duplicamos hi para buscar los siguientes valores.
            #Si hi es muy grande y no deja de ser factible esa restriccion no limita,
            #el break hace que salgas del bucle while. Después, limite será True pasará a la
            #siguiente restricción del bucle inicial(for gi in g_funcs).
            if hi>1e15:
                limite=True
                break
        if limite:
            continue
        #Si hemos llegado aquí, tenemos un intervalo [lo, hi] como buscábamos para la restricción i.
        #Entre lo y hi habrá una raíz de gi(x+λ*d).
        #Fase 2:Bisección en [lo,hi]. Afinamos el valor de la raíz.
        #Si el punto medio mid sigue siendo factible, imponemos lo=mid, Si no, hi=mid. Así vamos acortando 
        #cada vez más el intervalo donde se encuentra la raíz.
        if gi(x+hi*d)>0:
            while (hi-lo)>tol:
                mid=(lo+hi)/2.0
                if gi(x+mid*d)<=0:
                    lo=mid
                else:
                    hi=mid
            lambda_max = min(lambda_max, lo) #Actualizamos λ como el mínimo de los obtenidos con cada
            #restricción, para que sea factible en todas. 
    return lambda_max

# ─────────────────────────────────────────────────────────────────
# FUNCIÓN 2
# ─────────────────────────────────────────────────────────────────

def identify_active_constraints(
    x: np.ndarray,
    g_funcs: List[Callable],
    tol: float = 1e-6
) -> List[int]:
    """
    Devuelve los índices i tales que gi(x) >= -tol.

    Parámetros
    ----------
    x       : punto actual
    g_funcs : lista de callables; gi(x) -> escalar
    tol     : umbral de activación

    Devuelve
    --------
    Lista de índices (puede ser vacía).
    """
    #Creamos una lista de índices, active_idx. Recorremos todas
    #las restricciones. Si la restricción está activa, añadimos su 
    # índice a active_idx. La función devolverá esta lista, con todos
    #los índices de las restricciones activas.
    active_idx=[]
    for i,gi in enumerate(g_funcs):
        if gi(x)>=-tol:
            active_idx.append(i)
    return active_idx

# ─────────────────────────────────────────────────────────────────
# FUNCIÓN 3
# ─────────────────────────────────────────────────────────────────

def zoutendijk_step(
    x: np.ndarray,
    f: Callable,
    grad_f: Callable,
    g_funcs: List[Callable],
    grad_g_funcs: List[Callable],
    tol_active: float = 1e-6
) -> Tuple[bool, np.ndarray, dict]:
    """
    Un paso completo del algoritmo de Zoutendijk.

    Parámetros
    ----------
    x            : punto factible actual
    f            : función objetivo
    grad_f       : gradiente de f; grad_f(x) -> array (n,)
    g_funcs      : restricciones gi(x) <= 0
    grad_g_funcs : gradientes; grad_g_funcs[i](x) -> array (n,)
    tol_active   : tolerancia para restricciones activas

    Devuelve
    --------
    converged : True si z* >= 0 (condición de optimalidad)
    x_new     : punto actualizado (o x si converged)
    info      : dict con claves 'z', 'd', 'lambda', 'f_new', 'active'
    """
    #Calculamos gradiente de f en x.
    gf_x=grad_f(x)
    #Identificamos las restricciones activas.
    active_idx=identify_active_constraints(x, g_funcs, tol_active)
    #Obtenemos los gradientes de las restricciones activas.
    #Para llamar a solve_direction_lp, active_grads debe ser una
    #matriz (n_active x n) o [] si no hay activas (helpers.m).
    if len(active_idx)==0:
        active_grads=[]
    else:
        lista_grads=[]
        for i in active_idx:
            grad_i=grad_g_funcs[i](x)
            lista_grads.append(grad_i)
        active_grads=np.array(lista_grads)
    #Resolvemos el LP con solve_direction_lp.
    z_opt, d = solve_direction_lp(gf_x, active_grads)
    #Decidimos entre los dos casos (z*>=0 vs z*<0).
    # Si z*>=0, no hay dirección de descenso factible
    if z_opt>=-tol_active:
        converged=True
        x_new = x.copy()
        info={'z': float(z_opt),
            'd': d,
            'lambda': 0.0, 
            'f_new': float(f(x_new)), 
            'active': active_idx}
        return converged, x_new, info
    else:
        #Si z*<0 si hay dirección de descenso factible.
        converged=False
        lambda_max=compute_lambda_max(x,d,g_funcs) #Hallamos el mayor paso factible  λ_max.
        lambda_star=line_search(f,x,d,lambda_max)#Y el mejor λ dentro de 
        #(0, λ_max) con la función auxiliar de helpers.
        x_new=x+lambda_star*d #Actualizamos el punto y la información.
        info = {'z': float(z_opt),
            'd': d,
            'lambda': float(lambda_star), 
            'f_new': float(f(x_new)), 
            'active': active_idx}
        return converged, x_new, info

# ─────────────────────────────────────────────────────────────────
# FUNCIÓN 4
# ─────────────────────────────────────────────────────────────────

def zoutendijk(
    f: Callable,
    grad_f: Callable,
    g_funcs: List[Callable],
    grad_g_funcs: List[Callable],
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 500
) -> dict:
    """
    Algoritmo de Zoutendijk completo.

    Parámetros
    ----------
    f, grad_f    : función objetivo y gradiente
    g_funcs      : lista de restricciones gi(x) <= 0
    grad_g_funcs : lista de gradientes ∇gi(x)
    x0           : punto inicial (DEBE ser factible)
    tol          : tolerancia de parada
    max_iter     : límite de iteraciones

    Devuelve
    --------
    dict con: 'x', 'f', 'iterations', 'converged', 'history', 'x_history'
    x_history debe incluir x0 como primer elemento.

    Raises
    ------
    ValueError si x0 no es factible.
    """
    #Convertimos el x0 a array porque la función de helpers
    #check_feasibility lo necesita así.
    x_actual=np.array(x0, dtype=float).copy()
    #Comprobamos que el punto inicial es factible.
    if not check_feasibility(x_actual, g_funcs, tol):
        raise ValueError("x0 no es factible")
    #Creamos una lista con el historial de cada iteración.
    history=[]
    #Creamos una lista de los puntos recorridos, añadiendo el x0 en primer lugar.
    x_history=[x_actual.copy()]
    #Creamos la variable conv que indica si el algoritmo ha convergido.
    conv=False
    #Creamos un bucle hasta el límite de iteraciones.
    for i in range(max_iter):
        converged, x_new, info=zoutendijk_step(x_actual, f, grad_f, g_funcs, grad_g_funcs, tol)
        history.append(info) #Guardamos la información de la iteración.
        x_actual=x_new.copy() #Actualizamos el punto actual.
        x_history.append(x_actual.copy()) #Guardamos el nuevo punto en la lista x_history.
        #Si converge, entonces paramos.
        if converged:
            conv=True
            break

    return {'x': x_actual,
            'f': f(x_actual),
            'iterations': len(history),
            'converged': conv,
            'history': history,
            'x_history': x_history}


# ─────────────────────────────────────────────────────────────────
# REFLEXIÓN (responder aquí como comentario)
# ─────────────────────────────────────────────────────────────────
#
# P1. La barrera logarítmica exige x0 estrictamente interior (gi(x0) < 0).
#     Zoutendijk solo pide gi(x0) <= 0. ¿Qué ocurre si x0 está en la frontera?
#     ¿Es un problema para Zoutendijk? ¿Por qué la barrera no puede permitírselo?
#
# P2. La restricción dj ∈ [-1,1] (donde dj es la j-ésima componente de d)
#     no aparece en la formulación teórica. ¿Por qué es necesaria en la práctica?
#     ¿Qué degeneración ocurre sin ella?
#
# P3. Comparando los tres métodos en los tres problemas: ¿dónde es Zoutendijk
#     más lento que penalización? ¿Hay algún caso donde es más rápido?
#     Propón una hipótesis que explique ambas observaciones.
#
# P1:Si x0 está en la frontera (gi(x0)=0 para alguna restricción) no es un problema 
# para Zoutendijk. Este solo exige factibilidad, gi(x0)<=0. Si una
# restricción está en la frontera, pasa a formar parte de las restricciones 
# activas I y su gradiente se tendrá en cuenta para el cálculo de la dirección.
# Así, la dirección de descenso d que calculamos hará decrecer f y además, 
# respetará las restricciones activas. Por esto Zoutendijk puede comenzar tanto en
# puntos interiores como en la frontera.
# Por otro lado la barrera logarítima no puede comenzar en la frontera porque
# cambia el problema con restricciones, a uno sin restricciones del tipo:
# min {f(x)-mu*sum(ln(-gi(x)))}. Para que el logarítmo esté definido, se necesita que 
# -gi(x)>0, o lo que es lo mismo, gi(x)<0 (estrictamente). Si estuvieramos en el 
# caso en el que gi(x0)=0, tendríamos ln(0) que no está definido. Cuando gi tiende a
# 0-, el término ln(-gi) tiende a -infinito y -mu*ln(-gi) tenderá a más infinito, es
# decir, la función crece sin límite al acercarse a la frontera. Por esto la barrera
# necesita un punto estrictamente interior, mientra que Zoutendijk no. 

# P2:Esa cota prohíbe escalar, es decir, evita que el problema sea no acotado.
# Si existe una dirección d tal que ∇f(x)^T*d < 0 y respeta también las restricciones 
# activas (∇gi(x)^T*d <= 0), entonces cualquier múltiplo c*d (con c>0 entero) cumpliría
# las condiciones (se podría "estirar" la dirección indefinidamente). Entonces, el valor 
# z el problema  min z s.a. ∇f(x)^T*d<=z, ∇gi(x)^T*d<=z podría hacerse arbitrariamente
# negativo aumentando c y sería no acotado. Si imponemos dj ∈ [-1,1] se evita esto,
# fijando el tamaño máximo para d. Así el algoritmo no mejoraría z "estirando" la dirección,
# sino que debe buscar una dirección que aporte una mejora real.

# P3: Podemos observar que Zoutendijk converge en menos iteraciones que los otros métodos en
# todos los casos. Pero el tiempo total es parecido, o incluso Zoutendijk tarda algo más, 
# ya que es más costoso. Esto se debe a que tiene que resolver un problema lineal y también
# calcular el paso máximo que haga que siga siendo factible.
# Zoutendijk es más eficaz en los problemas P1 y P2, donde el óptimo está en la frontera.
# Ya que Zoutendijk trabaja directamente con las restricciones activas, moviéndose dentro 
# de la región factible y llegando con precisión a la solución.
# En el P3, al estar el óptimo en el interior, Zoutendijk funciona pero tarda más que 
# los otros métodos y no aporta ninguna ventaja frente a los demás.
# En conclusión, Zoutendijk trabaja mejor usando la geometría del problema, siendo más preciso 
# en los problemas con la solución óptima en la frontera.
#
# OPCIONAL. El algoritmo está formulado para restricciones de desigualdad.
# Suponed que el problema incluye también h(x) = 0.
# (a) ¿Cómo incorporaríais la igualdad sin modificar ninguna función implementada?
# (b) ¿Qué implicación tiene esa transformación sobre el conjunto I en cada iteración?
# (c) ¿Qué efecto tiene sobre compute_lambda_max cuando h(x^{(k)}) = 0?
#
# OPCIONAL:
# (a) La igualdad la transformaría en dos desigualdades: h(x)<=0 y -h(x)<=0. Así, seguiría 
# habiendo solo desigualdades y podría resolverse con Zoutendijk sin modificar ninguna función.
# Se añadirían las dos a g_funcs y sus gradientes a grad_g_funcs y se ejecutaría el código que ya 
# tenemos sin problema llegando a la solución.
# (b) Las dos desigualdades estarán activas a la vez en cada iteración, y aparecerán simultáneamente
#  en el conjunto I en cada iteración (si se está sobre la igualdad). El algoritmo incluirá los 
# dos gradientes ∇h(x) y -∇h(x). La dirección debe cumplir h(x)^T*d<=0 y -∇h(x)^T*d<=0, es decir,
# ∇h(x)^T*d=0, y la dirección será estrictamente tangente a la superficie de la igualdad.
# (c) Si h(x^{(k)}) = 0, estamos sobre la igualdad. Esto forzará a que la dirección d sea ortogonal
# al gradiente de la restricción (es decir, d será tangente a la superficie de la igualdad) porque
# debe cumplir: ∇h(x)^T*d<=0 y -∇h(x)^T*d<=0, esto es, ∇h(x)^T*d=0.
# El algoritmo actualiza el nuevo punto mediante x^(k+1)=x^(k)+ lambda*d.
# Si la restricción h(x)=0 define una superficie no lineal (tiene curvatura), cualquier movimiento 
# línea hace que el nuevo punto se salga de la superficie factible.
# Debido a esto, la función compute_lamda_max detectará que incluso para valores muy pequeños de lambda
# se dejaría de cumplir alguna restricción, luego devolvería lambda_max=0.
# Esto hace que el algoritmo se "bloquee", ya que no puede seguir sin salirse de la región
# factible. Geométricamente, el método no puede seguir la curvatura de la superficie de la igualdad.
# En cambio, si la restricción h(x) es lineal compute_lambda_max no estaría limitado, lambda_max puede
# ser mayor que cero y estaría determinado por las demás restricciones.