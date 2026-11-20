# Optimización No Lineal con Restricciones: Método de Zoutendijk

Implementación desde cero en Python del **algoritmo de direcciones factibles de Zoutendijk** y comparativa de rendimiento y convergencia frente a los métodos clásicos de **Penalización Cuadrática** y **Barrera Logarítmica**.

## Características Técnicas

* **Búsqueda de paso máximo:** Combinación de búsqueda exponencial y método de bisección para calcular el mayor paso factible en cada iteración.
* **Cálculo de direcciones factibles:** Identificación dinámica de restricciones activas y resolución de subproblemas lineales de dirección de descenso.
* **Benchmarking:** Evaluación comparativa de trayectorias, precisión y velocidad de convergencia en problemas convexos, restricciones no lineales (disco) y óptimos interiores.

## Resultados y Gráficas Comparativas

[Ver gráficas de trayectorias y convergencia (PDF)](resultados_lab.pdf)

## Tecnologías
* **Python 3**
* **NumPy**, **SciPy** & **Matplotlib**

## Cómo ejecutarlo
```bash
pip install numpy scipy matplotlib
python test_env.py
