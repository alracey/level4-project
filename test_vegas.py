import vegas
import math

#example from VEGAS documentation tutorial#
def f(x):
    dx2 = 0
    for d in range(4):
        dx2 += (x[d] - 0.5)**2
    return math.exp(-dx2 * 100) * 1013.2118364296088

integ = vegas.Integrator([[-1, 1], [0, 1], [0, 1], [0, 1]])

result = integ(f, nitn=10, neval=1000)

print(result.summary())