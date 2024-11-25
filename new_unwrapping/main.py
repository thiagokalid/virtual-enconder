import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
from numpy import pi
import time

from src.visual_odometer.displacement_estimators.svd import phase_unwrapping
matplotlib.use('QtAgg')

def G(x, k):
    N = len(x)
    y = np.zeros(N)
    for n in range(N-2):
        y[n] = D(x, k, n+1) - D(x, k, n)
    return y

def D(x, k, n):
    return (x[n+1] + 2*pi*k[n+1]) - (x[n] + 2*pi*k[n])

def costfun(k):
    N = len(k)
    s = 0
    for n in range(N-2):
        s += 4*pi**2 * (k[n+2]**2 - 4*k[n+2]*k[n+1] + 4*k[n+2]*k[n] + 4*k[n+1]**2 - 4*k[n+1]*k[n] + k[n]**2)
    return s

def build_k_matrix(x):
    N = len(x)
    K = scipy.sparse.csc_array((N, N))
    for n in range(N-2):
        K[n, n] = -2 * pi  # k[n]
        K[n, n+1] = 4 * pi  # k[n+1]
        K[n, n+2] = -2 * pi  # k[n+2]
    return K

def build_c_vector(x):
    N = len(x)
    C = np.zeros(shape=(N, 1))
    for n in range(N - 2):
        C[n] = x[n] - 2*x[n+1] + x[n+2]
    return C
def milp_problem(signal: np.ndarray):
    N = len(signal)

    # Supondo que o vetor c = [k1, k2, ..., kN, a1, a2, ..., aN, b1, b2, ..., bN]
    c = np.ones(3*N)
    c[:N] = 0 # kn não está na função custo

    #
    eye_matrix = scipy.sparse.eye_array(N, N, format='csc')
    zero_matrix = scipy.sparse.csc_array((N, N))

    # b_l é 2N x 1
    C = build_c_vector(signal)
    lb = np.vstack([
        C,
        -C
    ])

    k_matrix = build_k_matrix(signal)

    A = scipy.sparse.vstack([
        scipy.sparse.hstack((k_matrix, eye_matrix, zero_matrix)),
        scipy.sparse.hstack((-k_matrix, zero_matrix, eye_matrix))
    ])

    integrality = np.zeros(3*N, dtype=int)
    integrality[:N] = 1

    constraints = scipy.optimize.LinearConstraint(A, lb[:, 0], ub=10)
    result = scipy.optimize.milp(c=c, constraints=constraints, integrality=integrality)
    return result.x[:N]


min_phase = -6*np.pi
max_phase = 6*np.pi
step_phase = 1 * np.pi/180
phase_ref = np.arange(min_phase, max_phase+step_phase, step_phase)
phases = (phase_ref + np.pi) % (2 * np.pi) - np.pi

k0 = np.zeros_like(phase_ref)

Gout = G(phases, k0)

t0 = time.time()
k = milp_problem(phases)
delta_t = time.time() - t0

t0 = time.time()
unwrapped = phase_unwrapping(phases)
delta_t2 = time.time() - t0

plt.subplot(1, 4, 1)
plt.title("$x[n]$")
plt.plot(phases, "o:", label="x[n]")
plt.xlabel(r"$1 \leq n \leq N$")
plt.legend()

plt.subplot(1, 4, 2)
plt.title(r"$y[n]=(x[n+2]-x[n+1]) - (x[n+1]-x[n])$")
plt.plot(Gout, 'o:', label="y[n]")
plt.plot(np.ones_like(phases) * 2*pi, 'k:', label=r"$\pm 2\pi$")
plt.plot(-np.ones_like(phases) * 2*pi, ':k')
plt.xlabel(r"$1 \leq n \leq N-2$")
plt.legend()

plt.subplot(1, 4, 3)
plt.title(r"$x_{unwrapped}[n]=x[n]+2 \pi k[n]$")
plt.plot(phases + k * 2 * pi, "o", label=r"$x_{unwrapped}[n]$")
plt.xlabel(r"$1 \leq n \leq N$")
plt.legend()

plt.subplot(1, 4, 4)
plt.title(r"$k[n]$")
plt.plot(k, 'rs')
plt.xlabel(r"$1 \leq n \leq N$")

plt.suptitle(f"Elapsed time for unwrapping: {delta_t:.2f} s")
plt.show()