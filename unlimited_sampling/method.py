import numpy as np
import matplotlib.pyplot as plt
import audio2numpy as a2n


def wrap(x, lbd=np.pi):
    return (x + lbd) % (2 * lbd) - lbd


def S(x):
    y = np.zeros_like(x)
    for k in range(0, len(x)):
        y[k] = np.sum(x[:k])
    return y


def interpolation(t, T, gamma):
    sinc = lambda t: np.sin(np.pi * t) / (np.pi * t)
    g = 0
    for k in range(0, len(gamma)):
        g += gamma[k] * sinc(t / (T - k))
    return g


def unwrap_unlimited_sampling(y, fc, m, T=None, lbd=None, beta_g=None):
    # Omega = 10
    # fc = Omega / 2 * np.pi
    Omega = fc * 2 * np.pi

    # if T is None:
    #     # print(f"T <= {1 / (2 * Omega * np.e)}")
    #     T = 1 / (2 * Omega * np.e) * 1 / 10
    #     # print(f"fs = {1 / T}")
    #
    # if lbd is None:
    #     print(f"fc = {fc}")
    #     lbd = np.pi
    #
    # if beta_g is None:
    #     beta_g = np.max(np.abs(y)) * 10

    Ncomp = np.floor((np.log10(lbd) - np.log10(beta_g)) / np.log10(T * Omega * np.e))
    N = int(np.max([Ncomp, 1]))


    #N = 2
    print(f"Ntheory = {Ncomp}; N = ", N)

    deltaN_yn = np.diff(y, N)
    deltaE = wrap(deltaN_yn, lbd) - deltaN_yn
    s0 = deltaE

    for n in range(0, (N - 2) + 1):
        s = S(s0)
        s = 2 * lbd * np.floor(np.ceil(s / lbd) / 2)

        J = int(6 * beta_g / lbd * 1/10)

        deltaN_yn = np.diff(y, n)
        deltaE_n = wrap(deltaN_yn, lbd) - deltaN_yn

        arg = S(S(deltaE_n))

        print(J)

        kn = np.floor(
            (arg[1] - arg[J]) / (12 * beta_g) + 1 / 2
        )
        print(f"kn = {kn}")

        s = s + 2 * lbd * kn

        s0 = s

    print("No iterations.")
    gamma = S(s0) + y[:len(S(s0))] + 2 * m * lbd

    g = lambda t: interpolation(t, T, gamma)

    return gamma, g
