from method import *
import numpy as np
import matplotlib.pyplot as plt


def generate_signal(t, freqs, amps):
    y = np.zeros_like(t)
    for f, a in zip(freqs, amps):
        y += a * np.cos(2 * np.pi * f * t)
    y = y / y.max()
    return y


if __name__ == '__main__':
    #
    lbd = 809/3125
    beta_g = 20 * lbd
    T = 11/200

    # Generate time-grid:
    tstep = T
    t = np.arange(-10, 10 + tstep, tstep)

    # Generate signal to be wrapped:
    freqs = [0, 0.09, .03, .07, .11, .23, .5]
    np.random.seed(0)
    amplitudes = np.random.rand(len(freqs))

    # Original:
    g = generate_signal(t, freqs, amplitudes)

    # Wrapped version:
    y = wrap(g, lbd)


    # N = 2
    # plt.subplot(3, 3, 1)
    # plt.plot(t, g)
    #
    # plt.subplot(3, 3, 2)
    # plt.plot(t, g, "k", alpha=.5)
    # plt.stem(t, y, 'k')
    #
    # plt.subplot(3, 3, 3)
    # plt.stem(t, g - wrap(g, lbd), 'r')
    #
    # plt.subplot(3, 3, 4)
    # plt.stem(t[:len(np.diff(g, N))], np.diff(g, N))
    # plt.plot(t, lbd * np.ones_like(t), color='k', alpha=.3)
    # plt.plot(t, -lbd * np.ones_like(t), color='k', alpha=.3)
    # plt.title(f"N={N}")
    #
    # plt.subplot(3, 3, 5)
    # plt.stem(t[:len(np.diff(y, N))], np.diff(y, N))
    # plt.plot(t, lbd * np.ones_like(t), color='k', alpha=.3)
    # plt.plot(t, -lbd * np.ones_like(t), color='k', alpha=.3)
    # plt.title(f"N={N}")
    #
    # plt.subplot(3, 3, 6)
    # plt.stem(t[:len(np.diff(g - y, N))], np.diff(g - y, N))
    # plt.plot(t, lbd * np.ones_like(t), color='k', alpha=.3)
    # plt.plot(t, -lbd * np.ones_like(t), color='k', alpha=.3)
    # plt.title(f"N={N}")
    #
    # plt.subplot(3, 3, 7)
    # plt.stem(t[:len(np.diff(g, N))], wrap(np.diff(g, N), lbd))
    # plt.title(f"N={N}")
    #
    # plt.subplot(3, 3, 8)
    # plt.stem(t[:len(np.diff(y, N))], wrap(np.diff(y, N), lbd))
    # plt.title(f"N={N}")
    #
    # plt.subplot(3, 3, 9)
    # plt.stem(t[:len(np.diff(g - y, N))], wrap(np.diff(g - y, N), lbd))
    # plt.title(f"N={N}")
    #
    #
    #
    # plt.figure()
    # plt.stem(t[:len(np.diff(g - y, N))], np.diff(g - y, N))
    # plt.stem(t[:len(np.diff(y - y, N))], wrap(np.diff(y, N), lbd) - np.diff(y, N), 'k')
    # plt.plot(t, lbd * np.ones_like(t), color='k', alpha=.3)
    # plt.plot(t, -lbd * np.ones_like(t), color='k', alpha=.3)
    # plt.title(f"N={N}")

    # Unwrapped via Unlimited Sampling:
    gamma, g_hat = unwrap_unlimited_sampling(y, fc=np.max(freqs), m=1, lbd=lbd, beta_g=beta_g, T=T)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, g, color="k", alpha=.6, label="Bandlimited Function")
    plt.plot(t, y, 'o', color='r', label="Modulo Samples")
    plt.plot(t[:len(gamma)], gamma, 'ok', label="Unlimted Sampling recovered")
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.ylim([-.7, 1.1])
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, g, color="k", alpha=.6, label="Bandlimited Function")
    plt.plot(t, y, 'o', color='r', label="Modulo Samples")
    plt.plot(t[:len(np.unwrap(g))], np.unwrap(g), 'om', label="np.unwrap(.)")
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.ylim([-.7, 1.1])
    plt.legend()

    plt.show()