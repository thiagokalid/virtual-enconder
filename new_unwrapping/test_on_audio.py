from main import *
import audio2numpy as a2n

from unlimited_sampling.method import unwrap_unlimited_sampling

if __name__ == "__main__":
    ang, _ = a2n.audio_from_file("audio.mp3")
    ang = ang[14900:17900]
    ang = ang[::15][:-30]
    ang = 2 * (ang - ang.min()) / (ang.max() - ang.min()) - 1
    gain = 1*np.pi * (35/100)**(-1)
    ang *= gain
    plt.plot(ang)
    N = len(ang)
    print("N=", N)

    phases = (ang + np.pi) % (2 * np.pi) - np.pi

    k0 = np.zeros_like(phases)
    Gout = G(phases, k0)

    t0 = time.time()
    #k = milp_problem(phases) - 1
    k = k0
    unwrapped_milp = phases + k * 2 * pi
    delta_t = time.time() - t0

    t0 = time.time()
    unwrapped = phase_unwrapping(phases)
    delta_t2 = time.time() - t0

    t0 = time.time()
    unwrapped_us = unwrap_unlimited_sampling(phases, fc=10e3, m=-2)

    plt.figure(figsize=(14, 8))
    plt.subplot(1, 3, 1)
    plt.title("$x[n]$")
    plt.plot(ang, "-k", linewidth=2, markersize=2, label="Original")
    plt.plot(phases, "-or", linewidth=2, markersize=4, label="Wrapped")
    plt.xlabel(r"$1 \leq n \leq N$")
    plt.legend()

    # plt.subplot(1, 5, 2)
    # plt.title(r"$y[n]=D[n+1] - D[n]$")
    # plt.plot(Gout, 'o:k', label="y[n]")
    # plt.plot(np.ones_like(phases) * 2 * pi, 'k:', label=r"$\pm 2\pi$")
    # plt.plot(-np.ones_like(phases) * 2 * pi, ':k')
    # plt.xlabel(r"$1 \leq n \leq N-2$")
    # plt.legend()

    plt.subplot(1, 3, 2)
    plt.title(r"$x_{unwrapped}[n]=x[n]+2 \pi k[n]$")
    plt.plot(ang, "-k", alpha=1, linewidth=2, label="Original")
    #plt.plot(unwrapped_milp, "-b", markersize=3, linewidth=3, label=r"$MILP$")
    plt.plot(unwrapped, "-g", alpha=.5, linewidth=2, label=r"Itoh")
    plt.plot(np.unwrap(phases), "-m", linewidth=2, alpha=.5, label=r"np.unwrap(.)")
    plt.plot(unwrapped_us, "-y", linewidth=2, alpha=1, label=r"Unlimited Sampling")
    plt.xlabel(r"$1 \leq n \leq N$")
    plt.legend()


    plt.subplot(1, 3, 3)
    plt.title("Residue")
    #plt.plot(ang[:len(unwrapped_milp)] - unwrapped_milp, "-b", linewidth=3)
    plt.plot(ang[:len(unwrapped)] - unwrapped, "-g", linewidth=2)
    plt.plot(ang[:len(np.unwrap(phases))] - np.unwrap(phases), "-m", linewidth=2)
    plt.plot(ang[:len(unwrapped_us)] - unwrapped_us, "-y", linewidth=2)

    plt.suptitle(f"Elapsed time for unwrapping: {delta_t:.2f} s")
    plt.tight_layout()

    plt.figure()
    plt.title(r"$k[n]$")
    plt.plot(k, 'bs')
    plt.xlabel(r"$1 \leq n \leq N$")
    plt.show()