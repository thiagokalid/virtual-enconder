from main import *
import audio2numpy as a2n

if __name__ == "__main__":
    ang, _ = a2n.audio_from_file("audio.mp3")
    ang = ang[14900:15900]
    ang = ang[::15]
    ang = 2 * (ang - ang.min()) / (ang.max() - ang.min()) - 1
    gain = 2*np.pi * (35/100)**(-1)
    ang *= gain
    plt.plot(ang)
    N = len(ang)
    print("N=", N)

    phases = (ang + np.pi) % (2 * np.pi) - np.pi

    k0 = np.zeros_like(phases)
    Gout = G(phases, k0)

    t0 = time.time()
    k = milp_problem(phases)
    unwrapped_milp = phases + k * 2 * pi
    delta_t = time.time() - t0

    t0 = time.time()
    unwrapped = phase_unwrapping(phases)
    delta_t2 = time.time() - t0

    plt.figure(figsize=(14, 8))
    plt.subplot(1, 4, 1)
    plt.title("$x[n]$")
    plt.plot(phases, "-r", linewidth=2, markersize=2, label="x[n]")
    plt.xlabel(r"$1 \leq n \leq N$")
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.title(r"$y[n]=D[n+1] - D[n]$")
    plt.plot(Gout, 'o:k', label="y[n]")
    plt.plot(np.ones_like(phases) * 2 * pi, 'k:', label=r"$\pm 2\pi$")
    plt.plot(-np.ones_like(phases) * 2 * pi, ':k')
    plt.xlabel(r"$1 \leq n \leq N-2$")
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.title(r"$x_{unwrapped}[n]=x[n]+2 \pi k[n]$")
    plt.plot(unwrapped_milp, "-ob", markersize=3, label=r"$x_{unwrapped}[n]$")
    plt.plot(ang, "-r", alpha=1, linewidth=2, label="Original")
    plt.plot(unwrapped, "-g", alpha=.5, linewidth=2, label=r"Itoh 1982 (virtual encoder)")
    plt.plot(np.unwrap(phases), "-m", linewidth=2, alpha=.5, label=r"np.unwrap(.)")
    plt.xlabel(r"$1 \leq n \leq N$")
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.title(r"$k[n]$")
    plt.plot(k, 'bs')
    plt.xlabel(r"$1 \leq n \leq N$")

    plt.suptitle(f"Elapsed time for unwrapping: {delta_t:.2f} s")
    plt.show()
    plt.tight_layout()