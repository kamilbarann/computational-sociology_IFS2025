import numpy as np
import matplotlib.pyplot as plt
import os


def simulate_one(p, K, L, T_max):
    """
    Jeden przebieg symulacji:
      - L×L agentów, każdy z K fragmentami
      - każdy fragment startuje z prawdopodobieństwem p
      - synchroniczny transfer tylko, gdy sąsiad ma dokładnie o 1 fragment więcej
    Zwraca:
      history: list length T_max z stanami wiedzy (L×L×K)
      events:  list length T_max-1 z listami zdarzeń (i, j, k, dir)
      f:       array (K, T_max) → f[k,t] = udział agentów z fragmentem k
      n:       array (K+1, T_max) → n[j,t] = udział agentów z dokładnie j fragmentami
    """
    knowledge = np.random.rand(L, L, K) < p
    history = [knowledge.copy()]
    events = []
    f = np.zeros((K, T_max))
    n = np.zeros((K + 1, T_max))

    for t in range(T_max):
        counts = knowledge.sum(axis=2)
        # oblicz f(k,t) i n(j,t)
        for k in range(K):
            f[k, t] = (knowledge[:, :, k].sum()) / (L * L)
        for j in range(K + 1):
            n[j, t] = np.sum(counts == j) / (L * L)

        # transfer, jeśli nie ostatni krok
        if t < T_max - 1:
            new = knowledge.copy()
            step_events = []
            for i in range(L):
                for j in range(L):
                    nu = counts[i, j]
                    if nu < K:
                        nbrs = [
                            ((i - 1) % L, j, "up"),
                            ((i + 1) % L, j, "down"),
                            (i, (j - 1) % L, "left"),
                            (i, (j + 1) % L, "right"),
                        ]
                        # tylko sąsiedzi z dokładnie nu+1 fragmentami
                        cands = [
                            (x, y, d) for (x, y, d) in nbrs if counts[x, y] == nu + 1
                        ]
                        if not cands:
                            continue
                        x, y, d = cands[np.random.randint(len(cands))]
                        missing = np.where((knowledge[x, y]) & (~knowledge[i, j]))[0]
                        if missing.size == 0:
                            continue
                        k_chosen = np.random.choice(missing)
                        new[i, j, k_chosen] = True
                        step_events.append((i, j, k_chosen, d))
            knowledge = new
            history.append(knowledge.copy())
            events.append(step_events)

    return history, events, f, n


def plot_f_n_example(f, n, p, out_dir, eps=1e-6):
    """Przykładowe wykresy f(k,t) i n(k,t) dla pojedynczej symulacji,
    przycinane tak, aby oba miały tę samą długość (ostatnia zmiana >eps)."""
    os.makedirs(out_dir, exist_ok=True)
    T_max = f.shape[1]

    # dynamiczne przycinanie: różnice w f
    df = np.abs(f[:, 1:] - f[:, :-1])  # shape (K, T_max-1)
    df_max = df.max(axis=0)  # max po k, shape (T_max-1,)
    last_f = (np.max(np.where(df_max > eps)) + 1) if np.any(df_max > eps) else 0

    # różnice w n
    dn = np.abs(n[:, 1:] - n[:, :-1])  # shape (K+1, T_max-1)
    dn_max = dn.max(axis=0)
    last_n = (np.max(np.where(dn_max > eps)) + 1) if np.any(dn_max > eps) else 0

    last = max(last_f, last_n)

    x = np.arange(last + 1)

    # f(k,t)
    plt.figure(figsize=(6, 4))
    for k in range(f.shape[0]):
        plt.plot(x, f[k, : last + 1], label=f"k={k}")
    plt.xlim(0, last)
    plt.title(f"f(k,t), p={p}")
    plt.xlabel("t")
    plt.ylabel("f(k)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/f_vs_t_p{int(p * 100)}.png", dpi=150)
    plt.close()

    # n(k,t)
    plt.figure(figsize=(6, 4))
    for j in range(n.shape[0]):
        plt.plot(x, n[j, : last + 1], label=f"k={j}")
    plt.xlim(0, last)
    plt.title(f"n(k,t), p={p}")
    plt.xlabel("t")
    plt.ylabel("n(k)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/n_vs_t_p{int(p * 100)}.png", dpi=150)
    plt.close()


def plot_transfer_example(history, events, K, L, p, out_dir):
    """Przykładowa wizualizacja transferu, bez zmian."""

    os.makedirs(out_dir, exist_ok=True)
    block, gap = 1.0, 0.3
    T_max = len(history)
    for t in range(1, T_max):
        if not events[t - 1]:
            continue
        width = L * (K * block + gap)
        height = L * (block + gap)
        fig, axes = plt.subplots(2, 1, figsize=(width / 2, height / 2))
        for ax, panel in zip(axes, [t, t - 1]):
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.axis("off")
            for i in range(L):
                for j in range(L):
                    x0, y0 = j * (K * block + gap), i * (block + gap)
                    ax.add_patch(
                        plt.Rectangle(
                            (x0, y0),
                            K * block,
                            block,
                            facecolor="none",
                            edgecolor="gray",
                        )
                    )
                    state = history[panel][i, j]
                    for k in range(K):
                        ax.add_patch(
                            plt.Rectangle(
                                (x0 + k * block, y0),
                                block,
                                block,
                                facecolor="black" if state[k] else "white",
                                edgecolor="none",
                            )
                        )
            if panel == t:
                unit = 0.3 * block
                for i_rec, j_rec, k, d in events[t - 1]:
                    if d == "up":
                        i_d, j_d, dx, dy = (i_rec - 1) % L, j_rec, 0, +unit
                    elif d == "down":
                        i_d, j_d, dx, dy = (i_rec + 1) % L, j_rec, 0, -unit
                    elif d == "left":
                        i_d, j_d, dx, dy = i_rec, (j_rec - 1) % L, +unit, 0
                    else:
                        i_d, j_d, dx, dy = i_rec, (j_rec + 1) % L, -unit, 0
                    x_c = j_d * (K * block + gap) + (k + 0.5) * block
                    y_c = i_d * (block + gap) + 0.5 * block
                    ax.arrow(
                        x_c,
                        y_c,
                        dx,
                        dy,
                        head_width=0.1,
                        head_length=0.1,
                        fc="green",
                        ec="green",
                        linewidth=2,
                    )
        plt.tight_layout()
        plt.savefig(f"{out_dir}/transfer_p{int(p * 100)}_step{t}.png", dpi=150)
        plt.close()


def main():
    L, K, T_max, R = 5, 4, 20, 50
    p_values = [0.9, 0.7, 0.5, 0.3]
    out = "output"
    os.makedirs(out, exist_ok=True)

    # słowniki na uśrednione metryki
    avg_full = {p: np.zeros(T_max) for p in p_values}
    avg_half = {p: np.zeros(T_max) for p in p_values}

    for p in p_values:
        for r in range(R):
            history, events, f, n = simulate_one(p, K, L, T_max)
            avg_full[p] += n[K]
            avg_half[p] += n[K // 2]
            if r == 0:
                # rysuj przykłady z pierwszej symulacji
                plot_f_n_example(f, n, p, out)
                plot_transfer_example(history, events, K, L, p, out)

        # uśrednij
        avg_full[p] /= R
        avg_half[p] /= R

        # dynamiczne przycinanie tak, by oba miały tę samą długość
        eps = 1e-6
        df = np.abs(avg_full[p][1:] - avg_full[p][:-1])
        dn = np.abs(avg_half[p][1:] - avg_half[p][:-1])
        last_f = (np.max(np.where(df > eps)) + 1) if np.any(df > eps) else 0
        last_n = (np.max(np.where(dn > eps)) + 1) if np.any(dn > eps) else 0
        last = max(last_f, last_n)

        x = np.arange(last + 1)

        # wykres uśredniony n(K)
        plt.figure(figsize=(8, 4))
        plt.plot(x, avg_full[p][: last + 1], lw=2)
        plt.xlim(0, last)
        plt.title(f"n(K), avg p={p}")
        plt.xlabel("t")
        plt.ylabel("n(K)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{out}/n_full_avg_p{int(p * 100)}.png", dpi=150)
        plt.close()

        # wykres uśredniony n(K/2)
        plt.figure(figsize=(8, 4))
        plt.plot(x, avg_half[p][: last + 1], lw=2)
        plt.xlim(0, last)
        plt.title(f"n(K/2), avg p={p}")
        plt.xlabel("t")
        plt.ylabel("n(K/2)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{out}/n_half_avg_p{int(p * 100)}.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
