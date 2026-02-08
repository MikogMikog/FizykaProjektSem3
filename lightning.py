from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# -----------------------------
# Konfiguracja
# -----------------------------
@dataclass
class SimConfig:
    H: int = 160
    W: int = 240
    scenario: int = 1
    seed: int = 1

    # wzrost kanału
    max_steps: int = 2500
    eta: float = 1.5
    eps_E: float = 1e-6

    # Laplace
    relax_iters: int = 60
    omega: float = 1.85  # SOR

    # render
    out_dir: str = "output"
    save_every: int = 15
    save_images: bool = True


# -----------------------------
# Mapy scenariuszy
# -----------------------------
def make_k_map(H: int, W: int, scenario: int) -> np.ndarray:
    """
    k(x,y) –  latwość wzrostu kanału.
    Skala względna: powietrze ~1.0, bariera <<1, kanał preferowany >1.
    """
    k = np.ones((H, W), dtype=np.float32)

    if scenario == 1:
        # Jednorodne: powietrze
        pass

    elif scenario == 2:
        # Bariera izolacyjna (pozioma płyta z otworem)
        # Powietrze: 1.0, bariera: 0.00
        k[:, :] = 1.0
        y0 = H // 2
        thickness = max(2, H // 80)
        k[y0 - thickness: y0 + thickness, :] = 0.00

        # utworzenie szpary
        gap_w = max(10, W // 12)
        gap_x0 = (W // 2) - (gap_w // 2)
        k[y0 - thickness: y0 + thickness, gap_x0: gap_x0 + gap_w] = 0.6

    elif scenario == 3:
        # Kanał preferowany (ukośny pas o większym k)
        # Powietrze: 1.0, pas: 2.5
        k[:, :] = 1.0
        band = max(6, W // 30)
        for y in range(H):
            # linia ukośna:
            xc = int(W * 0.2 + (y / max(1, H - 1)) * W * 0.6)
            x0 = max(0, xc - band)
            x1 = min(W, xc + band)
            k[y, x0:x1] = 2.5

        # dodanie malej szpary dla kontrastu
        y1 = int(H * 0.55)
        x1 = int(W * 0.62)
        k[max(0, y1 - 6): min(H, y1 + 6), max(0, x1 - 8): min(W, x1 + 8)] = 0.7

    else:
        raise ValueError("scenario must be 1, 2 or 3")

    return k


def scenario_default_eta(scenario: int) -> float:
    # domyślne eta
    if scenario == 1:
        return 1.5
    if scenario == 2:
        return 1.8
    if scenario == 3:
        return 1.4
    return 1.5


# -----------------------------
# Numeryka: Laplace (red-black SOR)
# -----------------------------
def solve_laplace_redblack_sor(
        phi: np.ndarray,
        fixed: np.ndarray,
        omega: float,
        iters: int,
) -> None:

    H, W = phi.shape

    # maski parzystości (red/black)
    yy, xx = np.indices((H, W))
    red = ((yy + xx) & 1) == 0
    black = ~red

    # wewnetrzny obszar
    inner = np.zeros_like(fixed, dtype=bool)
    inner[1:-1, 1:-1] = True

    upd_red = inner & (~fixed) & red
    upd_black = inner & (~fixed) & black

    for _ in range(iters):
        nbr = (
                      phi[:-2, 1:-1] + phi[2:, 1:-1] + phi[1:-1, :-2] + phi[1:-1, 2:]
              ) * 0.25

        m = upd_red[1:-1, 1:-1]
        if m.any():
            cur = phi[1:-1, 1:-1]
            cur[m] = (1.0 - omega) * cur[m] + omega * nbr[m]

        nbr = (
                      phi[:-2, 1:-1] + phi[2:, 1:-1] + phi[1:-1, :-2] + phi[1:-1, 2:]
              ) * 0.25
        m = upd_black[1:-1, 1:-1]
        if m.any():
            cur = phi[1:-1, 1:-1]
            cur[m] = (1.0 - omega) * cur[m] + omega * nbr[m]


def grad_E_mag(phi: np.ndarray) -> np.ndarray:
    E = np.zeros_like(phi, dtype=np.float32)
    # dphi/dx, dphi/dy
    dx = 0.5 * (phi[:, 2:] - phi[:, :-2])
    dy = 0.5 * (phi[2:, :] - phi[:-2, :])

    Emid = np.sqrt(dx[1:-1, :] ** 2 + dy[:, 1:-1] ** 2).astype(np.float32)
    E[1:-1, 1:-1] = Emid
    return E


# -----------------------------
# Front kanalu
# -----------------------------
def frontier_8(channel: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    """
    Front = puste komórki sąsiadujące z kanałem, z wykluczeniem blocked i samego kanału
    """
    ch = channel
    # 8 przesuniec
    neigh = (
            np.roll(ch, 1, 0)
            | np.roll(ch, -1, 0)
            | np.roll(ch, 1, 1)
            | np.roll(ch, -1, 1)
            | np.roll(np.roll(ch, 1, 0), 1, 1)
            | np.roll(np.roll(ch, 1, 0), -1, 1)
            | np.roll(np.roll(ch, -1, 0), 1, 1)
            | np.roll(np.roll(ch, -1, 0), -1, 1)
    )
    front = neigh & (~ch) & (~blocked)

    # wyciecie brzegow
    front[0, :] = False
    front[-1, :] = False
    front[:, 0] = False
    front[:, -1] = False
    return front


# -----------------------------
# Render
# -----------------------------
def clear_output_dir(path: str) -> None:
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if os.path.isfile(p):
            os.remove(p)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def render_frame(
        k_map: np.ndarray,
        channel: np.ndarray,
        title: str,
        path: str,
) -> None:
    if plt is None:
        return

    # normalizacja tla
    k = k_map.astype(np.float32)
    kmin, kmax = float(k.min()), float(k.max())
    if kmax - kmin < 1e-9:
        bg = np.zeros_like(k)
    else:
        bg = (k - kmin) / (kmax - kmin)

    # ciemne tlo z k_map
    img = 0.15 + 0.35 * bg   # zakres ~[0.15, 0.5]

    # biały piorun
    img[channel] = 1.0

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(img, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title, color="black")
    ax.axis("off")

    # biala ramka
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("white")
        spine.set_linewidth(2)

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(path, dpi=140, facecolor="white", bbox_inches="tight")
    plt.close()


# -----------------------------
# Symulacja
# -----------------------------
def simulate(cfg: SimConfig) -> dict:
    if cfg.seed is None:
        # seed max 20 znaków
        cfg.seed = int(np.random.SeedSequence().entropy) % (10 ** 20)
    rng = np.random.default_rng(cfg.seed)

    H, W = cfg.H, cfg.W
    k_map = make_k_map(H, W, cfg.scenario)

    blocked = k_map <= 0.02

    # Potencjal
    phi = np.zeros((H, W), dtype=np.float32)

    # Warunki brzegowe: gora 1, doł 0
    phi[0, :] = 1.0
    phi[-1, :] = 0.0

    fixed = np.zeros((H, W), dtype=bool)
    fixed[0, :] = True
    fixed[-1, :] = True
    fixed[:, 0] = True
    fixed[:, -1] = True

    # Kanał startuje blisko gory i leci w dol
    channel = np.zeros((H, W), dtype=bool)
    sy = 1
    sx = W // 2
    channel[sy, sx] = True

    phi[channel] = 1.0
    fixed[channel] = True

    ensure_dir(cfg.out_dir)
    clear_output_dir(cfg.out_dir)

    # Render poczatkowy
    if cfg.save_images and cfg.save_every > 0:
        render_frame(
            k_map, channel,
            title=f"Scenario {cfg.scenario} | seed={cfg.seed} | step 0",
            path=os.path.join(cfg.out_dir, f"sc{cfg.scenario}_00000.png"),
        )

    reached = False
    last_step = 0

    for step in range(1, cfg.max_steps + 1):
        # Rozwiazanie Laplace'a w obecnosci kanalu i brzegow
        solve_laplace_redblack_sor(phi, fixed, cfg.omega, cfg.relax_iters)

        # Pole
        Emag = grad_E_mag(phi)

        # Front
        front = frontier_8(channel, blocked)

        if step % 50 == 0:
            ys = np.where(channel)[0]
            depth = ys.max() / (H - 1)
            print(
                f"seed={cfg.seed} | step={step:4d} | frontier={int(front.sum()):4d} | depth={depth:5.1%}",
                end="\r"
            )

        if not front.any():
            last_step = step
            break

        # Wagi wzrostu
        w = (Emag + cfg.eps_E) ** cfg.eta
        w *= k_map
        w *= front.astype(np.float32)

        s = float(w.sum())
        if s <= 0.0 or not np.isfinite(s):
            last_step = step
            break

        # Losowanie komórki z rozkładu P ~ w
        idx = np.flatnonzero(w.ravel() > 0)
        probs = w.ravel()[idx].astype(np.float64)
        probs /= probs.sum()

        choice = rng.choice(idx, p=probs)
        y = choice // W
        x = choice - y * W

        # Dodanie do kanału
        channel[y, x] = True
        phi[y, x] = 1.0
        fixed[y, x] = True

        # Stop po dotknieciu dolnej krawedzi
        if y >= H - 2:
            reached = True
            last_step = step
            break

        # Zapis klatek co N kroków
        if cfg.save_images and cfg.save_every > 0 and (step % cfg.save_every == 0):
            render_frame(
                k_map, channel,
                title=f"Scenario {cfg.scenario} | seed={cfg.seed} | step {step}",
                path=os.path.join(cfg.out_dir, f"sc{cfg.scenario}_{step:05d}.png"),
            )

        last_step = step

    # Final
    if cfg.save_images:
        render_frame(
            k_map, channel,
            title=f"Scenario {cfg.scenario} | seed={cfg.seed} | final step {last_step} | reached={reached}",
            path=os.path.join(cfg.out_dir, f"sc{cfg.scenario}_FINAL.png"),
        )

    return {
        "reached_bottom": reached,
        "steps": last_step,
        "cfg": cfg,
    }


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> SimConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--H", type=int, default=160)
    p.add_argument("--W", type=int, default=240)

    p.add_argument("--max_steps", type=int, default=2500)
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--relax_iters", type=int, default=60)
    p.add_argument("--omega", type=float, default=1.85)

    p.add_argument("--out_dir", type=str, default="output")
    p.add_argument("--save_every", type=int, default=15)
    p.add_argument("--no_images", action="store_true")

    args = p.parse_args()

    eta = args.eta if args.eta is not None else scenario_default_eta(args.scenario)

    return SimConfig(
        H=args.H,
        W=args.W,
        scenario=args.scenario,
        seed=args.seed,
        max_steps=args.max_steps,
        eta=eta,
        relax_iters=args.relax_iters,
        omega=args.omega,
        out_dir=args.out_dir,
        save_every=args.save_every,
        save_images=(not args.no_images),
    )


def main():
    cfg = parse_args()
    res = simulate(cfg)
    print(
        f"[OK] scenario={cfg.scenario} steps={res['steps']} reached_bottom={res['reached_bottom']} "
        f"eta={cfg.eta} relax_iters={cfg.relax_iters} omega={cfg.omega} "
        f"out_dir={cfg.out_dir}"
    )
    if plt is None and cfg.save_images:
        print("[INFO] matplotlib not available -> images were not saved.")


if __name__ == "__main__":
    main()
