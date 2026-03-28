from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from method_figure import (
    OUTPUT_DIR,
    draw_basis,
    draw_plane,
    draw_sphere,
    nn_subspace_alignment,
    normalize_rows,
    resolve_save_path,
    set_equal_3d,
    simulate_modalities,
)


ANIMATION_DIR = OUTPUT_DIR / "animation"


def lerp(a, b, t):
    return (1.0 - t) * a + t * b


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def as_numpy(x):
    return x.detach().cpu().numpy()


def stage_progress(frame, start, length):
    if frame < start:
        return 0.0
    if frame >= start + length:
        return 1.0
    return smoothstep((frame - start) / max(length - 1, 1))


def build_animation_frames(X, Y, out):
    Xc_shifted = out["Xc"] + out["mu_X"]
    Yc_shifted = out["Yc"] + out["mu_Y"]
    Yrot_shifted = out["Yrot"] + out["mu_X"]

    return {
        "X_orig": X,
        "Y_orig": Y,
        "X_centered": Xc_shifted,
        "Y_centered": Yc_shifted,
        "X_ref": X,
        "Y_rot": Yrot_shifted,
        "Y_al": out["Yal"],
        "X_hat": normalize_rows(X),
        "Y_hat": out["Yhat"],
    }


def draw_cloud(ax, points, color, alpha=0.72, size=18):
    pts = as_numpy(points)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, alpha=alpha, color=color, depthshade=False)


def make_animation(
    X,
    Y,
    out,
    save_path="nn_subspace_alignment_animation.gif",
    fps=20,
    stage_frames=32,
):
    states = build_animation_frames(X, Y, out)
    total_frames = 4 * stage_frames

    fig = plt.figure(figsize=(7.4, 7.4))
    ax = fig.add_subplot(111, projection="3d")

    titles = [
        r"$\mathrm{(1)\ Original\ modalities}$",
        r"$\mathrm{(2)\ Centering}$",
        r"$\mathrm{(3)\ Subspace\ alignment}$",
        r"$\mathrm{(4)\ Reconstruction\ and\ normalization}$",
    ]
    subtitles = [
        r"$X,\; Y$",
        r"$X - \mu_X,\; Y - \mu_Y$",
        r"$Y_{\mathrm{rot}} = Y_c W_Y \Phi^\ast W_X^\top$",
        r"$Y_{\mathrm{al}} = Y_{\mathrm{rot}} + \mu_X,\; \hat{y}_{\mathrm{al}} = y_{\mathrm{al}}/\|y_{\mathrm{al}}\|_2$",
    ]

    def update(frame):
        ax.cla()
        draw_sphere(ax, alpha=0.05)
        set_equal_3d(ax, lim=1.15)

        p1 = stage_progress(frame, 0 * stage_frames, stage_frames)
        p2 = stage_progress(frame, 1 * stage_frames, stage_frames)
        p3 = stage_progress(frame, 2 * stage_frames, stage_frames)
        p4 = stage_progress(frame, 3 * stage_frames, stage_frames)

        X_now = lerp(states["X_orig"], states["X_centered"], p1)
        Y_now = lerp(states["Y_orig"], states["Y_centered"], p1)
        X_now = lerp(X_now, states["X_ref"], p2)
        Y_now = lerp(Y_now, states["Y_rot"], p2)
        Y_now = lerp(Y_now, states["Y_al"], p3)
        X_now = lerp(X_now, states["X_hat"], p4)
        Y_now = lerp(Y_now, states["Y_hat"], p4)

        draw_cloud(ax, X_now, "tab:blue")
        draw_cloud(ax, Y_now, "tab:red")

        if frame < stage_frames:
            title = titles[0]
            subtitle = subtitles[0]
        elif frame < 2 * stage_frames:
            title = titles[1]
            subtitle = subtitles[1]
            draw_basis(ax, out["mu_X"], out["WX"], color="tab:blue", label=r"$W_X$", scale=0.45)
            draw_basis(ax, out["mu_Y"], out["WY"], color="tab:red", label=r"$W_Y$", scale=0.45)
        elif frame < 3 * stage_frames:
            title = titles[2]
            subtitle = subtitles[2]
            draw_plane(ax, out["mu_X"], out["WX"], radius=0.58, color="tab:blue", alpha=0.12)
            draw_basis(ax, out["mu_X"], out["WX"], color="tab:blue", label=r"$W_X$", scale=0.5)
        else:
            title = titles[3]
            subtitle = subtitles[3]
            draw_plane(ax, out["mu_X"], out["WX"], radius=0.58, color="tab:blue", alpha=0.10)

        mu_x = as_numpy(out["mu_X"])
        mu_y = as_numpy(out["mu_Y"])
        ax.scatter(*mu_x, s=60, color="tab:blue", marker="X", depthshade=False)
        ax.scatter(*mu_y, s=60, color="tab:red", marker="X", depthshade=False)
        ax.text2D(0.5, 0.97, title, transform=ax.transAxes, ha="center", va="top")
        ax.text2D(0.5, 0.92, subtitle, transform=ax.transAxes, ha="center", va="top")
        ax.text2D(
            0.5,
            0.04,
            r"$\mathrm{Blue}: X \qquad \mathrm{Red}: Y$",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
        )
        return []

    anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps, blit=False)
    output_path = resolve_save_path(Path("animation") / save_path)
    anim.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    X, Y, labels = simulate_modalities(n=150, noise=0.04, seed=7, num_clusters=2)
    out = nn_subspace_alignment(X, Y, dsub=2)
    save_path = make_animation(X, Y, out)
    print(f"Saved animation to {save_path}")
