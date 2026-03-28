
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


# ------------------------------------------------------------
# Tiny simulation of NN_Subspace_Alignment
# X = reference modality (e.g. text)
# Y = source modality to be aligned (e.g. image)
# Following the paper:
#   Xc = X - mu_X
#   Yc = Y - mu_Y
#   Phi* = W_Y^T W_X
#   Y_al = Yc W_Y Phi* W_X^T + mu_X
#   y_hat = y_al / ||y_al||
# ------------------------------------------------------------


OUTPUT_DIR = Path(__file__).resolve().parent / "method_figure"


def normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def orthonormal_basis(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    q1 = v1 / v1.norm()
    v2 = v2 - (v2 @ q1) * q1
    q2 = v2 / v2.norm()
    return torch.stack([q1, q2], dim=1)  # [3, 2]


def build_plane(center: torch.Tensor, basis: torch.Tensor, radius: float = 0.8, res: int = 6):
    u = torch.linspace(-radius, radius, res)
    v = torch.linspace(-radius, radius, res)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    pts = center[None, None, :] + uu[..., None] * basis[:, 0] + vv[..., None] * basis[:, 1]
    return pts[..., 0].cpu().numpy(), pts[..., 1].cpu().numpy(), pts[..., 2].cpu().numpy()


def simulate_modalities(n: int = 150, noise: float = 0.06, seed: int = 7, num_clusters: int = 1):
    torch.manual_seed(seed)

    if num_clusters < 1:
        raise ValueError("num_clusters must be >= 1")

    if num_clusters == 1:
        # One compact latent cloud so each modality appears as a single group.
        z = 0.28 * torch.randn(n, 2)
        labels = torch.zeros(n, dtype=torch.long)
    else:
        angles = torch.linspace(0, 2 * torch.pi, steps=num_clusters + 1)[:-1]
        cluster_centers = 0.55 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        labels = torch.randint(0, num_clusters, (n,))
        z = cluster_centers[labels] + 0.16 * torch.randn(n, 2)

    # Reference modality subspace
    W_x_true = orthonormal_basis(
        torch.tensor([1.0, 0.0, 0.15]),
        torch.tensor([0.1, 1.0, 0.2]),
    )

    # Source modality subspace: rotated / tilted
    W_y_true = orthonormal_basis(
        torch.tensor([0.65, 0.72, 0.25]),
        torch.tensor([-0.55, 0.25, 0.80]),
    )

    orth_x = torch.linalg.cross(W_x_true[:, 0], W_x_true[:, 1])
    orth_y = torch.linalg.cross(W_y_true[:, 0], W_y_true[:, 1])

    # Same latent cloud, different subspace geometry
    X = z @ W_x_true.T + noise * torch.randn(n, 1) * orth_x
    Y = 0.95 * z @ W_y_true.T + noise * torch.randn(n, 1) * orth_y

    # Add separated means so the two modalities occupy distinct regions
    # of the sphere while remaining internally compact.
    mu_X = torch.tensor([0.90, 0.22, 0.10])
    mu_Y = torch.tensor([-0.42, 0.78, 0.46])

    X = normalize_rows(X + mu_X)
    Y = normalize_rows(Y + mu_Y)
    return X, Y, labels


def nn_subspace_alignment(X: torch.Tensor, Y: torch.Tensor, dsub: int = 2):
    mu_X = X.mean(dim=0, keepdim=True)
    mu_Y = Y.mean(dim=0, keepdim=True)

    Xc = X - mu_X
    Yc = Y - mu_Y

    # SVD on centered embeddings
    _, _, VhX = torch.linalg.svd(Xc, full_matrices=False)
    _, _, VhY = torch.linalg.svd(Yc, full_matrices=False)
    VX = VhX.T
    VY = VhY.T

    WX = VX[:, :dsub]
    WY = VY[:, :dsub]

    # Closed-form subspace alignment
    Phi = WY.T @ WX

    # Rotate source into reference subspace, then reconstruct in ambient ref frame
    Yrot = Yc @ WY @ Phi @ WX.T
    Yal = Yrot + mu_X

    # Final L2 normalization
    Xhat = normalize_rows(X)
    Yhat = normalize_rows(Yal)

    return {
        "mu_X": mu_X.squeeze(0),
        "mu_Y": mu_Y.squeeze(0),
        "Xc": Xc,
        "Yc": Yc,
        "WX": WX,
        "WY": WY,
        "Phi": Phi,
        "Yrot": Yrot,
        "Yal": Yal,
        "Xhat": Xhat,
        "Yhat": Yhat,
    }


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def set_equal_3d(ax, lim=1.2):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def draw_sphere(ax, radius=1.0, alpha=0.06):
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = radius * np.outer(np.cos(u), np.sin(v))
    ys = radius * np.outer(np.sin(u), np.sin(v))
    zs = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, rstride=4, cstride=4, linewidth=0.5, alpha=alpha, color="gray")


def draw_plane(ax, center, basis, radius=0.6, color="C0", alpha=0.15):
    px, py, pz = build_plane(center, basis, radius=radius, res=5)
    ax.plot_surface(px, py, pz, color=color, alpha=alpha, shade=False, linewidth=0)


def draw_basis(ax, center, basis, color="C0", label=None, scale=0.6):
    for k in range(basis.shape[1]):
        vec = basis[:, k]
        arr = Arrow3D(
            [center[0], center[0] + scale * vec[0]],
            [center[1], center[1] + scale * vec[1]],
            [center[2], center[2] + scale * vec[2]],
            mutation_scale=12,
            lw=2,
            arrowstyle="-|>",
            color=color,
            alpha=0.9,
        )
        ax.add_artist(arr)

    if label is not None:
        pt = center + scale * 1.1 * basis[:, 0]
        ax.text(float(pt[0]), float(pt[1]), float(pt[2]), label, color=color)


def resolve_save_path(save_path="nn_subspace_alignment_panels.png") -> Path:
    path = Path(save_path)
    if not path.is_absolute():
        path = OUTPUT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def add_panel_caption(ax, text: str, y: float = -0.07):
    ax.text2D(0.5, y, text, transform=ax.transAxes, ha="center", va="top")


def make_panel_figure(X, Y, out, save_path="nn_subspace_alignment_panels.png"):
    fig = plt.figure(figsize=(12.6, 8.4))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.015,
        right=0.995,
        bottom=0.165,
        top=0.955,
        wspace=-0.16,
        hspace=0.02,
    )
    axes = [fig.add_subplot(grid[i, j], projection="3d") for i in range(2) for j in range(3)]

    # 1. Original embeddings
    ax = axes[0]
    draw_sphere(ax)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=12, alpha=0.65, color="tab:blue")
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=12, alpha=0.65, color="tab:red")
    ax.scatter(*out["mu_X"], s=60, color="tab:blue", marker="X")
    ax.scatter(*out["mu_Y"], s=60, color="tab:red", marker="X")
    ax.add_artist(
        Arrow3D(
            [out["mu_Y"][0], out["mu_X"][0]],
            [out["mu_Y"][1], out["mu_X"][1]],
            [out["mu_Y"][2], out["mu_X"][2]],
            mutation_scale=14,
            lw=1.8,
            arrowstyle="-|>",
            color="black",
            alpha=0.8,
        )
    )
    ax.text(float(out["mu_X"][0]), float(out["mu_X"][1]), float(out["mu_X"][2]), r"$\mu_X$", color="tab:blue")
    ax.text(float(out["mu_Y"][0]), float(out["mu_Y"][1]), float(out["mu_Y"][2]), r"$\mu_Y$", color="tab:red")
    ax.set_title(r"$\mathrm{(a)\ Original\ embeddings}$" + "\n" + r"$\mathrm{modality\ gap} + \mathrm{tilted\ supports}$", pad=6)

    # 2. Centering
    ax = axes[1]
    ax.scatter(out["Xc"][:, 0], out["Xc"][:, 1], out["Xc"][:, 2], s=12, alpha=0.65, color="tab:blue")
    ax.scatter(out["Yc"][:, 0], out["Yc"][:, 1], out["Yc"][:, 2], s=12, alpha=0.65, color="tab:red")
    ax.scatter([0], [0], [0], s=50, color="black", marker="o")
    draw_basis(ax, torch.zeros(3), out["WX"], color="tab:blue", label=r"$W_X$")
    draw_basis(ax, torch.zeros(3), out["WY"], color="tab:red", label=r"$W_Y$")
    ax.set_title(r"$\mathrm{(b)\ Centering}$" + "\n" + r"$X_c = X - \mu_X,\; Y_c = Y - \mu_Y$", pad=6)

    # 3. Principal subspaces
    ax = axes[2]
    ax.scatter(out["Xc"][:, 0], out["Xc"][:, 1], out["Xc"][:, 2], s=12, alpha=0.5, color="tab:blue")
    ax.scatter(out["Yc"][:, 0], out["Yc"][:, 1], out["Yc"][:, 2], s=12, alpha=0.5, color="tab:red")
    draw_plane(ax, torch.zeros(3), out["WX"], radius=0.7, color="tab:blue", alpha=0.18)
    draw_plane(ax, torch.zeros(3), out["WY"], radius=0.7, color="tab:red", alpha=0.18)
    draw_basis(ax, torch.zeros(3), out["WX"], color="tab:blue", label=r"$W_X$")
    draw_basis(ax, torch.zeros(3), out["WY"], color="tab:red", label=r"$W_Y$")
    ax.set_title(r"$\mathrm{(c)\ Principal\ subspaces}$" + "\n" + r"$\mathrm{SVD} \;\rightarrow\; \mathrm{top}\, d_{\mathrm{sub}}\, \mathrm{directions}$", pad=6)

    # 4. Subspace alignment
    ax = axes[3]
    ax.scatter(out["Xc"][:, 0], out["Xc"][:, 1], out["Xc"][:, 2], s=12, alpha=0.25, color="tab:blue")
    ax.scatter(out["Yrot"][:, 0], out["Yrot"][:, 1], out["Yrot"][:, 2], s=12, alpha=0.65, color="tab:red")
    draw_plane(ax, torch.zeros(3), out["WX"], radius=0.7, color="tab:blue", alpha=0.18)
    draw_basis(ax, torch.zeros(3), out["WX"], color="tab:blue", label=r"$W_X$")
    add_panel_caption(
        ax,
        r"$\mathrm{(d)\ Subspace\ alignment}$" + "\n" + r"$\Phi^\ast = W_Y^\top W_X,\; Y_{\mathrm{rot}} = Y_c W_Y \Phi^\ast W_X^\top$",
    )

    # 5. Ambient reconstruction
    ax = axes[4]
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=12, alpha=0.35, color="tab:blue")
    ax.scatter(out["Yal"][:, 0], out["Yal"][:, 1], out["Yal"][:, 2], s=12, alpha=0.65, color="tab:red")
    draw_plane(ax, out["mu_X"], out["WX"], radius=0.55, color="tab:blue", alpha=0.14)
    ax.scatter(*out["mu_X"], s=60, color="tab:blue", marker="X")
    add_panel_caption(ax, r"$\mathrm{(e)\ Ambient\ reconstruction}$" + "\n" + r"$Y_{\mathrm{al}} = Y_{\mathrm{rot}} + \mu_X$")

    # 6. Final normalized state
    ax = axes[5]
    draw_sphere(ax)
    ax.scatter(out["Xhat"][:, 0], out["Xhat"][:, 1], out["Xhat"][:, 2], s=12, alpha=0.65, color="tab:blue")
    ax.scatter(out["Yhat"][:, 0], out["Yhat"][:, 1], out["Yhat"][:, 2], s=12, alpha=0.65, color="tab:red")
    muX_hat = normalize_rows(out["mu_X"].unsqueeze(0)).squeeze(0)
    muY_hat = normalize_rows(out["Yal"].mean(dim=0, keepdim=True)).squeeze(0)
    ax.scatter(*muX_hat, s=60, color="tab:blue", marker="X")
    ax.scatter(*muY_hat, s=60, color="tab:red", marker="X")
    add_panel_caption(ax, r"$\mathrm{(f)\ Final\ normalized\ space}$" + "\n" + r"$\hat{y}_{\mathrm{al}} = y_{\mathrm{al}} / \|y_{\mathrm{al}}\|_2$")

    for ax, lim in zip(axes, [1.15, 0.9, 0.9, 0.9, 1.15, 1.15]):
        set_equal_3d(ax, lim=lim)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=8, label=r"$\mathrm{Reference\ modality}\ X$"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red", markersize=8, label=r"$\mathrm{Source/aligned\ modality}\ Y$"),
        plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="black", markersize=8, label=r"$\mathrm{Centroid}$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.022))
    # fig.suptitle("Tiny simulation of NN_Subspace_Alignment", fontsize=16, y=0.98)
    fig.savefig(resolve_save_path(save_path), dpi=180, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    X, Y, labels = simulate_modalities(n=150, noise=0.04, seed=7, num_clusters=2)
    out = nn_subspace_alignment(X, Y, dsub=2)

    print("mu_X =", out["mu_X"])
    print("mu_Y =", out["mu_Y"])
    print("Phi* =\n", out["Phi"])

    make_panel_figure(X, Y, out, save_path="nn_subspace_alignment_panels.png")
    plt.show()
