import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Config
# ============================================================
FEW_PATH = Path("/mnt/media/emanuele/few_dimensions/comparison/results/few_dimensions_results.json")
METHOD_PATH = Path("/mnt/media/emanuele/few_dimensions/comparison/results/method_verification_results.json")
OUTPUT_DIR = Path("/mnt/media/emanuele/few_dimensions/figures_paper/parallel_line_plots")

CLUSTERING_METRIC = "V-measure"   # or "NMI", "ARI", "Homogeneity"
FIGSIZE = (12, 6)
LINEWIDTH_TRUNK = 2.0
LINEWIDTH_BRANCH = 2.0
LINEWIDTH_BASELINE = 2.3
ALPHA = 0.9
CURVE_STEPS = 50
CURVE_STRENGTH = 0.38

STYLE_ALIGNED = "-"
STYLE_ORIGINAL = (0, (5, 2))
STYLE_BASELINE = (0, (2, 2))

BASELINE_COLOR = "#2f2f2f"

DATASET_TITLES = {
    "flickr30k": "Flickr30k",
    "mscoco_imagenet": "MSCOCO + ImageNet labels",
    "msrvtt": "MSRVTT",
}


# ============================================================
# Helpers
# ============================================================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_main_model_key(data):
    keys = [k for k in data.keys() if k != "seed"]
    if not keys:
        raise ValueError("No model key found.")
    return keys[0]


def map_dataset_name_for_few(dataset_name):
    # In few_dimensions_results, msrvtt is stored as msrvtt_v2
    if dataset_name == "msrvtt":
        return "msrvtt_v2"
    return dataset_name


def safe_get_mv_retrieval_at_1(entry):
    """
    Prefer retrieval_aligned@1 if available, otherwise fallback to retrieval_orig@1.
    """
    retrieval_aligned = entry.get("retrieval_aligned")
    if retrieval_aligned is not None:
        return float(retrieval_aligned["1"])
    return float(entry["retrieval_orig"]["1"])


def safe_get_mv_clustering(entry, dataset_block, clustering_metric):
    """
    Prefer aligned clustering from method verification, then original.
    If both are missing, fallback to the generic/orthogonal CLIP baseline
    available in the same dataset block.
    """
    clustering_aligned = entry.get("clustering_aligned")
    if clustering_aligned is not None:
        return float(clustering_aligned[clustering_metric])

    clustering_orig = entry.get("clustering_orig")
    if clustering_orig is not None:
        return float(clustering_orig[clustering_metric])

    for method_name in ("generic_procrustes", "orthogonal_procrustes"):
        method_entry = dataset_block.get(method_name)
        if method_entry is not None and method_entry.get("clustering_orig") is not None:
            return float(method_entry["clustering_orig"][clustering_metric])

    return None


def pick_clip_baseline_entry(dataset_block):
    """
    Use the method-verification original CLIP metrics as the baseline.
    Per request, take them from subspace_alignment at d_sub=32 and use
    retrieval_orig / gaps_orig / clustering_orig.
    """
    subspace_block = dataset_block.get("subspace_alignment", {})
    if "32" in subspace_block:
        return subspace_block["32"]

    raise ValueError("No subspace_alignment['32'] baseline entry found for dataset block.")


def build_dataset_payloads(few_data, method_data, clustering_metric="V-measure"):
    few_model = get_main_model_key(few_data)
    method_model = list(method_data.keys())[0]

    payloads = {}

    for dataset_name, dataset_block in method_data[method_model].items():
        subspace_block = dataset_block["subspace_alignment"]
        few_dataset_name = map_dataset_name_for_few(dataset_name)

        records = []
        for d_sub_str, mv_entry in sorted(subspace_block.items(), key=lambda kv: int(kv[0])):
            if d_sub_str not in few_data[few_model]:
                continue
            if few_dataset_name not in few_data[few_model][d_sub_str]:
                continue

            fd_metrics = few_data[few_model][d_sub_str][few_dataset_name]["metrics"]

            records.append(
                {
                    "model": method_model,
                    "dataset": dataset_name,
                    "d_sub": int(d_sub_str),
                    "mv_clustering": safe_get_mv_clustering(mv_entry, dataset_block, clustering_metric),
                    "mv_retrieval1": safe_get_mv_retrieval_at_1(mv_entry),
                    "few_n_dims": int(fd_metrics["n_dims"]),
                    "aligned_clustering": float(fd_metrics["clustering_aligned"][clustering_metric]),
                    "aligned_retrieval1": float(fd_metrics["retrieval_aligned"]["1"]),
                    "original_clustering": float(fd_metrics["clustering_orig"][clustering_metric]),
                    "original_retrieval1": float(fd_metrics["retrieval_orig"]["1"]),
                }
            )

        if not records:
            continue

        baseline_entry = pick_clip_baseline_entry(dataset_block)
        baseline = {
            "model": method_model,
            "dataset": dataset_name,
            "method_clustering": safe_get_mv_clustering(baseline_entry, dataset_block, clustering_metric),
            "method_retrieval1": float(baseline_entry["retrieval_orig"]["1"]),
            "clustering": (
                safe_get_mv_clustering(baseline_entry, dataset_block, clustering_metric)
            ),
            "retrieval1": float(baseline_entry["retrieval_orig"]["1"]),
            "gaps_orig": baseline_entry.get("gaps_orig"),
        }

        payloads[dataset_name] = {
            "model": method_model.replace("__", "_"),
            "records": records,
            "baseline": baseline,
        }

    return payloads


def minmax(values):
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        return vmin, vmax + 1e-9
    return vmin, vmax


def normalize(value, vmin, vmax):
    return (value - vmin) / (vmax - vmin)


def make_axis_maps(payload):
    records = payload["records"]
    dsub_minmax = minmax([r["d_sub"] for r in records])
    ndims_minmax = minmax([r["few_n_dims"] for r in records])

    return {
        "model_y": 0.5,
        "dsub_minmax": dsub_minmax,
        "method_cluster_minmax": (0.0, 1.0),
        "method_ret_minmax": (0.0, 1.0),
        "ndims_minmax": ndims_minmax,
        "clustering_minmax": (0.0, 1.0),
        "retrieval_minmax": (0.0, 1.0),
    }


def get_record_y_positions(record, maps):
    return {
        "model": maps["model_y"],
        "d_sub": normalize(record["d_sub"], *maps["dsub_minmax"]),
        "mv_cluster": (
            None
            if record["mv_clustering"] is None
            else normalize(record["mv_clustering"], *maps["method_cluster_minmax"])
        ),
        "mv_ret": normalize(record["mv_retrieval1"], *maps["method_ret_minmax"]),
        "few_n_dims": normalize(record["few_n_dims"], *maps["ndims_minmax"]),
        "aligned_cluster": normalize(record["aligned_clustering"], *maps["clustering_minmax"]),
        "aligned_ret": normalize(record["aligned_retrieval1"], *maps["retrieval_minmax"]),
        "original_cluster": normalize(record["original_clustering"], *maps["clustering_minmax"]),
        "original_ret": normalize(record["original_retrieval1"], *maps["retrieval_minmax"]),
    }


def get_baseline_y_positions(baseline, maps):
    return {
        "model": maps["model_y"],
        "mv_cluster": (
            None
            if baseline["method_clustering"] is None
            else normalize(baseline["method_clustering"], *maps["method_cluster_minmax"])
        ),
        "mv_ret": normalize(baseline["method_retrieval1"], *maps["method_ret_minmax"]),
        "cluster": (
            None
            if baseline["clustering"] is None
            else normalize(baseline["clustering"], *maps["clustering_minmax"])
        ),
        "ret": normalize(baseline["retrieval1"], *maps["retrieval_minmax"]),
    }


def draw_vertical_axis(ax, x, label, tick_values=None, tick_labels=None):
    ax.plot([x, x], [0, 1], color="black", lw=1.0, zorder=1)
    ax.text(x, 1.015, label, ha="center", va="bottom", fontsize=12, fontweight="bold")

    if tick_values is not None and tick_labels is not None:
        for y, tick_label in zip(tick_values, tick_labels):
            ax.plot([x - 0.015, x + 0.015], [y, y], color="black", lw=0.8)
            ax.text(x - 0.03, y, tick_label, ha="right", va="center", fontsize=9)


def make_ticks(vmin, vmax, n=5, fmt="{:.2f}"):
    vals = np.linspace(vmin, vmax, n)
    ys = [normalize(v, vmin, vmax) for v in vals]
    labels = [fmt.format(v) for v in vals]
    return ys, labels


def bezier_segment(p0, p1, strength=CURVE_STRENGTH, steps=CURVE_STEPS):
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    cp1 = (x0 + strength * dx, y0)
    cp2 = (x1 - strength * dx, y1)

    t = np.linspace(0.0, 1.0, steps)
    omt = 1.0 - t

    xs = (
        (omt ** 3) * x0
        + 3 * (omt ** 2) * t * cp1[0]
        + 3 * omt * (t ** 2) * cp2[0]
        + (t ** 3) * x1
    )
    ys = (
        (omt ** 3) * y0
        + 3 * (omt ** 2) * t * cp1[1]
        + 3 * omt * (t ** 2) * cp2[1]
        + (t ** 3) * y1
    )
    return xs, ys


def plot_smooth_path(ax, points, **kwargs):
    all_x = []
    all_y = []
    for i in range(len(points) - 1):
        xs, ys = bezier_segment(points[i], points[i + 1])
        if i > 0:
            xs = xs[1:]
            ys = ys[1:]
        all_x.extend(xs)
        all_y.extend(ys)
    ax.plot(all_x, all_y, **kwargs)


def make_dsub_colors(records):
    d_subs = sorted({r["d_sub"] for r in records})
    cmap = plt.get_cmap("viridis")
    if len(d_subs) == 1:
        return {d_subs[0]: cmap(0.6)}

    color_positions = np.linspace(0.15, 0.9, len(d_subs))
    return {d_sub: cmap(pos) for d_sub, pos in zip(d_subs, color_positions)}


def dataset_output_name(dataset_name):
    return f"parallel_subspace_few_dimensions_{dataset_name}.png"


# ============================================================
# Plot
# ============================================================
def plot_dataset_parallel(dataset_name, payload, output_dir, clustering_metric="V-measure"):
    maps = make_axis_maps(payload)
    records = payload["records"]
    baseline = payload["baseline"]
    dsub_colors = make_dsub_colors(records)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    xs = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)

    draw_vertical_axis(ax, xs[0], "Model", [maps["model_y"]], [payload["model"]])
    draw_vertical_axis(ax, xs[1], r"$d_{sub}$", *make_ticks(*maps["dsub_minmax"], n=min(6, len(records)), fmt="{:.0f}"))
    draw_vertical_axis(ax, xs[2], f"Method\n{clustering_metric}", *make_ticks(0.0, 1.0, n=6, fmt="{:.1f}"))
    draw_vertical_axis(ax, xs[3], "Method\nRetrieval@1", *make_ticks(0.0, 1.0, n=6, fmt="{:.1f}"))
    draw_vertical_axis(ax, xs[4], "Few\nn_dims", *make_ticks(*maps["ndims_minmax"], n=min(6, len(records)), fmt="{:.0f}"))
    draw_vertical_axis(ax, xs[5], clustering_metric, *make_ticks(0.0, 1.0, n=6, fmt="{:.1f}"))
    draw_vertical_axis(ax, xs[6], "Retrieval@1", *make_ticks(0.0, 1.0, n=6, fmt="{:.1f}"))

    # few-dim records
    for record in records:
        y = get_record_y_positions(record, maps)
        color = dsub_colors[record["d_sub"]]

        trunk_points = [
            (xs[0], y["model"]),
            (xs[1], y["d_sub"]),
            (xs[2], y["mv_cluster"]),
            (xs[3], y["mv_ret"]),
            (xs[4], y["few_n_dims"]),
        ]
        plot_smooth_path(
            ax,
            trunk_points,
            color=color,
            lw=LINEWIDTH_TRUNK,
            alpha=ALPHA,
            linestyle="-",
            solid_capstyle="round",
            zorder=3,
        )

        aligned_points = [
            (xs[4], y["few_n_dims"]),
            (xs[5], y["aligned_cluster"]),
            (xs[6], y["aligned_ret"]),
        ]
        plot_smooth_path(
            ax,
            aligned_points,
            color=color,
            lw=LINEWIDTH_BRANCH,
            alpha=ALPHA,
            linestyle=STYLE_ALIGNED,
            zorder=4,
        )

        original_points = [
            (xs[4], y["few_n_dims"]),
            (xs[5], y["original_cluster"]),
            (xs[6], y["original_ret"]),
        ]
        plot_smooth_path(
            ax,
            original_points,
            color=color,
            lw=LINEWIDTH_BRANCH,
            alpha=0.8,
            linestyle=STYLE_ORIGINAL,
            zorder=4,
        )

        scatter_points = trunk_points + aligned_points[1:] + original_points[1:]
        unique_points = list(dict.fromkeys(scatter_points))
        ax.scatter(
            [p[0] for p in unique_points],
            [p[1] for p in unique_points],
            s=18,
            color=color,
            zorder=5,
        )

    # baseline: bypass d_sub and few_n_dims
    baseline_y = get_baseline_y_positions(baseline, maps)
    baseline_points = [(xs[0], baseline_y["model"])]
    if baseline_y["mv_cluster"] is not None:
        baseline_points.append((xs[2], baseline_y["mv_cluster"]))
    baseline_points.append((xs[3], baseline_y["mv_ret"]))
    if baseline_y["cluster"] is not None:
        baseline_points.append((xs[5], baseline_y["cluster"]))
    baseline_points.append((xs[6], baseline_y["ret"]))
    plot_smooth_path(
        ax,
        baseline_points,
        color=BASELINE_COLOR,
        lw=LINEWIDTH_BASELINE,
        alpha=0.95,
        linestyle=STYLE_BASELINE,
        zorder=6,
    )
    ax.scatter(
        [p[0] for p in baseline_points],
        [p[1] for p in baseline_points],
        s=26,
        color=BASELINE_COLOR,
        zorder=7,
    )

    # legends
    dsub_handles = []
    for d_sub in sorted(dsub_colors):
        handle, = ax.plot([], [], color=dsub_colors[d_sub], lw=2.4, label=rf"$d_{{sub}}={d_sub}$")
        dsub_handles.append(handle)

    aligned_handle, = ax.plot([], [], color="black", lw=2, linestyle=STYLE_ALIGNED, label="few-dim aligned")
    original_handle, = ax.plot([], [], color="black", lw=2, linestyle=STYLE_ORIGINAL, label="few-dim original")
    baseline_handle, = ax.plot([], [], color=BASELINE_COLOR, lw=2.2, linestyle=STYLE_BASELINE, label="CLIP pretrained baseline")

    legend1 = ax.legend(handles=dsub_handles, title=r"$d_{sub}$", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(legend1)
    ax.legend(
        handles=[aligned_handle, original_handle, baseline_handle],
        title="Branches",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.56),
    )

    ax.set_xlim(-0.3, 6.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{DATASET_TITLES.get(dataset_name, dataset_name)}: method verification to few-dim view",
        fontsize=15,
        fontweight="bold",
        pad=28,
    )
    ax.set_frame_on(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / dataset_output_name(dataset_name)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    few_data = load_json(FEW_PATH)
    method_data = load_json(METHOD_PATH)

    payloads = build_dataset_payloads(
        few_data=few_data,
        method_data=method_data,
        clustering_metric=CLUSTERING_METRIC,
    )

    print(f"Loaded {sum(len(p['records']) for p in payloads.values())} few-dim records across {len(payloads)} datasets.")
    for dataset_name, payload in payloads.items():
        output_path = plot_dataset_parallel(
            dataset_name=dataset_name,
            payload=payload,
            output_dir=OUTPUT_DIR,
            clustering_metric=CLUSTERING_METRIC,
        )
        print(f"Saved {output_path}")
