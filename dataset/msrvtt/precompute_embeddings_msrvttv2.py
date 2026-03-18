"""
Precompute embeddings for MSRVTT v2 — one entry per unique video_id.

Differences from v1:
- Deduplicates annotations by video_id (keeps first occurrence)
- Always uses the first caption (no text_mode option)
- No audio existence checks
- Output: text_emb (N, D), vision_emb (N, D), video_ids (N,)
"""

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath("../.."))

import argparse
import json
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import open_clip

from dataset.msrvtt.vision_mapper import VisionMapper


# =========================================================
# Utils
# =========================================================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def build_clip(device: torch.device, model_name: str, pretrained: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def _save_npz(
    out_path: str,
    text_emb: np.ndarray,
    vision_emb: np.ndarray,
    video_ids: np.ndarray,
    fns: List[str],
    meta: Dict[str, Any],
) -> None:
    np.savez_compressed(
        out_path,
        text_emb=text_emb.astype(np.float32),
        vision_emb=vision_emb.astype(np.float32),
        video_ids=video_ids,
        fns=np.array(fns, dtype=object),
        meta=np.array([meta], dtype=object),
    )


def _deduplicate_annotations(annos: list) -> list:
    """Keep only the first annotation per unique video_id."""
    seen = set()
    unique = []
    for a in annos:
        vid = a["video_id"]
        if vid not in seen:
            seen.add(vid)
            unique.append(a)
    return unique


def _get_caption(anno: dict) -> Optional[str]:
    """Return the first non-empty caption string, or None."""
    desc = anno.get("desc", anno.get("caption", ""))
    if isinstance(desc, list):
        for d in desc:
            d = str(d).strip()
            if d:
                return d
        return None
    desc = str(desc).strip()
    return desc if desc else None


# =========================================================
# Precompute
# =========================================================
@torch.no_grad()
def precompute_msrvtt_embeddings_v2(
    d_cfg: dict,
    annos: list,
    vision_mapper: VisionMapper,
    clip_model,
    tokenizer,
    device: torch.device,
    out_dir: str,
    normalize: bool = True,
    shard_size: Optional[int] = 2048,
):
    _ensure_dir(out_dir)

    # Deduplicate: one entry per video_id, first caption
    annos = _deduplicate_annotations(annos)
    print(f"Unique videos after deduplication: {len(annos)}")

    shard_idx = 0
    shard_text: List[np.ndarray] = []
    shard_vis: List[np.ndarray] = []
    shard_video_ids: List[str] = []
    shard_fns: List[str] = []

    def flush():
        nonlocal shard_idx
        if len(shard_video_ids) == 0:
            return

        text_np = np.stack(shard_text, axis=0)      # (N, D)
        vis_np = np.stack(shard_vis, axis=0)         # (N, D)
        video_ids_np = np.array(shard_video_ids, dtype=object)

        meta = {
            "dataset_name": "msrvtt",
            "split": d_cfg.get("split", "unknown"),
            "num_items": int(len(video_ids_np)),
            "embedding_dim": int(vis_np.shape[-1]),
            "text_mode": "first_caption",
            "normalization": bool(normalize),
            "vision_format": d_cfg.get("vision_format"),
            "vision_sample_num": d_cfg.get("vision_sample_num"),
            "vision_transforms": d_cfg.get("vision_transforms"),
            "version": "v2_unique_videos",
        }

        out_path = os.path.join(
            out_dir,
            f"{d_cfg.get('split', 'split')}_shard{shard_idx:03d}.npz",
        )
        _save_npz(out_path, text_np, vis_np, video_ids_np, shard_fns, meta)

        print(
            f"[SAVED] {out_path} | "
            f"N={len(video_ids_np)} | "
            f"vision={vis_np.shape} | "
            f"text={text_np.shape}"
        )

        shard_idx += 1
        shard_text.clear()
        shard_vis.clear()
        shard_video_ids.clear()
        shard_fns.clear()

    split_name = d_cfg.get("split", "unknown")
    print(f"\n=== Precomputing MSRVTT-v2 embeddings: {split_name} ===")

    skipped = 0
    for anno in tqdm(annos, desc=f"MSRVTT-v2 {split_name}"):
        video_id = anno["video_id"]

        # First caption
        caption = _get_caption(anno)
        if caption is None:
            skipped += 1
            continue

        # ----- Vision: sample frames -> CLIP -> mean-pool -----
        vision_pixels = vision_mapper.read(video_id)
        if vision_pixels is None:
            skipped += 1
            continue

        vision_pixels = vision_pixels.to(device)
        frame_emb = clip_model.encode_image(vision_pixels)  # (T, D)
        if normalize:
            frame_emb = F.normalize(frame_emb, dim=-1)
        video_emb = frame_emb.mean(dim=0)  # (D,)
        if normalize:
            video_emb = F.normalize(video_emb, dim=-1)

        # ----- Text: first caption -----
        tokens = tokenizer([caption]).to(device)
        text_emb = clip_model.encode_text(tokens)  # (1, D)
        if normalize:
            text_emb = F.normalize(text_emb, dim=-1)

        shard_text.append(_to_numpy(text_emb)[0])   # (D,)
        shard_vis.append(_to_numpy(video_emb))       # (D,)
        shard_video_ids.append(video_id)
        shard_fns.append(f"{video_id}.mp4")

        if shard_size is not None and len(shard_video_ids) >= shard_size:
            flush()

    flush()
    print(f"\nDone. Skipped {skipped} videos. Saved under: {out_dir}")


# =========================================================
# Main
# =========================================================
def main() -> None:
    data_cfg = {
        "train": {
            "split": "train",
            "txt": "/mnt/media/emanuele/few_dimensions/dataset/msrvtt/MSRVTT/txts",
            "vision": "/mnt/media/emanuele/few_dimensions/dataset/msrvtt/MSRVTT/videos/videos",
            "vision_format": "video_rawvideo",
            "vision_sample_num": 4,
            "vision_transforms": "crop_flip",
        },
        "test": {
            "split": "test",
            "txt": "/mnt/media/emanuele/few_dimensions/dataset/msrvtt/MSRVTT/txt_test",
            "vision": "/mnt/media/emanuele/few_dimensions/dataset/msrvtt/MSRVTT/video_test",
            "vision_format": "video_rawvideo",
            "vision_sample_num": 8,
            "vision_transforms": "crop_flip",
        },
    }

    parser = argparse.ArgumentParser(
        description="Precompute MSRVTT embeddings v2 (unique videos, first caption only)"
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--normalize", default=True, action="store_true")
    parser.add_argument("--shard_size", type=int, default=2048)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    device_str = args.device or ("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (
            f"/mnt/media/emanuele/few_dimensions/dataset/msrvtt/"
            f"{args.clip_model}___{args.clip_pretrained}_v2/precomputed_{args.split}"
        )

    clip_model, preprocess, tokenizer = build_clip(device, args.clip_model, args.clip_pretrained)
    vision_mapper = VisionMapper(data_cfg[args.split])

    anno_path = os.path.join(
        data_cfg[args.split]["txt"],
        f"descs_ret_{args.split}.json",
    )
    annos = json.load(open(anno_path))

    print("\n[CONFIG]")
    print(f"SPLIT           : {args.split}")
    print(f"ANNO_PATH       : {anno_path}")
    print(f"VISION_DIR      : {data_cfg[args.split]['vision']}")
    print(f"OUT_DIR         : {out_dir}")
    print(f"DEVICE          : {device}")
    print(f"NORMALIZE       : {bool(args.normalize)}")
    print(f"SHARD_SIZE      : {args.shard_size}")
    print(f"TOTAL ANNOS     : {len(annos)}")

    precompute_msrvtt_embeddings_v2(
        d_cfg=data_cfg[args.split],
        annos=annos,
        vision_mapper=vision_mapper,
        clip_model=clip_model,
        tokenizer=tokenizer,
        device=device,
        out_dir=out_dir,
        normalize=bool(args.normalize),
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main()
