"""
Precompute embeddings for MSRVTT using OpenCLIP.

Inspired by the MSCOCO precomputation pipeline, but without ImageNet labels.
For each video:
- sample frames with VisionMapper
- encode frames with CLIP image encoder
- mean-pool frame embeddings into a single video embedding
- encode text from either the first caption or all captions
- save shards as compressed NPZ files
"""

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath("../.."))

import argparse
import json
from typing import Optional, Dict, Any, List, Union

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
        pretrained=pretrained
    )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def _save_npz(
    out_path: str,
    text_emb: Union[np.ndarray, object],
    vision_emb: np.ndarray,
    video_ids: np.ndarray,
    caption_ids: Union[np.ndarray, object],
    fns: List[str],
    meta: Dict[str, Any],
) -> None:
    payload = {
        "text_emb": text_emb,
        "vision_emb": vision_emb.astype(np.float32),
        "video_ids": video_ids,
        "caption_ids": caption_ids,
        "fns": np.array(fns, dtype=object),
        "meta": np.array([meta], dtype=object),
    }
    np.savez_compressed(out_path, **payload)


def _normalize_caption_field(desc_field):
    """
    MSRVTT annotations can contain:
    - a string caption
    - a list of captions
    """
    if isinstance(desc_field, list):
        return [str(x) for x in desc_field if str(x).strip() != ""]
    if isinstance(desc_field, str):
        desc = desc_field.strip()
        return [desc] if desc != "" else []
    return []


# =========================================================
# Precompute
# =========================================================
@torch.no_grad()
def precompute_msrvtt_embeddings(
    d_cfg,
    annos,
    vision_mapper,
    preprocess,
    clip_model,
    tokenizer,
    device,
    out_dir,
    normalize: bool = True,
    shard_size: Optional[int] = 2048,
    text_mode: str = "all_captions",   # "first_caption" | "all_captions"
):
    _ensure_dir(out_dir)

    # Keep same filtering spirit as your dataset code: only examples with audio
    annos_new = []
    for key in annos:
        video_id = key["video_id"]
        path = os.path.join(d_cfg["audio"], f"{video_id}.mp3")
        if os.path.exists(path):
            annos_new.append(key)

    print(f"Found {len(annos_new)} valid annotations")

    shard_idx = 0
    shard_text, shard_vis = [], []
    shard_video_ids, shard_caption_ids = [], []
    shard_fns = []

    def flush():
        nonlocal shard_idx

        if len(shard_video_ids) == 0:
            return

        if text_mode == "first_caption":
            text_np = np.stack(shard_text, axis=0)  # (N, D)
            caption_ids_np = np.array(shard_caption_ids, dtype=np.int64)  # (N,)
        elif text_mode == "all_captions":
            text_np = np.array(shard_text, dtype=object)        # each item is (K_i, D)
            caption_ids_np = np.array(shard_caption_ids, dtype=object)  # each item is (K_i,)
        else:
            raise ValueError(f"Unknown text_mode={text_mode}")

        vis_np = np.stack(shard_vis, axis=0)  # (N, D)
        video_ids_np = np.array(shard_video_ids, dtype=object)

        meta = {
            "dataset_name": "msrvtt",
            "split": d_cfg.get("split", "unknown"),
            "num_items": int(len(video_ids_np)),
            "embedding_dim": int(vis_np.shape[-1]),
            "text_mode": text_mode,
            "normalization": bool(normalize),
            "vision_format": d_cfg.get("vision_format", None),
            "vision_sample_num": d_cfg.get("vision_sample_num", None),
            "vision_transforms": d_cfg.get("vision_transforms", None),
            "clip_model": getattr(clip_model, "__class__", type(clip_model)).__name__,
        }

        out_path = os.path.join(out_dir, f"{d_cfg.get('split', 'split')}_shard{shard_idx:03d}.npz")
        _save_npz(
            out_path=out_path,
            text_emb=text_np,
            vision_emb=vis_np,
            video_ids=video_ids_np,
            caption_ids=caption_ids_np,
            fns=shard_fns,
            meta=meta,
        )

        print(
            f"[SAVED] {out_path} | "
            f"N={len(video_ids_np)} | "
            f"vision={vis_np.shape} | "
            f"text_type={type(text_np)}"
        )

        shard_idx += 1
        shard_text.clear()
        shard_vis.clear()
        shard_video_ids.clear()
        shard_caption_ids.clear()
        shard_fns.clear()

    print(f"\n=== Precomputing MSRVTT embeddings: {d_cfg.get('split', 'unknown')} ===")

    for anno in tqdm(annos_new, desc=f"MSRVTT {d_cfg.get('split', 'unknown')}"):
        video_id = anno["video_id"]
        captions = _normalize_caption_field(anno.get("desc", anno.get("caption", [])))

        if len(captions) == 0:
            continue

        # -------------------------------------------------
        # Vision: sample frames and encode with CLIP
        # -------------------------------------------------
        vision_pixels = vision_mapper.read(video_id)
        if vision_pixels is None:
            continue

        # vision_pixels: (T, 3, 224, 224)
        vision_pixels = vision_pixels.to(device)

        frame_emb = clip_model.encode_image(vision_pixels)  # (T, D)

        if normalize:
            frame_emb = F.normalize(frame_emb, dim=-1)

        # Mean-pool frame embeddings into a single video embedding
        video_emb = frame_emb.mean(dim=0, keepdim=True)  # (1, D)

        if normalize:
            video_emb = F.normalize(video_emb, dim=-1)

        # -------------------------------------------------
        # Text
        # -------------------------------------------------
        if text_mode == "first_caption":
            tokens = tokenizer([captions[0]]).to(device)
            text_emb = clip_model.encode_text(tokens)  # (1, D)
            if normalize:
                text_emb = F.normalize(text_emb, dim=-1)

            shard_text.append(_to_numpy(text_emb)[0])   # (D,)
            shard_caption_ids.append(0)

        elif text_mode == "all_captions":
            tokens = tokenizer(captions).to(device)
            text_emb = clip_model.encode_text(tokens)   # (K, D)
            if normalize:
                text_emb = F.normalize(text_emb, dim=-1)

            shard_text.append(_to_numpy(text_emb))      # (K, D)
            shard_caption_ids.append(np.arange(len(captions), dtype=np.int64))
        else:
            raise ValueError(f"Unknown text_mode={text_mode}")

        shard_vis.append(_to_numpy(video_emb)[0])       # (D,)
        shard_video_ids.append(video_id)
        shard_fns.append(f"{video_id}.mp4")

        if shard_size is not None and len(shard_video_ids) >= shard_size:
           flush()

    flush()
    print(f"\nDone. Saved under: {out_dir}")


# =========================================================
# Main
# =========================================================
def main() -> None:
    data_cfg = {
        "train": {
            "split": "train",
            "txt": "/mnt/media/emanuele/few_dimensions/dataset/msrvtt/MSRVTT/txts",
            "vision": "/mnt/media/emanuele/few_dimensions/dataset/msrvtt/MSRVTT/videos/videos",
            "audio": "/mnt/media/emanuele/few_dimensions/dataset/msrvtt/MSRVTT/audios",
            "vision_format": "video_rawvideo",
            "vision_sample_num": 4,
            "vision_transforms": "crop_flip",
        },
        "test": {
            "split": "test",
            "txt": "/mnt/media/emanuele/few_dimensions/dataset/msrvtt/MSRVTT/txt_test",
            "vision": "/mnt/media/emanuele/few_dimensions/dataset/msrvtt/MSRVTT/video_test",
            "audio": "/mnt/media/emanuele/few_dimensions/dataset/msrvtt/MSRVTT/audio_test",
            "vision_format": "video_rawvideo",
            "vision_sample_num": 8,
            "vision_transforms": "crop_flip",
        }
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Split to process",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP model name.",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help="OpenCLIP pretrained tag.",
    )
    parser.add_argument(
        "--normalize",
        default=True,
        action="store_true",
        help="If set, L2-normalize CLIP embeddings.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=2048,
        help="Videos per shard.",
    )
    parser.add_argument(
        "--text_mode",
        type=str,
        default="all_captions",
        choices=["first_caption", "all_captions"],
        help="Use the first caption or all captions per video.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for CLIP (e.g. 'cuda:0', 'cuda:1', or 'cpu'). Default: auto.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional custom output directory.",
    )
    args = parser.parse_args()

    device_str = args.device or ("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (
            f"/mnt/media/emanuele/few_dimensions/dataset/msrvtt/"
            f"{args.clip_model}___{args.clip_pretrained}/precomputed_{args.split}"
        )

    clip_model, preprocess, tokenizer = build_clip(device, args.clip_model, args.clip_pretrained)
    vision_mapper = VisionMapper(data_cfg[args.split])

    anno_path = os.path.join(
        data_cfg[args.split]["txt"],
        f"descs_ret_{args.split}.json"
    )
    annos = json.load(open(anno_path))

    print("\n[CONFIG]")
    print(f"SPLIT           : {args.split}")
    print(f"ANNO_PATH       : {anno_path}")
    print(f"VISION_DIR      : {data_cfg[args.split]['vision']}")
    print(f"AUDIO_DIR       : {data_cfg[args.split]['audio']}")
    print(f"OUT_DIR         : {out_dir}")
    print(f"DEVICE          : {device}")
    print(f"TEXT_MODE       : {args.text_mode}")
    print(f"NORMALIZE       : {bool(args.normalize)}")
    print(f"SHARD_SIZE      : {args.shard_size}")

    precompute_msrvtt_embeddings(
        d_cfg=data_cfg[args.split],
        annos=annos,
        vision_mapper=vision_mapper,
        preprocess=preprocess,
        clip_model=clip_model,
        tokenizer=tokenizer,
        device=device,
        out_dir=out_dir,
        normalize=bool(args.normalize),
        shard_size=args.shard_size,
        text_mode=args.text_mode,
    )


if __name__ == "__main__":
    main()