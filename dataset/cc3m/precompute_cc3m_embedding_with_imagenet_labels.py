"""
Precompute CC3M CLIP embeddings and assign ImageNet top-1 labels
using a ConvNeXt classifier, following the same conventions used for
the Flickr30k and MSCOCO pipelines in this repository.
"""

import argparse
import io
import itertools
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import open_clip
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_url
from transformers import ConvNextForImageClassification, ConvNextImageProcessor
import webdataset as wds


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def _save_npz(
    out_path: str,
    text_emb: np.ndarray,
    vision_emb: np.ndarray,
    fns: List[str],
    label_ids: np.ndarray,
    label_names: List[str],
    meta: Dict[str, Any],
    label_logits: Optional[np.ndarray] = None,
) -> None:
    payload = {
        "text_emb": text_emb,
        "vision_emb": vision_emb,
        "fns": np.array(fns, dtype=object),
        "label_ids": label_ids.astype(np.int64),
        "label_names": np.array(label_names, dtype=object),
        "meta": np.array([meta], dtype=object),
    }
    if label_logits is not None:
        payload["label_logits"] = label_logits.astype(np.float32)

    np.savez_compressed(out_path, **payload)


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


def build_convnext(model_name: str, device: torch.device):
    processor = ConvNextImageProcessor.from_pretrained(model_name)
    model = ConvNextForImageClassification.from_pretrained(model_name)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return processor, model


def build_model_output_name(model_name: str, pretrained: str) -> str:
    return f"{model_name}___{pretrained}".replace("/", "_")


def _find_first_present_key(example: Dict[str, Any], candidates: Iterable[str]) -> Optional[str]:
    for key in candidates:
        if key in example and example[key] is not None:
            return key
    return None


def _extract_caption(example: Dict[str, Any], caption_key: str) -> str:
    value = example[caption_key]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return " ".join(parts).strip()
    return str(value).strip()


def _to_pil_image(image_value: Any) -> Image.Image:
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")

    if isinstance(image_value, bytes):
        return Image.open(io.BytesIO(image_value)).convert("RGB")

    if isinstance(image_value, dict):
        if "bytes" in image_value and image_value["bytes"] is not None:
            return Image.open(io.BytesIO(image_value["bytes"])).convert("RGB")
        if "path" in image_value and image_value["path"]:
            return Image.open(image_value["path"]).convert("RGB")

    raise TypeError(f"Unsupported image payload type: {type(image_value)}")


def _filename_from_url(url: str) -> Optional[str]:
    try:
        path = urlparse(url).path
    except Exception:
        return None
    if not path:
        return None
    name = os.path.basename(path)
    return name or None


def _extract_sample_id(example: Dict[str, Any], idx: int) -> str:
    for key in ("__key__", "key", "sample_id", "id", "image_id"):
        value = example.get(key)
        if value is not None:
            if isinstance(value, bytes):
                value_str = value.decode("utf-8", errors="ignore").strip()
            else:
                value_str = str(value).strip()
            if value_str:
                return value_str

    for key in ("url", "image_url", "__url__"):
        value = example.get(key)
        if value:
            if isinstance(value, bytes):
                value_str = value.decode("utf-8", errors="ignore")
            else:
                value_str = str(value)
            maybe_name = _filename_from_url(value_str)
            if maybe_name:
                return maybe_name
            return value_str

    return f"cc3m_{idx:09d}"


def _cc3m_webdataset_urls(dataset_name: str, split_name: str) -> List[str]:
    split_to_prefix = {
        "train": "cc3m-train-",
        "validation": "cc3m-validation-",
    }
    if split_name not in split_to_prefix:
        raise ValueError(
            f"Unsupported CC3M split '{split_name}'. Expected one of: "
            f"{sorted(split_to_prefix.keys())}"
        )

    prefix = split_to_prefix[split_name]
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=dataset_name, repo_type="dataset")
    tar_files = sorted(
        f for f in repo_files
        if f.startswith(prefix) and f.endswith(".tar")
    )

    if not tar_files:
        raise RuntimeError(
            f"No shard files matching '{prefix}*.tar' were found in dataset {dataset_name}"
        )

    return [
        hf_hub_url(
            repo_id=dataset_name,
            repo_type="dataset",
            filename=filename,
        )
        for filename in tar_files
    ]


def _load_cc3m_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split_name: str,
    streaming: bool,
):
    """
    For pixparse/cc3m-wds, prefer direct WebDataset streaming because some
    environments hit a `datasets` CastError when extra sidecar columns such as
    `json` are present in the tar shards.
    """
    if streaming and dataset_name == "pixparse/cc3m-wds":
        urls = _cc3m_webdataset_urls(dataset_name, split_name)
        return wds.WebDataset(
            urls,
            shardshuffle=False,
            resampled=False,
        )

    ds_kwargs: Dict[str, Any] = {
        "path": dataset_name,
        "split": split_name,
        "streaming": streaming,
    }
    if dataset_config is not None:
        ds_kwargs["name"] = dataset_config
    return load_dataset(**ds_kwargs)


def _infer_total_samples(
    dataset_name: str,
    split_name: str,
    skip_items: int,
    max_items: Optional[int],
) -> Optional[int]:
    if dataset_name == "pixparse/cc3m-wds":
        # Counts from the dataset card / shard metadata on Hugging Face.
        known_totals = {
            "train": 2_905_954,
            "validation": 13_443,
        }
        total = known_totals.get(split_name)
        if total is None:
            return max_items
        remaining = max(0, total - max(0, skip_items))
        if max_items is not None:
            return min(remaining, max_items)
        return remaining

    return max_items


def _resolve_columns(
    sample: Dict[str, Any],
    image_key: Optional[str],
    caption_key: Optional[str],
) -> Tuple[str, str]:
    resolved_image_key = image_key or _find_first_present_key(
        sample,
        ("image", "jpg", "jpeg", "png", "webp"),
    )
    resolved_caption_key = caption_key or _find_first_present_key(
        sample,
        ("caption", "txt", "text", "sentence"),
    )

    if resolved_image_key is None:
        raise RuntimeError(
            f"Could not infer image column from sample keys: {sorted(sample.keys())}"
        )
    if resolved_caption_key is None:
        raise RuntimeError(
            f"Could not infer caption column from sample keys: {sorted(sample.keys())}"
        )

    return resolved_image_key, resolved_caption_key


@torch.no_grad()
def precompute_cc3m_embeddings_with_imagenet_labels(
    *,
    dataset_name: str,
    dataset_config: Optional[str],
    split_name: str,
    clip_model_name: str,
    clip_pretrained_name: str,
    clip_model,
    preprocess,
    tokenizer,
    device: torch.device,
    out_dir: str,
    normalize: bool,
    shard_size: Optional[int],
    convnext_processor,
    convnext_model,
    convnext_device: Optional[torch.device] = None,
    store_logits: bool = True,
    streaming: bool = True,
    max_items: Optional[int] = None,
    skip_items: int = 0,
    image_key: Optional[str] = None,
    caption_key: Optional[str] = None,
) -> None:
    _ensure_dir(out_dir)

    if convnext_device is None:
        convnext_device = device

    dataset = _load_cc3m_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split_name=split_name,
        streaming=streaming,
    )

    if skip_items:
        if hasattr(dataset, "skip"):
            dataset = dataset.skip(skip_items)
        else:
            dataset = itertools.islice(dataset, skip_items, None)

    shard_idx = 0
    shard_text: List[np.ndarray] = []
    shard_vis: List[np.ndarray] = []
    shard_fns: List[str] = []
    shard_label_ids: List[int] = []
    shard_label_names: List[str] = []
    shard_label_logits: List[float] = []

    seen = 0
    saved = 0
    skipped_no_caption = 0
    skipped_bad_image = 0
    skipped_bad_record = 0
    resolved_image_key: Optional[str] = image_key
    resolved_caption_key: Optional[str] = caption_key

    def flush() -> None:
        nonlocal shard_idx, saved

        if len(shard_fns) == 0:
            return

        text_np = np.stack(shard_text, axis=0)
        vis_np = np.stack(shard_vis, axis=0)
        label_ids_np = np.array(shard_label_ids, dtype=np.int64)
        label_logits_np = None
        if store_logits:
            label_logits_np = np.array(shard_label_logits, dtype=np.float32)

        meta = {
            "dataset_name": "cc3m",
            "hf_dataset_name": dataset_name,
            "hf_dataset_config": dataset_config,
            "split": split_name,
            "num_items": int(len(shard_fns)),
            "embedding_dim": int(vis_np.shape[-1]),
            "text_mode": "single_caption",
            "normalization": bool(normalize),
            "clip_model": clip_model_name,
            "clip_pretrained": clip_pretrained_name,
            "label_model": getattr(convnext_model, "name_or_path", None),
            "label_space": "ImageNet-1k",
            "stores_label_logits": bool(store_logits),
            "streaming": bool(streaming),
            "image_key": resolved_image_key,
            "caption_key": resolved_caption_key,
        }

        if shard_size is None:
            out_path = os.path.join(out_dir, f"{split_name}.npz")
        else:
            out_path = os.path.join(out_dir, f"{split_name}_shard{shard_idx:05d}.npz")

        _save_npz(
            out_path=out_path,
            text_emb=text_np,
            vision_emb=vis_np,
            fns=shard_fns,
            label_ids=label_ids_np,
            label_names=shard_label_names,
            meta=meta,
            label_logits=label_logits_np,
        )

        print(
            f"[SAVED] {out_path} | N={len(shard_fns)} | "
            f"vision={vis_np.shape} | text={text_np.shape}"
            + (" | logits=(N,)" if store_logits else "")
        )

        saved += len(shard_fns)
        shard_idx += 1
        shard_text.clear()
        shard_vis.clear()
        shard_fns.clear()
        shard_label_ids.clear()
        shard_label_names.clear()
        shard_label_logits.clear()

    total = _infer_total_samples(
        dataset_name=dataset_name,
        split_name=split_name,
        skip_items=skip_items,
        max_items=max_items,
    )
    iterator = tqdm(
        dataset,
        total=total,
        desc=f"{dataset_name}:{split_name}",
        unit="sample",
        dynamic_ncols=True,
    )

    for sample in iterator:
        if max_items is not None and seen >= max_items:
            break

        sample_idx = skip_items + seen
        seen += 1

        try:
            if resolved_image_key is None or resolved_caption_key is None:
                resolved_image_key, resolved_caption_key = _resolve_columns(
                    sample,
                    resolved_image_key,
                    resolved_caption_key,
                )

            caption = _extract_caption(sample, resolved_caption_key)
            if not caption:
                skipped_no_caption += 1
                continue

            sample_id = _extract_sample_id(sample, sample_idx)

            try:
                img = _to_pil_image(sample[resolved_image_key])
            except Exception:
                skipped_bad_image += 1
                continue

            img_tensor = preprocess(img).unsqueeze(0).to(device)
            tokens = tokenizer([caption]).to(device)

            text_emb = clip_model.encode_text(tokens)
            vision_emb = clip_model.encode_image(img_tensor)

            if normalize:
                text_emb = F.normalize(text_emb, dim=-1)
                vision_emb = F.normalize(vision_emb, dim=-1)

            label_inputs = convnext_processor(images=img, return_tensors="pt")
            label_inputs = {k: v.to(convnext_device) for k, v in label_inputs.items()}
            logits = convnext_model(**label_inputs).logits
            pred_id = int(torch.argmax(logits, dim=-1).item())
            pred_name = convnext_model.config.id2label.get(pred_id, str(pred_id))

            shard_text.append(_to_numpy(text_emb)[0])
            shard_vis.append(_to_numpy(vision_emb)[0])
            shard_fns.append(sample_id)
            shard_label_ids.append(pred_id)
            shard_label_names.append(pred_name)
            if store_logits:
                shard_label_logits.append(float(logits[0, pred_id].detach().cpu().item()))

            if shard_size is not None and len(shard_fns) >= shard_size:
                flush()

        except KeyboardInterrupt:
            raise
        except Exception:
            skipped_bad_record += 1
            continue

    flush()

    print("\n[SUMMARY]")
    print(f"seen                : {seen}")
    print(f"saved               : {saved}")
    print(f"skipped_no_caption  : {skipped_no_caption}")
    print(f"skipped_bad_image   : {skipped_bad_image}")
    print(f"skipped_bad_record  : {skipped_bad_record}")
    if resolved_image_key is not None:
        print(f"resolved_image_key  : {resolved_image_key}")
    if resolved_caption_key is not None:
        print(f"resolved_caption_key: {resolved_caption_key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="pixparse/cc3m-wds",
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Optional HuggingFace dataset config name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./precomputed_embeddings_with_imagenet_labels",
        help="Output root directory. Files are stored under <out_dir>/<model_name>/",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=2048,
        help="Samples per shard. Set to 0 or a negative value to write a single NPZ.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for CLIP (e.g. 'cuda:0' or 'cpu'). Default: auto.",
    )
    parser.add_argument(
        "--label_device",
        type=str,
        default=None,
        help="Device for ConvNeXt. Default: same as CLIP device.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="If set, L2-normalize CLIP embeddings.",
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
        "--convnext_name",
        type=str,
        default="facebook/convnext-xlarge-384-22k-1k",
        help="HuggingFace ConvNeXt ImageNet-1k checkpoint.",
    )
    parser.add_argument(
        "--store_logits",
        dest="store_logits",
        action="store_true",
        help="If set, store top-1 label logits in the NPZ.",
    )
    parser.add_argument(
        "--no-store_logits",
        dest="store_logits",
        action="store_false",
        help="Do not store top-1 label logits in the NPZ.",
    )
    parser.add_argument(
        "--streaming",
        dest="streaming",
        action="store_true",
        help="Load the dataset in streaming mode.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Load the dataset eagerly instead of streaming.",
    )
    parser.set_defaults(streaming=True, store_logits=True)
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Optional cap on processed samples, useful for debugging or partial runs.",
    )
    parser.add_argument(
        "--skip_items",
        type=int,
        default=0,
        help="Skip this many dataset samples before processing.",
    )
    parser.add_argument(
        "--image_key",
        type=str,
        default=None,
        help="Optional explicit image column name.",
    )
    parser.add_argument(
        "--caption_key",
        type=str,
        default=None,
        help="Optional explicit caption column name.",
    )
    args = parser.parse_args()

    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    label_device = torch.device(args.label_device) if args.label_device else device

    clip_model, preprocess, tokenizer = build_clip(
        device=device,
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
    )
    convnext_processor, convnext_model = build_convnext(
        args.convnext_name,
        device=label_device,
    )

    model_output_name = build_model_output_name(args.clip_model, args.clip_pretrained)
    out_dir = os.path.join(args.out_dir, model_output_name)
    _ensure_dir(out_dir)

    shard_size = args.shard_size if args.shard_size and args.shard_size > 0 else None

    print("\n[CONFIG]")
    print(f"DATASET_NAME    : {args.dataset_name}")
    print(f"DATASET_CONFIG  : {args.dataset_config}")
    print(f"SPLIT           : {args.split}")
    print(f"MODEL_OUTPUT    : {model_output_name}")
    print(f"OUT_DIR         : {out_dir}")
    print(f"DEVICE          : {device}")
    print(f"LABEL_DEVICE    : {label_device}")
    print(f"STREAMING       : {args.streaming}")
    print(f"MAX_ITEMS       : {args.max_items}")
    print(f"SKIP_ITEMS      : {args.skip_items}")

    precompute_cc3m_embeddings_with_imagenet_labels(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split_name=args.split,
        clip_model_name=args.clip_model,
        clip_pretrained_name=args.clip_pretrained,
        clip_model=clip_model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        out_dir=out_dir,
        normalize=bool(args.normalize),
        shard_size=shard_size,
        convnext_processor=convnext_processor,
        convnext_model=convnext_model,
        convnext_device=label_device,
        store_logits=bool(args.store_logits),
        streaming=bool(args.streaming),
        max_items=args.max_items,
        skip_items=int(args.skip_items),
        image_key=args.image_key,
        caption_key=args.caption_key,
    )

    print(f"\nDone. Saved under: {out_dir}")
    print(
        "NPZ keys: text_emb, vision_emb, fns, label_ids, label_names, meta"
        + (", label_logits" if args.store_logits else "")
    )


if __name__ == "__main__":
    main()
