"""
MSRVTT dataloader for precomputed CLIP text/vision embeddings.
"""

import os
import json
from typing import List, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MSRVTTEmbeddingsDataset(Dataset):
    def __init__(
        self,
        precomputed_dir: str,
        split_name: str = "train_shard",
        return_metadata: bool = False,
        allow_multi_caption: bool = False,
        text_index: int = 0,
    ) -> None:
        """
        Supported NPZ formats (from precompute_embeddings_msrvtt.py):
          - vision_emb:  (N, D)
          - text_emb:    (N, D) OR object array (N,), each item (K_i, D)
          - video_ids:   (N,) object/string array
          - caption_ids: (N,) OR object array (N,), optional
          - fns:         (N,) optional
          - meta:        optional

        Args:
            precomputed_dir: directory with NPZ shards
            split_name: substring used to select split files
            return_metadata: if True returns label/video_id/caption_id/file_name too
            allow_multi_caption: if True supports variable number of captions per sample
            text_index: caption index used when a sample has multiple captions
        """
        self.text_embeddings: List[Any] = []
        self.vision_embeddings: List[np.ndarray] = []
        self.video_ids: List[np.ndarray] = []
        self.labels: List[np.ndarray] = []
        self.caption_ids: List[Any] = []
        self.file_names: List[np.ndarray] = []

        self.return_metadata = return_metadata
        self.allow_multi_caption = allow_multi_caption
        self.text_index = text_index

        self.videoid_to_category = self._load_videoid_to_category(split_name)

        found = 0

        for fn in sorted(os.listdir(precomputed_dir)):
            if fn.endswith(".npz") and (split_name in fn):
                found += 1
                path = os.path.join(precomputed_dir, fn)
                data = np.load(path, allow_pickle=True)

                for k in ["vision_emb", "text_emb", "video_ids"]:
                    if k not in data:
                        raise RuntimeError(f"Missing key '{k}' in {path}")

                v = data["vision_emb"]
                t = data["text_emb"]
                video_ids = data["video_ids"]

                caption_ids = (
                    data["caption_ids"]
                    if "caption_ids" in data
                    else np.full(len(video_ids), -1, dtype=np.int64)
                )

                file_names = (
                    data["fns"]
                    if "fns" in data
                    else np.array([""] * len(video_ids), dtype=object)
                )

                # -------------------------
                # Validate text embeddings
                # -------------------------
                if isinstance(t, np.ndarray) and t.dtype != object:
                    if t.ndim == 2:
                        pass
                    elif t.ndim == 3 and self.allow_multi_caption:
                        pass
                    else:
                        raise RuntimeError(
                            f"Unexpected dense text_emb shape {t.shape} in {path}. "
                            f"Expected (N,D) or (N,K,D) with allow_multi_caption=True."
                        )
                elif isinstance(t, np.ndarray) and t.dtype == object:
                    if not self.allow_multi_caption:
                        raise RuntimeError(
                            f"{path} contains variable-length multi-caption text_emb "
                            f"but allow_multi_caption=False."
                        )
                else:
                    raise RuntimeError(f"Unsupported text_emb format in {path}")

                if len(v) != len(video_ids):
                    raise RuntimeError(f"vision_emb and video_ids length mismatch in {path}")
                if len(t) != len(video_ids):
                    raise RuntimeError(f"text_emb and video_ids length mismatch in {path}")
                if len(caption_ids) != len(video_ids):
                    raise RuntimeError(f"caption_ids and video_ids length mismatch in {path}")
                if len(file_names) != len(video_ids):
                    raise RuntimeError(f"fns and video_ids length mismatch in {path}")

                labels = []
                missing_video_ids = []
                for vid in video_ids:
                    vid_str = str(vid)
                    if vid_str not in self.videoid_to_category:
                        missing_video_ids.append(vid_str)
                        labels.append(-1)
                    else:
                        labels.append(self.videoid_to_category[vid_str])

                if missing_video_ids:
                    preview = missing_video_ids[:10]
                    raise RuntimeError(
                        f"Found {len(missing_video_ids)} video_ids missing from metadata "
                        f"for split '{split_name}' in {path}. First few: {preview}"
                    )

                self.vision_embeddings.append(v)
                self.text_embeddings.extend(list(t))
                self.video_ids.append(video_ids)
                self.labels.append(np.array(labels, dtype=np.int64))
                self.caption_ids.append(caption_ids)
                self.file_names.append(file_names)

        if found == 0:
            raise RuntimeError(f"No .npz files found in {precomputed_dir} for split '{split_name}'")

        self.vision_embeddings = np.concatenate(self.vision_embeddings, axis=0)
        self.video_ids = np.concatenate(self.video_ids, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.file_names = np.concatenate(self.file_names, axis=0)

        # caption_ids may be dense or object
        if len(self.caption_ids) > 0:
            if any(isinstance(x, np.ndarray) and x.dtype == object for x in self.caption_ids):
                self.caption_ids = np.concatenate(self.caption_ids, axis=0)
            else:
                try:
                    self.caption_ids = np.concatenate(self.caption_ids, axis=0)
                except Exception:
                    self.caption_ids = np.array(sum([list(x) for x in self.caption_ids], []), dtype=object)
        else:
            self.caption_ids = np.array([], dtype=np.int64)

        assert len(self.text_embeddings) == len(self.vision_embeddings) == len(self.video_ids)
        assert len(self.labels) == len(self.vision_embeddings)
        assert len(self.caption_ids) == len(self.vision_embeddings)
        assert len(self.file_names) == len(self.vision_embeddings)

        print(
            f"[Loaded MSRVTT] {len(self)} samples from {precomputed_dir} | "
            f"vision_emb shape={self.vision_embeddings.shape} | "
            f"num_classes={len(np.unique(self.labels))}"
        )

    def _load_videoid_to_category(self, split_name: str):
        """
        Load MSRVTT metadata and build video_id -> category map.

        Expected layout:
            msrvtt_dataloader.py
            MSRVTT/
                train_val_videodatainfo.json
                test_videodatainfo.json
        """
        this_dir = os.path.dirname(os.path.abspath(__file__))
        msrvtt_root = os.path.join(this_dir, "MSRVTT")

        if "train" in split_name:
            metadata_path = os.path.join(msrvtt_root, "train_val_videodatainfo.json")
        elif "test" in split_name:
            metadata_path = os.path.join(msrvtt_root, "test_videodatainfo.json")
        else:
            raise RuntimeError(
                f"Unsupported split_name='{split_name}'. Expected something containing "
                f"'train' or 'test'."
            )

        if not os.path.exists(metadata_path):
            raise RuntimeError(
                f"MSRVTT metadata file not found: {metadata_path}\n"
                f"Expected the JSON file inside the MSRVTT folder next to this script."
            )

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if "videos" not in metadata:
            raise RuntimeError(f"Metadata file {metadata_path} has no 'videos' field")

        videoid_to_category = {}
        for item in metadata["videos"]:
            if "video_id" not in item or "category" not in item:
                continue
            videoid_to_category[str(item["video_id"])] = int(item["category"])

        if len(videoid_to_category) == 0:
            raise RuntimeError(
                f"Could not build video_id -> category map from {metadata_path}"
            )

        return videoid_to_category

    def __len__(self) -> int:
        return len(self.vision_embeddings)

    def _get_text_embedding(self, idx: int) -> torch.Tensor:
        t = self.text_embeddings[idx]

        if isinstance(t, np.ndarray) and t.dtype != object and t.ndim == 1:
            return torch.as_tensor(t).float()

        if isinstance(t, np.ndarray) and t.dtype != object and t.ndim == 2:
            k = min(self.text_index, t.shape[0] - 1)
            return torch.as_tensor(t[k]).float()

        if isinstance(t, np.ndarray) and t.dtype == object:
            t = np.array(t.tolist(), dtype=np.float32)
            k = min(self.text_index, t.shape[0] - 1)
            return torch.as_tensor(t[k]).float()

        if isinstance(t, (list, tuple)):
            t = np.array(t, dtype=np.float32)
            if t.ndim == 1:
                return torch.as_tensor(t).float()
            k = min(self.text_index, t.shape[0] - 1)
            return torch.as_tensor(t[k]).float()

        raise RuntimeError(f"Unsupported text embedding format at idx={idx}: {type(t)}")

    def _get_caption_id(self, idx: int):
        cid = self.caption_ids[idx]

        if isinstance(cid, np.ndarray):
            if cid.ndim == 0:
                return int(cid.item())
            if cid.size == 0:
                return -1
            k = min(self.text_index, cid.shape[0] - 1)
            return int(cid[k])

        if isinstance(cid, (list, tuple)):
            if len(cid) == 0:
                return -1
            k = min(self.text_index, len(cid) - 1)
            return int(cid[k])

        try:
            return int(cid)
        except Exception:
            return -1

    def __getitem__(self, idx: int):
        text_emb = self._get_text_embedding(idx)
        vision_emb = torch.as_tensor(self.vision_embeddings[idx]).float()
        label = int(self.labels[idx])

        if self.return_metadata:
            return (
                text_emb,
                vision_emb,
                label,
                str(self.video_ids[idx]),
                self._get_caption_id(idx),
                str(self.file_names[idx]),
            )

        return text_emb, vision_emb, label


def msrvtt_collate_fn(batch):
    """
    Collate for text/vision embeddings.
    Supports:
      - (text_emb, vision_emb, label)
      - (text_emb, vision_emb, label, video_id, caption_id, file_name)
    """
    first = batch[0]
    n = len(first)

    if n == 3:
        texts, visions, labels = zip(*batch)
        return (
            torch.stack(texts, dim=0),
            torch.stack(visions, dim=0),
            torch.tensor(labels, dtype=torch.long),
        )

    if n == 6:
        texts, visions, labels, video_ids, caption_ids, file_names = zip(*batch)
        return (
            torch.stack(texts, dim=0),
            torch.stack(visions, dim=0),
            torch.tensor(labels, dtype=torch.long),
            list(video_ids),
            torch.tensor(caption_ids, dtype=torch.long),
            list(file_names),
        )

    raise RuntimeError(f"Unsupported batch item size: {n}")


def make_loaders_msrvtt(
    precomputed_dir: str,
    precomputed_dir_test: str,
    batch_size: int = 256,
    num_workers: int = 0,
    allow_multi_caption: bool = False,
    text_index: int = 0,
):
    ds_train = MSRVTTEmbeddingsDataset(
        precomputed_dir,
        split_name="train_shard",
        allow_multi_caption=allow_multi_caption,
        text_index=text_index,
    )
    ds_test = MSRVTTEmbeddingsDataset(
        precomputed_dir_test,
        split_name="test_shard",
        allow_multi_caption=allow_multi_caption,
        text_index=text_index,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=msrvtt_collate_fn,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=msrvtt_collate_fn,
    )
    return train_loader, test_loader