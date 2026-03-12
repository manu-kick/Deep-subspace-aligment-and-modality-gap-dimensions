"""
MSCOCO dataloader for CLIP embeddings with ImageNet labels (ConvNeXt top-1).
"""

import os
from typing import Optional, List, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MSCOCOEmbeddingsDatasetWithImageNetLabels(Dataset):
	def __init__(
		self,
		precomputed_dir: str,
		split_name: str = "train_shard",
		return_metadata: bool = False,
		allow_multi_caption: bool = False,
		text_index: int = 0,
		require_labels: bool = True,
		return_label_name: bool = False,
		return_logits: bool = False,
	) -> None:
		"""
		Supported NPZ formats (from precompute_mscoco_embeddings_with_imagenet_labels.py):
		  - vision_emb:  (N, D)
		  - text_emb:    (N, D) OR object array (N,), each item (K_i, D)
		  - img_ids:     (N,)
		  - caption_ids: (N,) optional
		  - label_ids:   (N,) int64
		  - label_names: (N,) object array of strings
		  - label_logits:(N,) float32 (optional)

		Args:
			precomputed_dir: directory with NPZ shards
			split_name: substring used to select split files
			return_metadata: if True returns img_id/caption_id/label_name too
			allow_multi_caption: if True supports variable number of captions per sample
			text_index: caption index used when a sample has multiple captions
			require_labels: if True, fail if label_ids / label_names are missing
			return_label_name: if True, return label_name in non-metadata mode
			return_logits: if True, return label_logits when available
		"""
		self.text_embeddings: List[Any] = []
		self.vision_embeddings: List[np.ndarray] = []
		self.img_ids: List[np.ndarray] = []
		self.caption_ids: List[np.ndarray] = []
		self.label_ids: List[np.ndarray] = []
		self.label_names: List[np.ndarray] = []
		self.label_logits: List[np.ndarray] = []

		self.return_metadata = return_metadata
		self.allow_multi_caption = allow_multi_caption
		self.text_index = text_index
		self.require_labels = require_labels
		self.return_label_name = return_label_name
		self.return_logits = return_logits

		found = 0

		for fn in os.listdir(precomputed_dir):
			if fn.endswith(".npz") and (split_name in fn):
				found += 1
				path = os.path.join(precomputed_dir, fn)
				data = np.load(path, allow_pickle=True)

				for k in ["vision_emb", "text_emb", "img_ids"]:
					if k not in data:
						raise RuntimeError(f"Missing key '{k}' in {path}")

				v = data["vision_emb"]
				t = data["text_emb"]
				img_ids = data["img_ids"]

				caption_ids = (
					data["caption_ids"]
					if "caption_ids" in data
					else np.full(len(img_ids), -1, dtype=np.int64)
				)

				if self.require_labels and ("label_ids" not in data or "label_names" not in data):
					raise RuntimeError(f"{path} is missing ImageNet labels required for training.")

				if "label_ids" in data:
					label_ids = data["label_ids"]
				else:
					label_ids = np.full(len(img_ids), -1, dtype=np.int64)

				if "label_names" in data:
					label_names = data["label_names"]
				else:
					label_names = np.array(["unknown"] * len(img_ids), dtype=object)

				if self.return_logits:
					if "label_logits" in data:
						label_logits = data["label_logits"]
					else:
						label_logits = np.full(len(img_ids), np.nan, dtype=np.float32)
				else:
					label_logits = None

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

				if len(v) != len(img_ids):
					raise RuntimeError(f"vision_emb and img_ids length mismatch in {path}")
				if len(t) != len(img_ids):
					raise RuntimeError(f"text_emb and img_ids length mismatch in {path}")
				if len(caption_ids) != len(img_ids):
					raise RuntimeError(f"caption_ids and img_ids length mismatch in {path}")
				if len(label_ids) != len(img_ids):
					raise RuntimeError(f"label_ids and img_ids length mismatch in {path}")
				if len(label_names) != len(img_ids):
					raise RuntimeError(f"label_names and img_ids length mismatch in {path}")
				if label_logits is not None and len(label_logits) != len(img_ids):
					raise RuntimeError(f"label_logits and img_ids length mismatch in {path}")

				self.vision_embeddings.append(v)
				self.text_embeddings.extend(list(t))
				self.img_ids.append(img_ids)
				self.caption_ids.append(caption_ids)
				self.label_ids.append(label_ids)
				self.label_names.append(label_names)
				if label_logits is not None:
					self.label_logits.append(label_logits)

		if found == 0:
			raise RuntimeError(f"No .npz files found in {precomputed_dir} for split '{split_name}'")

		self.vision_embeddings = np.concatenate(self.vision_embeddings, axis=0)
		self.img_ids = np.concatenate(self.img_ids, axis=0)
		self.caption_ids = np.concatenate(self.caption_ids, axis=0)
		self.label_ids = np.concatenate(self.label_ids, axis=0)
		self.label_names = np.concatenate(self.label_names, axis=0)
		if self.return_logits:
			self.label_logits = np.concatenate(self.label_logits, axis=0)

		assert len(self.text_embeddings) == len(self.vision_embeddings) == len(self.img_ids)
		assert len(self.label_ids) == len(self.vision_embeddings)
		assert len(self.label_names) == len(self.vision_embeddings)
		if self.return_logits:
			assert len(self.label_logits) == len(self.vision_embeddings)

		print(
			f"[Loaded COCO ImageNet] {len(self)} samples from {precomputed_dir} | "
			f"vision_emb shape={self.vision_embeddings.shape}"
		)

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

	def __getitem__(self, idx: int):
		text_emb = self._get_text_embedding(idx)
		vision_emb = torch.as_tensor(self.vision_embeddings[idx]).float()

		label_id = torch.as_tensor(int(self.label_ids[idx])).long()
		label_name = str(self.label_names[idx])

		if self.return_metadata:
			item = (
				text_emb,
				vision_emb,
				label_id,
				int(self.img_ids[idx]),
				int(self.caption_ids[idx]),
				label_name,
			)
			if self.return_logits:
				item = item + (float(self.label_logits[idx]),)
			return item

		if self.return_label_name:
			item = (text_emb, vision_emb, label_id, label_name)
		else:
			item = (text_emb, vision_emb, label_id)

		if self.return_logits:
			item = item + (float(self.label_logits[idx]),)

		return item


def mscoco_imagenet_collate_fn(batch):
	"""
	Collate for scalar ImageNet labels. Keeps optional label_name/logit as lists.
	"""
	first = batch[0]
	n = len(first)

	if n == 3:
		texts, visions, label_ids = zip(*batch)
		return torch.stack(texts, dim=0), torch.stack(visions, dim=0), torch.stack(label_ids, dim=0)

	if n == 4:
		texts, visions, label_ids, extra = zip(*batch)
		texts = torch.stack(texts, dim=0)
		visions = torch.stack(visions, dim=0)
		label_ids = torch.stack(label_ids, dim=0)

		if isinstance(extra[0], (str, np.str_)):
			return texts, visions, label_ids, list(extra)
		return texts, visions, label_ids, torch.tensor(extra, dtype=torch.float32)

	if n == 6:
		texts, visions, label_ids, img_ids, caption_ids, label_names = zip(*batch)
		return (
			torch.stack(texts, dim=0),
			torch.stack(visions, dim=0),
			torch.stack(label_ids, dim=0),
			torch.tensor(img_ids, dtype=torch.long),
			torch.tensor(caption_ids, dtype=torch.long),
			list(label_names),
		)

	if n == 7:
		texts, visions, label_ids, img_ids, caption_ids, label_names, logits = zip(*batch)
		return (
			torch.stack(texts, dim=0),
			torch.stack(visions, dim=0),
			torch.stack(label_ids, dim=0),
			torch.tensor(img_ids, dtype=torch.long),
			torch.tensor(caption_ids, dtype=torch.long),
			list(label_names),
			torch.tensor(logits, dtype=torch.float32),
		)

	raise RuntimeError(f"Unsupported batch item size: {n}")


def make_loaders_mscoco_imagenet(
	precomputed_dir: str,
	precomputed_dir_test: str,
	batch_size: int = 256,
	num_workers: int = 0,
):
	ds_train = MSCOCOEmbeddingsDatasetWithImageNetLabels(
		precomputed_dir,
		split_name="train_shard",
		require_labels=True,
	)
	ds_test = MSCOCOEmbeddingsDatasetWithImageNetLabels(
		precomputed_dir_test,
		split_name="val_shard",
		require_labels=True,
	)

	train_loader = DataLoader(
		ds_train,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		collate_fn=mscoco_imagenet_collate_fn,
	)
	test_loader = DataLoader(
		ds_test,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		collate_fn=mscoco_imagenet_collate_fn,
	)
	return train_loader, test_loader
