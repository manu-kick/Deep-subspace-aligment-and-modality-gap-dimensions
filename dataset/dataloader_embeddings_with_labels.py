import os
import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingsDatasetWithLabels(Dataset):
    def __init__(
        self,
        precomputed_dir,
        split_name="flickr30k",
        text_index=0,              # which caption to use (0–4)
        return_label_name=False    # if True also return string label
    ):
        """
        Args:
            precomputed_dir (str): Directory containing precomputed .npz files.
            split_name (str): Dataset split name used in filenames.
            text_index (int): Which of the 5 captions to use (default: 0).
            return_label_name (bool): If True, also return label name string.
        """

        self.text_embeddings = []
        self.vision_embeddings = []
        self.label_ids = []
        self.label_names = []

        self.text_index = text_index
        self.return_label_name = return_label_name

        # Load all matching shards
        for fn in os.listdir(precomputed_dir):
            if fn.endswith(".npz") and split_name in fn:
                path = os.path.join(precomputed_dir, fn)
                data = np.load(path, allow_pickle=True)

                self.vision_embeddings.append(data["vision_emb"])  # (N, D)
                self.text_embeddings.append(data["text_emb"])      # (N, 5, D)
                self.label_ids.append(data["label_ids"])           # (N,)
                self.label_names.append(data["label_names"])       # (N,)

        if len(self.text_embeddings) == 0:
            raise RuntimeError(f"No .npz files found in {precomputed_dir} for split '{split_name}'")

        # Concatenate shards
        self.text_embeddings = np.concatenate(self.text_embeddings, axis=0)     # (N, 5, D)
        self.vision_embeddings = np.concatenate(self.vision_embeddings, axis=0) # (N, D)
        self.label_ids = np.concatenate(self.label_ids, axis=0)                 # (N,)
        self.label_names = np.concatenate(self.label_names, axis=0)             # (N,)

        assert len(self.text_embeddings) == len(self.vision_embeddings) == len(self.label_ids)

        print(f"[Loaded] {len(self)} samples from {precomputed_dir}")

    def __len__(self):
        return len(self.vision_embeddings)

    def __getitem__(self, idx):
        # Select one caption embedding (default first)
        text_emb = self.text_embeddings[idx][self.text_index]   # (D,)
        vision_emb = self.vision_embeddings[idx]                 # (D,)
        label_id = self.label_ids[idx]
        label_name = self.label_names[idx]

        text_emb = torch.as_tensor(text_emb).float()
        vision_emb = torch.as_tensor(vision_emb).float()
        label_id = torch.as_tensor(label_id).long()

        if self.return_label_name:
            return text_emb, vision_emb, label_id, label_name
        else:
            return text_emb, vision_emb, label_id