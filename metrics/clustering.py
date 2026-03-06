import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, v_measure_score

def clustering_metrics_from_two_modalities(feat_t: torch.Tensor, feat_v: torch.Tensor, labels: torch.Tensor,
                                          n_clusters=10, random_state=0):
    """
    KMeans su stack([text, vision]) e metriche vs labels duplicate.
    """
    assert feat_t.shape == feat_v.shape, "text e vision devono avere stessa shape (N, D)"
    assert feat_t.shape[0] == labels.shape[0], "labels devono avere N elementi"

    embeddings = torch.vstack([feat_t, feat_v]).cpu().numpy()     # (2N, D)
    true_labels = labels.cpu().numpy()
    true_labels_2 = np.concatenate([true_labels, true_labels], axis=0)  # (2N,)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(embeddings)

    ari = adjusted_rand_score(true_labels_2, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels_2, cluster_labels)
    hom = homogeneity_score(true_labels_2, cluster_labels)
    v   = v_measure_score(true_labels_2, cluster_labels)

    print(f"[KMeans k={n_clusters}] ARI={ari:.4f} | NMI={nmi:.4f} | Hom={hom:.4f} | V={v:.4f}")

    return {"ARI": ari, "NMI": nmi, "Homogeneity": hom, "V-measure": v}