# Inspired by 'The Surprising Effectiveness of Deep Orthogonal Procrustes Alignment in Unsupervised Domain Adaptation"
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def collect_embeddings(loader, device):
    """
    Collects (text, vision) pairs from loader into torch tensors.
    Each batch is (text_emb, vision_emb).
    """
    Xs, Ys = [], []
    # if max_sample is none, we take all the available samples
    
    with torch.no_grad():
        for text_b, vis_b, _ in tqdm(loader, desc=f"Collecting samples"):
            text_b = F.normalize(text_b.to(device), dim=-1)
            vis_b  = F.normalize(vis_b.to(device), dim=-1)
            Xs.append(text_b); Ys.append(vis_b)
    X = torch.cat(Xs, dim=0)
    Y = torch.cat(Ys, dim=0)
    return X, Y

def fit_subspace_alignment(loader, d_sub = 256, device="cuda"):
    # 1. Collect embeddings
    X, Y = collect_embeddings(loader, device)
    print(f"Collected {X.shape[0]} samples of dimension {X.shape[1]}")

    # center (PCA-style)
    muX = X.mean(axis=0, keepdims=True)
    muY = Y.mean(axis=0, keepdims=True)
    
    Xc = X - muX    
    Yc = Y - muY
    
    # SVD to get bases Ws, Wt (top d_sub right singular vectors)
    # Xc = U S V^T => Ws = V[:, :d_sub]
    # Yc = U S V^T => Wt = V[:, :d_sub]
    Xc = Xc.cpu().numpy()
    Yc = Yc.cpu().numpy()
    _, _, VtX = np.linalg.svd(Xc, full_matrices=False)
    _, _, VtY = np.linalg.svd(Yc, full_matrices=False)

    Ws = VtX[:d_sub].T   # (D, d_sub)
    Wt = VtY[:d_sub].T   # (D, d_sub)

    # closed-form subspace alignment
    Phi = Wt.T @ Ws      # (d_sub, d_sub)   Eq (7) in the paper :contentReference[oaicite:4]{index=4}

    return {"muX": muX, "muY": muY, "Ws": Ws, "Wt": Wt, "Phi": Phi, "d_sub": d_sub}
        
def apply_subspace_alignment(X, Y, model, device, renorm=True):
    eps = 1e-12
    muX, muY = model["muX"], model["muY"]
    Ws, Wt, Phi = model["Ws"], model["Wt"], model["Phi"]
    muX = muX.to(device)
    muY = muY.to(device)
    Ws = torch.as_tensor(Ws).float().to(device)
    Wt = torch.as_tensor(Wt).float().to(device)
    Phi = torch.as_tensor(Phi).float().to(device)
    # center
    Xc = X - muX
    Yc = Y - muY

    # align Y into X-space (Eq. 10 style) :contentReference[oaicite:6]{index=6}
    Y_al = Yc @ Wt @ Phi @ Ws.T

    # optionally bring back to X mean
    Y_al = Y_al + muX

    if renorm:
        Xn = X / (torch.norm(X, dim=1, keepdim=True) + eps)
        Yn = Y / (torch.norm(Y, dim=1, keepdim=True) + eps)
        Yaln = Y_al / (torch.norm(Y_al, dim=1, keepdim=True) + eps)
        return Xn, Yn, Yaln
    
    return X, Y, Y_al

def analyze_subspace_dimensions(model, device):
    """
    sub_model must contain:
      - Ws: (D, d_sub) text basis
      - Wt: (D, d_sub) vision basis
    """
    Ws = model["Ws"]
    Wt = model["Wt"]

    #subspace importance per original dim
    imp_X = np.sum(Ws**2, axis=1)  # (D,)
    imp_Y = np.sum(Wt**2, axis=1)  # (D,)
    imp_joint = 0.5 * (imp_X + imp_Y)
    
    top_impX = np.argsort(imp_X)[::-1]
    top_impY = np.argsort(imp_Y)[::-1]
    top_imp_joint = np.argsort(imp_joint)[::-1]
    
    return top_impX, top_impY, top_imp_joint