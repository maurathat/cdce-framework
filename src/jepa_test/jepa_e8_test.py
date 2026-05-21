#!/usr/bin/env python3
"""
JEPA E8 Affinity Test
=====================
Tests whether I-JEPA vision transformer representations show preferential
alignment with E8 root system structure when linearly projected to 8D,
compared to D4 (4D) and random (16D) baselines.

Part of the CDCE framework.
"""

import json
import itertools
import ssl
import sys
from pathlib import Path

# Fix macOS Python SSL certificate issue
import certifi
import os
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR = ROOT_DIR / "weights"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

NUM_IMAGES = 1000
TRAIN_EPOCHS = 20
TRAIN_LR = 0.001


# ── Step 3: Build root systems ─────────────────────────────────────────────

def build_e8_roots():
    """Construct all 240 E8 roots in 8 dimensions using exact arithmetic."""
    roots = []

    # 112 integer roots: permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(8)
                    v[i] = si
                    v[j] = sj
                    roots.append(v)

    # 128 half-integer roots: (±½)^8 with even number of minus signs
    for signs in itertools.product([0.5, -0.5], repeat=8):
        num_neg = sum(1 for s in signs if s < 0)
        if num_neg % 2 == 0:
            roots.append(np.array(signs))

    roots = np.array(roots, dtype=np.float64)
    assert roots.shape == (240, 8), f"Expected 240 roots, got {roots.shape[0]}"

    # Normalize to unit length
    norms = np.linalg.norm(roots, axis=1, keepdims=True)
    roots = roots / norms
    return roots


def build_d4_roots():
    """Construct all 24 D4 roots in 4 dimensions."""
    roots = []
    # 24 roots: permutations of (±1, ±1, 0, 0)
    for i in range(4):
        for j in range(i + 1, 4):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(4)
                    v[i] = si
                    v[j] = sj
                    roots.append(v)
    roots = np.array(roots, dtype=np.float64)
    assert roots.shape == (24, 4), f"Expected 24 roots, got {roots.shape[0]}"
    norms = np.linalg.norm(roots, axis=1, keepdims=True)
    roots = roots / norms
    return roots


# ── Checkpoint loading ──────────────────────────────────────────────────────

def load_ijepa_encoder():
    """Load I-JEPA encoder from checkpoint, with ViT-H → ViT-L fallback."""

    # Try ViT-H first, then ViT-L
    candidates = [
        ("IN1K-vith14-300e.pth.tar", "vit_huge_patch14_224", 1280),
        ("IN1K-vitl16-300e.pth.tar", "vit_large_patch16_224", 1024),
    ]

    ckpt_path = None
    model_name = None
    embed_dim = None

    for fname, mname, edim in candidates:
        p = WEIGHTS_DIR / fname
        if p.exists() and p.stat().st_size > 1000:
            ckpt_path = p
            model_name = mname
            embed_dim = edim
            break

    if ckpt_path is None:
        print("WARNING: No valid I-JEPA checkpoint found in weights/")
        print("Falling back to timm pretrained ViT-L (ImageNet supervised)")
        model = timm.create_model("vit_large_patch16_224", pretrained=True, num_classes=0)
        model = model.to(DEVICE).eval()
        embed_dim = 1024
        return model, embed_dim

    print(f"Loading checkpoint: {ckpt_path.name} ({ckpt_path.stat().st_size / 1e9:.1f} GB)")
    try:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except RuntimeError as e:
        print(f"  ERROR loading checkpoint: {e}")
        print("  Falling back to timm pretrained ViT-L (ImageNet supervised)")
        model = timm.create_model("vit_large_patch16_224", pretrained=True, num_classes=0)
        model = model.to(DEVICE).eval()
        return model, 1024

    # Inspect top-level keys
    if isinstance(checkpoint, dict):
        print(f"  Checkpoint top-level keys: {list(checkpoint.keys())}")
    else:
        print(f"  Checkpoint is type: {type(checkpoint)}")

    # Extract encoder state dict
    state_dict = None
    for key in ["target_encoder", "encoder", "model", "state_dict"]:
        if isinstance(checkpoint, dict) and key in checkpoint:
            state_dict = checkpoint[key]
            print(f"  Using key: '{key}'")
            break

    if state_dict is None:
        if isinstance(checkpoint, dict):
            # Maybe the checkpoint IS the state dict
            state_dict = checkpoint
            print("  Using checkpoint directly as state_dict")
        else:
            raise RuntimeError("Cannot find encoder weights in checkpoint")

    # Strip 'module.' prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        cleaned[new_key] = v

    # Create timm model — I-JEPA has no CLS token, so use no_embed_class=True
    print(f"  Creating {model_name} (embed_dim={embed_dim})")
    model = timm.create_model(
        model_name, pretrained=False, num_classes=0,
        no_embed_class=True,  # I-JEPA doesn't use CLS token in pos_embed
    )

    # Handle pos_embed size mismatch: I-JEPA stores 256 (patches only),
    # timm may expect 257 (CLS + patches). Adapt if needed.
    model_pos_shape = model.pos_embed.shape  # e.g. (1, 256, 1280)
    ckpt_pos_shape = cleaned.get("pos_embed", torch.zeros(1)).shape
    if "pos_embed" in cleaned and ckpt_pos_shape != model_pos_shape:
        print(f"  Adapting pos_embed: checkpoint {ckpt_pos_shape} → model {model_pos_shape}")
        ckpt_pos = cleaned["pos_embed"]
        if ckpt_pos_shape[1] < model_pos_shape[1]:
            # Checkpoint has fewer tokens — pad with zeros (for missing CLS)
            pad = torch.zeros(1, model_pos_shape[1] - ckpt_pos_shape[1], ckpt_pos_shape[2])
            cleaned["pos_embed"] = torch.cat([pad, ckpt_pos], dim=1)
        elif ckpt_pos_shape[1] > model_pos_shape[1]:
            # Checkpoint has more tokens — trim
            cleaned["pos_embed"] = ckpt_pos[:, :model_pos_shape[1], :]

    # Try loading; report mismatches
    result = model.load_state_dict(cleaned, strict=False)
    if result.missing_keys:
        print(f"  Missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}...")
    if result.unexpected_keys:
        print(f"  Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}...")

    model = model.to(DEVICE).eval()
    print(f"  Encoder loaded successfully on {DEVICE}")
    return model, embed_dim


# ── Data loading ────────────────────────────────────────────────────────────

def load_cifar10_subset(n=NUM_IMAGES):
    """Load first n CIFAR-10 test images, resized to 224x224."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=str(ROOT_DIR / "data"), train=False, download=True, transform=transform
    )
    subset = torch.utils.data.Subset(dataset, list(range(n)))
    loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False, num_workers=0)
    return loader


# ── Extract embeddings ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, loader):
    """Extract [CLS] token embeddings from frozen encoder."""
    all_embs = []
    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(DEVICE)
        emb = model(images)  # timm num_classes=0 returns pooled features
        all_embs.append(emb.cpu())
        if (batch_idx + 1) % 5 == 0:
            print(f"  Extracted batch {batch_idx + 1}/{len(loader)}")
    return torch.cat(all_embs, dim=0)


# ── Projection & training ──────────────────────────────────────────────────

def train_projection(embeddings, roots_np, dim, epochs=TRAIN_EPOCHS, lr=TRAIN_LR):
    """Train a linear projection to minimize cosine distance to nearest root."""
    roots_t = torch.tensor(roots_np, dtype=torch.float32).to(DEVICE)
    proj = nn.Linear(embeddings.shape[1], dim, bias=False).to(DEVICE)
    optimizer = optim.Adam(proj.parameters(), lr=lr)

    embs = embeddings.to(DEVICE)

    for epoch in range(epochs):
        proj.train()
        projected = proj(embs)
        # Normalize to unit sphere
        projected_norm = projected / (projected.norm(dim=1, keepdim=True) + 1e-8)

        # Cosine similarity to all roots: (N, num_roots)
        cos_sim = projected_norm @ roots_t.T.float()
        # Nearest root cosine similarity
        nearest_cos, _ = cos_sim.max(dim=1)
        # Minimize negative cosine (maximize alignment)
        loss = -nearest_cos.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} — loss: {loss.item():.4f}, "
                  f"mean cos: {nearest_cos.mean().item():.4f}")

    proj.eval()
    return proj


@torch.no_grad()
def random_projection(embeddings, dim):
    """Apply a random linear projection (no training)."""
    proj = nn.Linear(embeddings.shape[1], dim, bias=False)
    projected = proj(embeddings)
    return projected.numpy()


# ── Analysis ────────────────────────────────────────────────────────────────

def analyze_alignment(projected_np, roots_np, label=""):
    """Compute alignment metrics between projected embeddings and root system."""
    # Normalize embeddings
    norms = np.linalg.norm(projected_np, axis=1, keepdims=True)
    projected_norm = projected_np / (norms + 1e-8)

    # Cosine similarity to all roots
    cos_sim = projected_norm @ roots_np.T  # (N, num_roots)
    nearest_cos = cos_sim.max(axis=1)
    nearest_idx = cos_sim.argmax(axis=1)

    # For E8: first 112 are integer roots, next 128 are half-integer
    num_roots = roots_np.shape[0]
    if num_roots == 240:
        integer_count = np.sum(nearest_idx < 112)
        half_int_count = np.sum(nearest_idx >= 112)
        pct_integer = 100.0 * integer_count / len(nearest_idx)
        pct_half_int = 100.0 * half_int_count / len(nearest_idx)
    else:
        pct_integer = None
        pct_half_int = None

    results = {
        "label": label,
        "dim": projected_np.shape[1],
        "num_roots": num_roots,
        "mean_cos": float(nearest_cos.mean()),
        "std_cos": float(nearest_cos.std()),
        "max_cos": float(nearest_cos.max()),
        "min_cos": float(nearest_cos.min()),
        "median_cos": float(np.median(nearest_cos)),
        "cosine_distribution": nearest_cos.tolist(),
    }
    if pct_integer is not None:
        results["pct_integer_roots"] = round(pct_integer, 1)
        results["pct_half_integer_roots"] = round(pct_half_int, 1)

    print(f"  [{label}] mean={results['mean_cos']:.4f}, "
          f"max={results['max_cos']:.4f}, median={results['median_cos']:.4f}", end="")
    if pct_integer is not None:
        print(f", int={pct_integer:.1f}%/half={pct_half_int:.1f}%", end="")
    print()

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("JEPA E8 AFFINITY TEST")
    print("=" * 60)

    # Build root systems
    print("\n[1] Building root systems...")
    e8_roots = build_e8_roots()
    d4_roots = build_d4_roots()
    np.save(str(RESULTS_DIR / "e8_roots.npy"), e8_roots)
    np.save(str(RESULTS_DIR / "d4_roots.npy"), d4_roots)
    print(f"  E8: {e8_roots.shape} roots, D4: {d4_roots.shape} roots")

    # Load encoder
    print("\n[2] Loading I-JEPA encoder...")
    model, embed_dim = load_ijepa_encoder()
    print(f"  Encoder embed_dim: {embed_dim}")

    # Load data
    print("\n[3] Loading CIFAR-10 test images...")
    loader = load_cifar10_subset(NUM_IMAGES)
    print(f"  Loaded {NUM_IMAGES} images")

    # Extract embeddings
    print("\n[4] Extracting embeddings...")
    embeddings = extract_embeddings(model, loader)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Run tests at each dimension
    configs = [
        (4, d4_roots, "D4"),
        (8, e8_roots, "E8"),
        (16, e8_roots, "E8"),  # E8 padded for 16D comparison
    ]

    all_results = {}

    for dim, roots, root_name in configs:
        print(f"\n{'─' * 60}")
        print(f"[5] Dimension {dim} (vs {root_name} roots)")
        print(f"{'─' * 60}")

        if dim == 16:
            # For 16D, pad roots to 16D with zeros for comparison
            roots_padded = np.zeros((roots.shape[0], 16))
            roots_padded[:, :8] = roots
            # Re-normalize
            norms = np.linalg.norm(roots_padded, axis=1, keepdims=True)
            # Padded roots won't all be unit-length; only keep the structure
            roots_padded = roots_padded / (norms + 1e-8)
            analysis_roots = roots_padded
        else:
            analysis_roots = roots

        # Trained projection
        print(f"  Training projection ({embed_dim} → {dim})...")
        proj = train_projection(embeddings, analysis_roots, dim)
        with torch.no_grad():
            trained_proj = proj(embeddings.to(DEVICE)).cpu().numpy()
        trained_res = analyze_alignment(trained_proj, analysis_roots, f"trained-{dim}D")

        # Random projection (no training)
        print(f"  Random projection ({embed_dim} → {dim})...")
        random_proj = random_projection(embeddings, dim)
        random_res = analyze_alignment(random_proj, analysis_roots, f"random-proj-{dim}D")

        # Null baseline: random unit vectors
        print(f"  Null baseline (random unit vectors in {dim}D)...")
        rand_vecs = np.random.randn(NUM_IMAGES, dim)
        rand_vecs = rand_vecs / (np.linalg.norm(rand_vecs, axis=1, keepdims=True) + 1e-8)
        null_res = analyze_alignment(rand_vecs, analysis_roots, f"null-{dim}D")

        all_results[dim] = {
            "trained": trained_res,
            "random_proj": random_res,
            "null": null_res,
        }

    # ── Summary table ───────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("JEPA E8 AFFINITY TEST RESULTS")
    print(f"{'─' * 90}")
    print(f"{'Dim':>4} | {'Trained Mean Cos':>17} | {'Random Proj Mean Cos':>21} | "
          f"{'Random Vectors Mean Cos':>23} | {'Signal?':>8}")
    print(f"{'─' * 90}")

    for dim in [4, 8, 16]:
        r = all_results[dim]
        trained_mc = r["trained"]["mean_cos"]
        random_mc = r["random_proj"]["mean_cos"]
        null_mc = r["null"]["mean_cos"]
        # Signal: trained meaningfully above both baselines
        lift_over_null = (trained_mc - null_mc) / (null_mc + 1e-8) * 100
        signal = "YES" if trained_mc > random_mc * 1.05 and trained_mc > null_mc * 1.1 else "maybe" if trained_mc > null_mc * 1.05 else "no"
        print(f"{dim:>4} | {trained_mc:>17.4f} | {random_mc:>21.4f} | "
              f"{null_mc:>23.4f} | {signal:>8}")

    print(f"{'─' * 90}")

    # ── 8D detailed breakdown ───────────────────────────────────────────────
    r8 = all_results[8]
    print(f"\n8D Detailed (E8 roots):")
    for key in ["trained", "random_proj", "null"]:
        d = r8[key]
        line = f"  {d['label']:>20}: mean={d['mean_cos']:.4f} ± {d['std_cos']:.4f}, max={d['max_cos']:.4f}"
        if "pct_integer_roots" in d:
            line += f", int_roots={d['pct_integer_roots']}%, half_int={d['pct_half_integer_roots']}%"
        print(line)

    # ── Histogram ───────────────────────────────────────────────────────────
    print("\n[6] Generating histogram...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for key, color, label in [
        ("trained", "#e63946", "Trained projection (8D → E8)"),
        ("random_proj", "#457b9d", "Random projection (8D)"),
        ("null", "#a8dadc", "Random vectors (8D)"),
    ]:
        data = r8[key]["cosine_distribution"]
        ax.hist(data, bins=50, alpha=0.6, color=color, label=label, density=True)

    ax.set_xlabel("Cosine similarity to nearest E8 root", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("JEPA E8 Affinity: 8D Projection Cosine Distributions", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = RESULTS_DIR / "jepa_e8_histogram.png"
    fig.savefig(str(hist_path), dpi=150)
    print(f"  Saved: {hist_path}")

    # ── Save JSON ───────────────────────────────────────────────────────────
    # Strip distribution arrays from JSON for readability (keep in separate file)
    json_results = {}
    for dim, data in all_results.items():
        json_results[str(dim)] = {}
        for key, val in data.items():
            entry = {k: v for k, v in val.items() if k != "cosine_distribution"}
            json_results[str(dim)][key] = entry

    json_results["metadata"] = {
        "device": str(DEVICE),
        "num_images": NUM_IMAGES,
        "embed_dim": embed_dim,
        "train_epochs": TRAIN_EPOCHS,
        "train_lr": TRAIN_LR,
        "e8_roots": 240,
        "d4_roots": 24,
    }

    json_path = RESULTS_DIR / "jepa_e8_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
