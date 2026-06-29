import os
import json
import argparse
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import accelerator
from torchvision import models

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    pass

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}


def list_images(folder: Path) -> list[Path]:
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return sorted(files)


def load_pil_fast(path: Path, draft_size: int = None) -> Image.Image:
    """
    Load image using PIL. If draft_size is provided, utilizes JPEG's native
    downscaling during decoding to dramatically reduce I/O and CPU overhead.
    """
    img = Image.open(path)
    if draft_size is not None:
        try:
            img.draft("RGB", (draft_size, draft_size))
        except Exception:
            pass
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def letterbox_pil(
    img: Image.Image, size: int = 224, fill=(128, 128, 128)
) -> Image.Image:
    """
    Resize image maintaining aspect ratio and pad to a square canvas.
    Prevents distortion on aspect-ratio-altered compressed files.
    """
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = int(w * scale), int(h * scale)

    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), fill)
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas.paste(resized, (x, y))
    return canvas


def pil_to_tensor(img: Image.Image, size: int = 224) -> torch.Tensor:
    img = letterbox_pil(img, size=size)
    arr = np.asarray(img).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return (
        cast(torch.device, accelerator.current_accelerator())
        if accelerator.is_available()
        else torch.device("cpu")
    )


def build_model(device: torch.device) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier = nn.Identity()  # Remove classifier to output embeddings
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def extract_embeddings(
    files: list[Path],
    model: nn.Module,
    device: torch.device,
    batch_size: int = 32,
    image_size: int = 224,
) -> torch.Tensor:
    all_feats = []

    for i in tqdm(range(0, len(files), batch_size), desc="Extracting deep features"):
        batch_files = files[i : i + batch_size]
        tensors = []
        for path in batch_files:
            try:
                # Use draft mode to load close to the target resolution
                img = load_pil_fast(path, draft_size=image_size * 2)
                t = pil_to_tensor(img, size=image_size)
                tensors.append(t)
            except Exception as e:
                print(f"[WARN] Failed to load {path}: {e}")

        if not tensors:
            continue

        x = torch.stack(tensors, dim=0).to(device)
        feat = model(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]

        feat = feat.float()
        feat = F.normalize(feat, dim=1)  # L2 normalize for cosine similarity
        all_feats.append(feat.cpu())

    if not all_feats:
        return torch.empty((0, 1280), dtype=torch.float32)

    return torch.cat(all_feats, dim=0)


def make_small_uint8(path: Path, size: int = 256) -> np.ndarray:
    img = load_pil_fast(path, draft_size=size * 2)
    img = letterbox_pil(img, size=size)
    return np.asarray(img, dtype=np.uint8)


def build_small_cache(files: list[Path], size: int = 256, desc: str = "") -> np.ndarray:
    arrs = []
    for p in tqdm(files, desc=desc):
        try:
            arr = make_small_uint8(p, size=size)
        except Exception as e:
            print(f"[WARN] Failed to cache thumbnail for {p}: {e}")
            arr = np.zeros((size, size, 3), dtype=np.uint8)
        arrs.append(arr)
    return np.stack(arrs, axis=0)


def calc_pixel_score_u8(a: np.ndarray, b: np.ndarray) -> float:
    """Computes Mean Squared Error (MSE) similarity on normalized RGB values."""
    af = a.astype(np.float32) / 255.0
    bf = b.astype(np.float32) / 255.0
    mse = np.mean((af - bf) ** 2)
    return float(1.0 / (1.0 + mse * 80.0))


def calc_gradient_score_u8(a: np.ndarray, b: np.ndarray) -> float:
    """Computes Sobel gradient similarity to capture high-frequency structural differences."""
    ag = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    bg = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)

    ax = cv2.Sobel(ag, cv2.CV_32F, 1, 0, ksize=3) / 255.0
    ay = cv2.Sobel(ag, cv2.CV_32F, 0, 1, ksize=3) / 255.0
    bx = cv2.Sobel(bg, cv2.CV_32F, 1, 0, ksize=3) / 255.0
    by = cv2.Sobel(bg, cv2.CV_32F, 0, 1, ksize=3) / 255.0

    mse = np.mean((ax - bx) ** 2 + (ay - by) ** 2)
    return float(1.0 / (1.0 + mse * 20.0))


def rerank_one_cached(
    selected_small: np.ndarray,
    original_files: list[Path],
    original_small_cache: np.ndarray,
    candidate_indices: list[int],
    candidate_cnn_scores: list[float],
) -> list[dict]:
    results = []
    for idx, cnn_score in zip(candidate_indices, candidate_cnn_scores):
        o_img = original_small_cache[idx]

        pixel_score = calc_pixel_score_u8(selected_small, o_img)
        grad_score = calc_gradient_score_u8(selected_small, o_img)

        # Final score weights: CNN (semantic) + Pixel (color/comp) + Gradient (edges/details)
        final_score = 0.45 * float(cnn_score) + 0.35 * pixel_score + 0.20 * grad_score

        results.append(
            {
                "original_path": str(original_files[idx]),
                "final_score": round(float(final_score), 6),
                "cnn_score": round(float(cnn_score), 6),
                "pixel_score": round(float(pixel_score), 6),
                "gradient_score": round(float(grad_score), 6),
            }
        )

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Match compressed selected photos back to high-res originals."
    )
    parser.add_argument(
        "--selected",
        required=True,
        help="Directory containing client-selected compressed images.",
    )
    parser.add_argument(
        "--originals",
        required=True,
        help="Directory containing original high-resolution photos.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of matched candidates to output per selection.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=30,
        help="Number of coarse candidates to pass to reranking.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for CNN feature extraction.",
    )
    parser.add_argument(
        "--image-size", type=int, default=224, help="Input size for the CNN encoder."
    )
    parser.add_argument(
        "--rerank-size",
        type=int,
        default=256,
        help="Resolution of cached thumbnails for reranking.",
    )
    parser.add_argument(
        "--output",
        default="outputs/result.json",
        help="Path to save matching results in JSON format.",
    )
    parser.add_argument(
        "--force-cpu", action="store_true", help="Force execution on CPU."
    )
    parser.add_argument(
        "--cache-original-emb",
        default="outputs/originals_emb.npz",
        help="Path to cache original embeddings.",
    )
    args = parser.parse_args()

    selected_files = list_images(Path(args.selected))
    original_files = list_images(Path(args.originals))

    print(f"Selected images found: {len(selected_files)}")
    print(f"Original images found: {len(original_files)}")

    device = get_device(force_cpu=args.force_cpu)
    print(f"Using device: {device}")

    model = build_model(device)

    # Handle embedding cache
    cache_path = Path(args.cache_original_emb)
    if cache_path.exists():
        print(f"Loading cached original embeddings from: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        cached_paths = [Path(p) for p in data["paths"].tolist()]
        original_emb = torch.from_numpy(data["embeddings"]).float()

        if [str(p) for p in cached_paths] != [str(p) for p in original_files]:
            print("[WARN] Cache mismatch detected. Rebuilding original embeddings...")
            original_emb = extract_embeddings(
                original_files, model, device, args.batch_size, args.image_size
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                paths=np.array([str(p) for p in original_files]),
                embeddings=original_emb.numpy(),
            )
    else:
        print("Extracting original image embeddings...")
        original_emb = extract_embeddings(
            original_files, model, device, args.batch_size, args.image_size
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            paths=np.array([str(p) for p in original_files]),
            embeddings=original_emb.numpy(),
        )

    print("Extracting selected image embeddings...")
    selected_emb = extract_embeddings(
        selected_files, model, device, args.batch_size, args.image_size
    )

    # Compute cosine similarity matrix
    sims = selected_emb @ original_emb.T

    print("Caching original thumbnails in RAM for fast structural reranking...")
    original_small_cache = build_small_cache(
        original_files, size=args.rerank_size, desc="Caching original thumbnails"
    )

    print("Caching selected thumbnails in RAM...")
    selected_small_cache = build_small_cache(
        selected_files, size=args.rerank_size, desc="Caching selected thumbnails"
    )

    all_results = {}

    for i, selected_path in enumerate(
        tqdm(selected_files, desc="Matching & Reranking")
    ):
        sim_row = sims[i]
        candidate_n = min(args.candidates, len(original_files))
        values, indices = torch.topk(sim_row, k=candidate_n)

        candidate_indices = [int(idx) for idx in indices]
        candidate_scores = [float(v) for v in values]

        reranked = rerank_one_cached(
            selected_small=selected_small_cache[i],
            original_files=original_files,
            original_small_cache=original_small_cache,
            candidate_indices=candidate_indices,
            candidate_cnn_scores=candidate_scores,
        )
        all_results[str(selected_path)] = reranked[: args.topk]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Matching complete. Results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
