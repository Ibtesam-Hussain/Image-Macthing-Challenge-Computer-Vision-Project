from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

try:
    import open_clip
except ImportError:
    open_clip = None

try:
    import timm
except ImportError:
    timm = None


def _device(device: str) -> torch.device:
    return torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")


def load_clip(model_name: str, pretrained: str, device: str):
    if open_clip is None:
        raise ImportError("open-clip-torch not installed")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(_device(device)).eval()
    return model, preprocess


def load_dino(model_name: str, device: str):
    if timm is None:
        raise ImportError("timm not installed")
    # Normalize common alias to timm’s dinov2 small 14 model name
    if model_name == "dinov2_vits14":
        model_name = "vit_small_patch14_dinov2.lvd142m"
    model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
    model = model.to(_device(device)).eval()
    size = model.default_cfg.get("input_size", (3, 224, 224))[1]
    preprocess = T.Compose(
        [
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=model.default_cfg["mean"], std=model.default_cfg["std"]),
        ]
    )
    return model, preprocess


def embed_images(
    image_paths: List[Path],
    device: str,
    batch_size: int = 8,
    clip_model_name: Optional[str] = None,
    clip_pretrained: str = "openai",
    dino_model_name: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    feats: Dict[str, np.ndarray] = {}
    device_t = _device(device)

    clip_model, clip_tf = (None, None)
    if clip_model_name:
        clip_model, clip_tf = load_clip(clip_model_name, clip_pretrained, device)

    dino_model, dino_tf = (None, None)
    if dino_model_name:
        dino_model, dino_tf = load_dino(dino_model_name, device)

    def run_model(model, tf, imgs: List[Path]) -> torch.Tensor:
        tensors = []
        for p in imgs:
            img = Image.open(p).convert("RGB")
            tensors.append(tf(img))
        batch = torch.stack(tensors).to(device_t)
        with torch.no_grad():
            out = model(batch)
            if isinstance(out, tuple):
                out = out[0]
        return out.detach().float()

    for idx in tqdm(range(0, len(image_paths), batch_size), desc="Embedding"):
        batch_paths = image_paths[idx : idx + batch_size]
        clip_out = (
            run_model(clip_model, clip_tf, batch_paths) if clip_model is not None else None
        )
        dino_out = (
            run_model(dino_model, dino_tf, batch_paths) if dino_model is not None else None
        )
        for i, p in enumerate(batch_paths):
            vecs = []
            if clip_out is not None:
                vecs.append(clip_out[i])
            if dino_out is not None:
                vecs.append(dino_out[i])
            feats[p.name] = torch.cat(vecs).float().cpu().numpy()
    return feats


def l2_normalize(feats: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    normed = {}
    for k, v in feats.items():
        v = v.astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        normed[k] = v
    return normed

