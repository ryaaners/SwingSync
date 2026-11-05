"""Frozen image features and event heads; no artifact here enables web inference.

Architecture APIs: torchvision0.24 MobileNetV2 and torch2.9 packed sequences.
Pretrained ImageNet weights are reference image initialization, not a trained
SwingSync event model. None is explicit random initialization for software tests.
"""
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def detect_devices() -> dict:
    return {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}


def select_device(requested: str = "auto") -> torch.device:
    available = detect_devices()
    if requested == "auto":
        requested = next(name for name in ("cuda", "mps", "cpu") if available[name])
    if requested not in available:
        raise ValueError("Device must be auto, cuda, mps or cpu")
    if not available[requested]:
        raise ValueError(f"Requested device {requested} is not available")
    return torch.device(requested)


def mask_from_lengths(lengths, max_length: int, *, device=None) -> torch.Tensor:
    if isinstance(max_length, bool) or not isinstance(max_length, Integral) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    lengths = torch.as_tensor(lengths)
    if lengths.ndim != 1 or not len(lengths) or lengths.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    ):
        raise ValueError("lengths must be a nonempty integer vector")
    if torch.any(lengths < 1) or torch.any(lengths > max_length):
        raise ValueError("Every sequence length must be within the padded time dimension")
    lengths = lengths.to(device=device)
    return torch.arange(max_length, device=lengths.device)[None, :] < lengths[:, None]


def _feature_batch(features, lengths, feature_dim):
    if (not isinstance(features, torch.Tensor) or features.ndim != 3
            or not features.is_floating_point() or features.shape[2] != feature_dim):
        raise ValueError("features must be floating [B,T,feature_dim]")
    mask = mask_from_lengths(lengths, features.shape[1], device=features.device)
    if mask.shape != features.shape[:2]:
        raise ValueError("One length is required for every batch item")
    if not torch.isfinite(features[mask]).all():
        raise ValueError("Valid frame features must be finite")
    # NaN/Inf sentinels in padding must not reach linear weights or recurrence.
    clean = torch.where(mask[..., None], features, torch.zeros_like(features))
    return clean, mask


class FrozenMobileNetV2(nn.Module):
    """Shared feature extractor. Calling train() never changes frozen BN state.

    Inputs must already follow the recorded image transform. This class does not
    silently choose/download weights; only an explicit IMAGENET1K_V2 request can
    download torchvision's reference weight file into its configured cache.
    """
    feature_dim = 1280

    def __init__(self, weights: str | None = None):
        super().__init__()
        if weights not in {None, "IMAGENET1K_V2"}:
            raise ValueError("weights must be None or explicit IMAGENET1K_V2")
        reference = None if weights is None else MobileNet_V2_Weights.IMAGENET1K_V2
        model = mobilenet_v2(weights=reference, progress=False)
        self.features = model.features
        self.requires_grad_(False)
        self.provenance = {
            "architecture": "torchvision.mobilenet_v2.features-global-average-pool",
            "feature_dim": self.feature_dim,
            "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
            "weights": "random-initialization-software-smoke-not-pretrained" if reference is None else reference.name,
            "weights_url": None if reference is None else reference.url,
            "swingsync_event_training": False,
        }
        self.train(False)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if (not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3
                or not images.is_floating_point() or images.shape[0] < 1
                or min(images.shape[2:]) < 32 or not torch.isfinite(images).all()):
            raise ValueError("images must be finite floating [N,3,H,W] with H,W >=32")
        with torch.no_grad():
            return self.features(images).mean(dim=(-2, -1))


class FrameClassifier(nn.Module):
    """Independent frame head using exactly the same cached image features."""
    def __init__(self, feature_dim: int = 1280, num_classes: int = 9):
        super().__init__()
        if feature_dim < 1 or num_classes < 2:
            raise ValueError("Invalid feature/class dimensions")
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor, lengths) -> torch.Tensor:
        clean, mask = _feature_batch(features, lengths, self.feature_dim)
        logits = self.classifier(clean)
        return torch.where(mask[..., None], logits, torch.zeros_like(logits))


class TemporalClassifier(nn.Module):
    """BiLSTM head packed per sequence: padding never changes backward context."""
    def __init__(self, feature_dim: int = 1280, hidden_size: int = 64, num_classes: int = 9):
        super().__init__()
        if feature_dim < 1 or hidden_size < 1 or num_classes < 2:
            raise ValueError("Invalid feature/hidden/class dimensions")
        self.feature_dim = feature_dim
        self.recurrent = nn.LSTM(feature_dim, hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, features: torch.Tensor, lengths) -> torch.Tensor:
        clean, mask = _feature_batch(features, lengths, self.feature_dim)
        # PyTorch requires CPU lengths; unsorted batches preserve caller ordering.
        cpu_lengths = mask.sum(dim=1).to(device="cpu", dtype=torch.int64)
        packed = pack_padded_sequence(clean, cpu_lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.recurrent(packed)
        padded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=features.shape[1])
        logits = self.classifier(padded)
        return torch.where(mask[..., None], logits, torch.zeros_like(logits))


def _validated_targets(labels, valid_mask, num_classes):
    if (not isinstance(labels, torch.Tensor) or labels.ndim != 2 or labels.dtype != torch.int64
            or not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool
            or valid_mask.shape != labels.shape or valid_mask.device != labels.device):
        raise ValueError("labels must be int64 [B,T] with a matching boolean mask on the same device")
    expected = mask_from_lengths(valid_mask.sum(dim=1), labels.shape[1], device=labels.device)
    if not torch.equal(expected, valid_mask):
        raise ValueError("valid_mask must contain valid prefixes followed by padding")
    targets = labels[valid_mask]
    if torch.any(targets < 0) or torch.any(targets >= num_classes):
        raise ValueError("Valid targets must be within class bounds")
    return targets


def counts_from_labels(labels: torch.Tensor, valid_mask: torch.Tensor, num_classes: int = 9) -> torch.Tensor:
    """Count only caller-supplied TRAIN partition labels; never pool held-out data."""
    targets = _validated_targets(labels, valid_mask, num_classes)
    return torch.bincount(targets, minlength=num_classes)


def class_weights_from_counts(counts) -> torch.Tensor:
    """Inverse training frequency, normalized mean one; missing classes fail closed.

    Input counts must come solely from the relevant training partition. Group
    manifests and run provenance enforce that externally, not by guessed IDs.
    """
    counts = torch.as_tensor(counts)
    if (counts.shape != (9,) or counts.dtype not in (torch.int32, torch.int64)
            or torch.any(counts <= 0)):
        raise ValueError("counts must contain nine strictly positive training class counts")
    inverse = counts.to(torch.float64).reciprocal()
    return (inverse / inverse.mean()).to(torch.float32)


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor,
                         class_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Weighted valid-frame CE; padded logits/labels may contain sentinels."""
    if (not isinstance(labels, torch.Tensor) or not isinstance(logits, torch.Tensor)
            or logits.ndim != 3 or not logits.is_floating_point()
            or logits.shape[:2] != labels.shape or logits.device != labels.device or logits.shape[2] < 2):
        raise ValueError("logits must be floating [B,T,C] matching labels on the same device")
    targets = _validated_targets(labels, valid_mask, logits.shape[2])
    valid_logits = logits[valid_mask]
    if not torch.isfinite(valid_logits).all():
        raise ValueError("Valid logits must be finite")
    if class_weights is not None:
        weights = torch.as_tensor(class_weights)
        if (weights.shape != (logits.shape[2],) or not weights.is_floating_point()
                or not torch.isfinite(weights).all() or torch.any(weights <= 0)):
            raise ValueError("class_weights must contain one finite positive floating weight per class")
        class_weights = weights.to(device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(valid_logits, targets, weight=class_weights)
