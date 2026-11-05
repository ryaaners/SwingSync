"""CPU smoke-training utilities. Checkpoints are local research artifacts only."""
import hashlib
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .models import mask_from_lengths, masked_cross_entropy


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate_features(clips):
    if not clips:
        raise ValueError('Need at least one feature clip')
    dim = clips[0].features.shape[1]
    lengths = torch.tensor([len(c.labels) for c in clips], dtype=torch.long)
    features = torch.zeros((len(clips), int(lengths.max()), dim), dtype=torch.float32)
    labels = torch.full(features.shape[:2], -100, dtype=torch.long)
    for row, clip in enumerate(clips):
        if clip.features.shape != (len(clip.labels), dim):
            raise ValueError('Inconsistent feature shape')
        length = len(clip.labels)
        features[row, :length] = torch.from_numpy(clip.features.copy())
        labels[row, :length] = torch.from_numpy(clip.labels.copy())
    return features, labels, lengths


def train_steps(model, optimizer, batch, weights, steps):
    if steps < 1:
        raise ValueError('steps must be positive')
    features, labels, lengths = batch
    mask = mask_from_lengths(lengths, features.shape[1], device=features.device)
    losses = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, lengths)
        loss = masked_cross_entropy(logits, labels, mask, weights)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite smoke loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def _rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'torch_cpu': torch.get_rng_state(),
            'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]]}


def save_checkpoint(path, model, optimizer, *, step, provenance):
    """Atomic no-overwrite checkpoint; tensor/primitives permit weights_only loading."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'schema_version': 1, 'step': step, 'provenance': provenance,
               'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': _rng_state()}
    descriptor, temporary = tempfile.mkstemp(prefix='.checkpoint-', dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, 'wb') as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        # Hardlink is an atomic exclusive publish on the same filesystem.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_checkpoint(path, model, optimizer, *, expected_provenance):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('schema_version') != 1 or checkpoint.get('provenance') != expected_provenance:
        raise ValueError('Checkpoint provenance mismatch')
    model.load_state_dict(checkpoint['model'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    rng = checkpoint['rng']
    random.setstate(rng['python'])
    torch.set_rng_state(rng['torch_cpu'])
    state = rng['numpy']
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4]))
    return checkpoint['step']
