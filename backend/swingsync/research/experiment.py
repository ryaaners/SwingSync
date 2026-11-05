"""Bounded CPU head training and internal-validation selection on frozen caches.

This module never accepts outer-test caches. It uses full-clip temporal context,
which differs from the reference's 64-frame recurrent reset and must be disclosed.
No stochastic augmentation is enabled. Process identity/locking belongs to the
caller; creating a driver alone is not a launched or completed training run.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral
import os
import platform
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .evaluation import evaluate_clip, events_from_labels, summarize_evaluations
from .features import load_feature_cache
from .golfdb import make_split_manifest
from .models import (
    FrameClassifier, TemporalClassifier, class_weights_from_counts, mask_from_lengths, masked_cross_entropy,
)
from .training import _rng_state, collate_features, seed_everything


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _hash_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in [{minimum},{maximum}]')


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int = 1
    seed: int = 20261105
    model: str = 'frame'
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = .001
    hidden_size: int = 64
    selection_metric: str = 'validation_pce'
    context: str = 'full-clip'
    augmentation: str = 'none'
    device: str = 'cpu'

    def __post_init__(self):
        _integer(self.schema_version, 'schema_version', 1, 1)
        _integer(self.seed, 'seed', 0, 2**32 - 1)
        _integer(self.epochs, 'epochs', 1, 200)
        _integer(self.batch_size, 'batch_size', 1, 64)
        _integer(self.hidden_size, 'hidden_size', 1, 512)
        if (isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float))
                or not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1):
            raise ValueError('learning_rate must be finite and in (0,1]')
        if self.model not in {'frame', 'temporal'}:
            raise ValueError('model must be frame or temporal')
        if (self.selection_metric, self.context, self.augmentation, self.device) != (
                'validation_pce', 'full-clip', 'none', 'cpu'):
            raise ValueError('Only validation_pce/full-clip/no-augmentation/CPU is supported')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError('Experiment config must contain exactly the preregistered schema fields')
        return cls(**value)


class HeadExperiment:
    """CPU deterministic full-clip training; model selection sees inner validation only.

    `cache_paths` must cover exactly all train and inner-validation clip IDs in
    the regenerated manifest. Missing clips, subsets and test caches fail closed.
    The constructor streams validated caches once to establish provenance and
    full-training class counts; it does not fit a model or retain the dataset in
    memory. Later batches reload and checksum their cache files before use.
    """
    def __init__(self, config, dataset, manifest, cache_paths):
        if not isinstance(config, ExperimentConfig):
            config = ExperimentConfig.from_dict(config)
        try:
            expected = make_split_manifest(dataset, manifest['outer_fold'], manifest['seed'],
                                           manifest['validation_fraction_sources'])
        except (KeyError, TypeError) as exc:
            raise ValueError('Invalid split manifest') from exc
        if manifest != expected:
            raise ValueError('Split manifest differs from audited source-grouped partitions')
        self.config = config
        self.train_ids = tuple(manifest['partitions']['train']['clip_ids'])
        self.validation_ids = tuple(manifest['partitions']['validation']['clip_ids'])
        if (not self.train_ids or not self.validation_ids or not isinstance(cache_paths, dict)
                or set(cache_paths) != set(self.train_ids) | set(self.validation_ids)):
            raise ValueError('Cache coverage must be exactly complete train + inner validation; outer test is forbidden')
        self.cache_paths = {i: Path(path) for i, path in cache_paths.items()}
        records = {r.clip_id: r for r in dataset.records}
        self.cache_fingerprints = {}
        signature = None
        counts = np.zeros(9, dtype=np.int64)
        for clip_id in sorted(cache_paths):
            path = self.cache_paths[clip_id]
            initial_hash = _hash_file(path)
            clip = load_feature_cache(path)
            identity = clip.metadata['identity']
            record = records[clip_id]
            if (identity.get('annotation_sha256') != dataset.sha256 or identity.get('clip_id') != clip_id
                    or identity.get('source_id') != record.source_id
                    or identity.get('outer_split') != record.outer_split
                    or identity.get('original_events') != list(record.original_events)
                    or identity.get('annotation_event_indices') != list(record.event_indices)
                    or identity.get('historical_source_indices') != [record.source_frame_index(i)
                                                                    for i in range(record.expected_frame_count)]
                    or identity.get('timestamp_provenance') != 'reencoded-clip-pts'
                    or clip.features.shape != (record.expected_frame_count, 1280)):
                raise ValueError('Cache annotation/data/feature provenance differs from manifest')
            pipeline = {key: identity.get(key) for key in (
                'backbone_class', 'backbone_state_sha256', 'backbone_provenance', 'transform', 'torch',
                'torchvision', 'numpy', 'pyav', 'ffmpeg_libraries', 'extraction_device',
                'implementation_sha256', 'python', 'timestamp_provenance')}
            if signature is None:
                signature = pipeline
            elif pipeline != signature:
                raise ValueError('All clips must use the exact same frozen backbone and feature pipeline')
            if _hash_file(path) != initial_hash:
                raise ValueError('Cache changed while establishing experiment provenance')
            self.cache_fingerprints[str(clip_id)] = {
                'sha256': initial_hash, 'cache_key': clip.metadata['cache_key'],
            }
            if clip_id in self.train_ids:
                counts += np.bincount(clip.labels, minlength=9)
        self.class_counts = counts
        self.weights = class_weights_from_counts(torch.from_numpy(counts.copy()))
        root = Path(__file__).parent
        implementations = {name: _hash_file(root / name) for name in (
            'experiment.py', 'models.py', 'training.py', 'evaluation.py', 'features.py', 'golfdb.py', 'timeline.py')}
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.provenance = {
            'schema_version': 1, 'config': config.to_dict(), 'dataset_sha256': dataset.sha256,
            'manifest_sha256': hashlib.sha256(_json_bytes(manifest)).hexdigest(),
            'cache_fingerprints': self.cache_fingerprints, 'feature_pipeline': signature,
            'class_counts_train_only': counts.tolist(), 'class_weights': self.weights.tolist(),
            'implementation_sha256': implementations, 'torch': str(torch.__version__), 'numpy': np.__version__,
            'outer_test_evaluated': False,
            'runtime': {'torch_num_threads': torch.get_num_threads(),
                        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                        'machine': platform.machine(), 'platform': platform.platform()},
        }
        seed_everything(config.seed)
        self.model = (FrameClassifier() if config.model == 'frame' else
                      TemporalClassifier(hidden_size=config.hidden_size)).to(device='cpu', dtype=torch.float32)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.epoch = self.cursor = self.optimizer_steps = 0
        self.order = self._epoch_order(0)
        self.epoch_loss_numerator = self.epoch_loss_denominator = 0.
        self.history = []
        self.best_epoch = None
        self.best_validation_pce = None
        self._best_model = None

    def _epoch_order(self, epoch):
        # Hash ranking is deterministic across RNG library implementations; no frame split.
        return sorted(self.train_ids, key=lambda i: hashlib.sha256(
            _json_bytes([self.config.seed, epoch, i])).hexdigest())

    def _clip(self, clip_id):
        path = self.cache_paths[clip_id]
        if _hash_file(path) != self.cache_fingerprints[str(clip_id)]['sha256']:
            raise ValueError('A feature cache changed after experiment preflight')
        return load_feature_cache(path)

    def _validate_inner(self):
        results = []
        numerator = denominator = 0.
        self.model.eval()
        with torch.no_grad():
            for clip_id in self.validation_ids:
                clip = self._clip(clip_id)
                features, labels, lengths = collate_features([clip])
                mask = mask_from_lengths(lengths, features.shape[1])
                logits = self.model(features, lengths)
                loss = masked_cross_entropy(logits, labels, mask, self.weights)
                mass = float(self.weights[labels[mask]].sum())
                numerator += float(loss) * mass
                denominator += mass
                probabilities = torch.softmax(logits[0], dim=-1).numpy()
                results.append(evaluate_clip(probabilities, events_from_labels(clip.labels),
                                             timestamps_seconds=clip.pts_seconds))
        report = summarize_evaluations(results)
        report.update(partition='inner-validation-only', loss=numerator / denominator,
                      temporal_context='full-clip', outer_test_evaluated=False)
        return report

    def _assert_runtime(self):
        runtime = self.provenance['runtime']
        if (torch.get_num_threads() != runtime['torch_num_threads']
                or torch.are_deterministic_algorithms_enabled() != runtime['deterministic_algorithms']):
            raise ValueError('CPU thread/deterministic runtime differs from experiment provenance')

    def advance(self, *, max_batches=None):
        """Advance a bounded number of optimizer batches; validate at epoch ends.

        Caller must save a checkpoint after its intended persistence boundary.
        A partial epoch never becomes a completed epoch or selection result.
        """
        self._assert_runtime()
        if max_batches is not None:
            _integer(max_batches, 'max_batches', 1, 1_000_000)
        done = 0
        while self.epoch < self.config.epochs and (max_batches is None or done < max_batches):
            if self.cursor < len(self.order):
                ids = self.order[self.cursor:self.cursor + self.config.batch_size]
                batch = collate_features([self._clip(i) for i in ids])
                features, labels, lengths = batch
                mask = mask_from_lengths(lengths, features.shape[1])
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(features, lengths)
                loss = masked_cross_entropy(logits, labels, mask, self.weights)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5., error_if_nonfinite=True)
                self.optimizer.step()
                mass = float(self.weights[labels[mask]].sum())
                self.epoch_loss_numerator += float(loss.detach()) * mass
                self.epoch_loss_denominator += mass
                self.cursor += len(ids)
                self.optimizer_steps += 1
                done += 1
            if self.cursor == len(self.order):
                validation = self._validate_inner()
                self.history.append({'epoch': self.epoch + 1,
                                     'training_loss': self.epoch_loss_numerator / self.epoch_loss_denominator,
                                     'validation': validation})
                score = validation['pce']
                # Exact ties retain earliest epoch; no held-out test or loss tie-break tuning.
                if self.best_validation_pce is None or score > self.best_validation_pce:
                    self.best_validation_pce, self.best_epoch = score, self.epoch + 1
                    self._best_model = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                self.epoch += 1
                self.cursor = 0
                self.order = self._epoch_order(self.epoch)
                self.epoch_loss_numerator = self.epoch_loss_denominator = 0.
        return self.summary()

    def summary(self):
        return json.loads(_json_bytes({
            'evidence_type': 'inner-validation-head-training-not-outer-benchmark',
            'state': 'completed' if self.epoch == self.config.epochs else 'partial',
            'completed_epochs': self.epoch, 'clips_processed_in_current_epoch': self.cursor,
            'optimizer_steps': self.optimizer_steps, 'best_epoch': self.best_epoch,
            'best_validation_pce': self.best_validation_pce, 'history': self.history,
            'class_counts_train_only': self.class_counts.tolist(),
            'outer_test_evaluated': False, 'validated_for_serving': False,
        }))

    def selected_model_state(self):
        if self._best_model is None:
            raise ValueError('No complete epoch/internal-validation selection exists yet')
        return {k: v.clone() for k, v in self._best_model.items()}

    def save(self, path):
        """Publish an immutable CPU continuation checkpoint, including data cursor/selection."""
        self._assert_runtime()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'schema_version': 1, 'provenance': self.provenance,
                   'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                   'rng': _rng_state(), 'best_model': self._best_model,
                   'progress': {'epoch': self.epoch, 'cursor': self.cursor, 'order': self.order,
                                'optimizer_steps': self.optimizer_steps, 'history': self.history,
                                'loss_numerator': self.epoch_loss_numerator,
                                'loss_denominator': self.epoch_loss_denominator,
                                'best_epoch': self.best_epoch, 'best_validation_pce': self.best_validation_pce}}
        descriptor, temporary = tempfile.mkstemp(prefix='.experiment-', dir=path.parent)
        try:
            with os.fdopen(descriptor, 'wb') as output:
                torch.save(payload, output)
                output.flush()
                os.fsync(output.fileno())
            os.link(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return _hash_file(path)

    def restore(self, path):
        """Restore only identical config/data/pipeline/runtime CPU experiment state."""
        self._assert_runtime()
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('schema_version') != 1 or saved.get('provenance') != self.provenance:
            raise ValueError('Experiment checkpoint provenance mismatch')
        progress = saved['progress']
        epoch, cursor = progress['epoch'], progress['cursor']
        _integer(epoch, 'saved epoch', 0, self.config.epochs)
        _integer(cursor, 'saved cursor', 0, len(self.train_ids))
        if (progress['order'] != self._epoch_order(epoch) or len(progress['history']) != epoch
                or (cursor != len(self.train_ids) and cursor % self.config.batch_size)
                or (epoch == self.config.epochs and cursor != 0)):
            raise ValueError('Invalid experiment checkpoint epoch/order/cursor')
        expected_steps = epoch * math.ceil(len(self.train_ids) / self.config.batch_size)
        expected_steps += math.ceil(cursor / self.config.batch_size)
        if progress['optimizer_steps'] != expected_steps:
            raise ValueError('Invalid experiment checkpoint optimizer step count')
        for key in ('loss_numerator', 'loss_denominator'):
            value = progress[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid experiment checkpoint loss accumulator')
        if (cursor == 0 and (progress['loss_numerator'] != 0 or progress['loss_denominator'] != 0)) or (
                cursor > 0 and progress['loss_denominator'] <= 0):
            raise ValueError('Checkpoint loss accumulation differs from data cursor')
        scores = []
        for number, record in enumerate(progress['history'], start=1):
            validation = record.get('validation', {})
            score = validation.get('pce')
            if (record.get('epoch') != number or validation.get('partition') != 'inner-validation-only'
                    or validation.get('outer_test_evaluated') is not False
                    or not isinstance(score, (int, float)) or isinstance(score, bool)
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError('Invalid experiment checkpoint validation history')
            scores.append(score)
        expected_best = max(scores) if scores else None
        expected_epoch = scores.index(expected_best) + 1 if scores else None
        if progress['best_validation_pce'] != expected_best or progress['best_epoch'] != expected_epoch:
            raise ValueError('Checkpoint selection differs from internal-validation history')
        if bool(saved['best_model'] is not None) != bool(scores):
            raise ValueError('Checkpoint selected weights are missing or premature')
        if saved['best_model'] is not None:
            template = self.model.state_dict()
            if set(saved['best_model']) != set(template) or any(
                    not isinstance(v, torch.Tensor) or v.shape != template[k].shape
                    or v.dtype != template[k].dtype or not torch.isfinite(v).all()
                    for k, v in saved['best_model'].items()):
                raise ValueError('Invalid selected model state')
        if any(not torch.isfinite(value).all() for value in saved['model'].values()):
            raise ValueError('Nonfinite model checkpoint')
        self.model.load_state_dict(saved['model'], strict=True)
        self.optimizer.load_state_dict(saved['optimizer'])
        random.setstate(saved['rng']['python'])
        torch.set_rng_state(saved['rng']['torch_cpu'])
        state = saved['rng']['numpy']
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4]))
        self.epoch, self.cursor, self.order = epoch, cursor, progress['order']
        self.optimizer_steps, self.history = progress['optimizer_steps'], progress['history']
        self.epoch_loss_numerator = progress['loss_numerator']
        self.epoch_loss_denominator = progress['loss_denominator']
        self.best_epoch, self.best_validation_pce = progress['best_epoch'], progress['best_validation_pce']
        self._best_model = saved['best_model']
        return self.summary()
