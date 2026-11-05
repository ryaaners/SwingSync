"""Synthetic cached tensors prove resume mechanics, never model quality."""
import random

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from backend.swingsync.research.models import FrameClassifier, TemporalClassifier  # noqa: E402
from backend.swingsync.research.training import (  # noqa: E402
    restore_checkpoint, save_checkpoint, seed_everything, train_steps,
)


@pytest.mark.parametrize('constructor', [FrameClassifier, TemporalClassifier])
def test_exact_cpu_resume_and_provenance(tmp_path, constructor):
    torch.set_num_threads(1)
    seed_everything(7)
    batch = (torch.randn(2, 10, 4), torch.arange(20).reshape(2, 10) % 9, torch.tensor([10, 8]))
    model = constructor(feature_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    provenance = {'dataset': 'synthetic', 'configuration': {'seed': 7, 'model': constructor.__name__}}
    train_steps(model, optimizer, batch, torch.ones(9), 2)
    path = tmp_path / 'checkpoint.pt'
    digest = save_checkpoint(path, model, optimizer, step=2, provenance=provenance)
    assert len(digest) == 64
    expected_random = (random.random(), float(np.random.random()), torch.rand(2))
    expected_losses = train_steps(model, optimizer, batch, torch.ones(9), 2)
    resumed = constructor(feature_dim=4)
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=.001)
    with pytest.raises(ValueError, match='provenance'):
        restore_checkpoint(path, resumed, resumed_optimizer, expected_provenance={'dataset': 'other'})
    assert restore_checkpoint(path, resumed, resumed_optimizer, expected_provenance=provenance) == 2
    assert random.random() == expected_random[0]
    assert float(np.random.random()) == expected_random[1]
    assert torch.equal(torch.rand(2), expected_random[2])
    assert train_steps(resumed, resumed_optimizer, batch, torch.ones(9), 2) == expected_losses
    assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in model.state_dict().items())
    with pytest.raises(FileExistsError):
        save_checkpoint(path, resumed, resumed_optimizer, step=4, provenance=provenance)
