"""Architecture/software tests; random weights and synthetic features are not ML evidence."""
import pytest

torch = pytest.importorskip("torch", reason="Install the optional ml extra for architecture tests")
pytest.importorskip("torchvision")

from backend.swingsync.research.models import (  # noqa: E402
    FrameClassifier, FrozenMobileNetV2, TemporalClassifier, class_weights_from_counts,
    counts_from_labels, mask_from_lengths, masked_cross_entropy, select_device,
)


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("head", [FrameClassifier, TemporalClassifier])
def test_valid_output_invariant_to_nan_padding_and_batch_partners(head):
    torch.manual_seed(123)
    kwargs = {"feature_dim": 4}
    if head is TemporalClassifier:
        kwargs["hidden_size"] = 5
    model = head(**kwargs).eval()
    first = torch.randn(3, 4)
    second = torch.randn(5, 4)
    batch = torch.full((2, 8, 4), float("nan"))
    batch[0, :3], batch[1, :5] = first, second
    outputs = model(batch, [3, 5])
    for row, features in enumerate([first, second]):
        individual = model(features[None], [len(features)])
        torch.testing.assert_close(outputs[row, :len(features)], individual[0], rtol=1e-5, atol=1e-6)
    assert torch.all(outputs[0, 3:] == 0)
    assert torch.all(outputs[1, 5:] == 0)
    assert torch.isfinite(outputs).all()


def test_frame_head_cannot_use_neighbor_features():
    torch.manual_seed(7)
    frame = FrameClassifier(feature_dim=3)
    inputs = torch.randn(1, 6, 3)
    original = frame(inputs, [6])
    changed = inputs.clone()
    changed[:, 1:] *= 10
    torch.testing.assert_close(frame(changed, [6])[:, 0], original[:, 0])


def test_frozen_backbone_ignores_parent_training_and_uses_shared_features():
    torch.manual_seed(11)
    backbone = FrozenMobileNetV2(weights=None)
    frame = FrameClassifier()
    parent = torch.nn.ModuleList([backbone, frame]).train()
    assert parent.training and frame.training
    assert not backbone.training
    assert all(not module.training for module in backbone.modules())
    assert all(not p.requires_grad for p in backbone.parameters())
    batchnorms = [m for m in backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    before = [m.running_mean.clone() for m in batchnorms]
    images = torch.randn(2, 3, 64, 64, requires_grad=True)
    features = backbone(images)
    assert features.shape == (2, 1280)
    assert not features.requires_grad
    torch.testing.assert_close(features, backbone(images))
    for old, module in zip(before, batchnorms):
        torch.testing.assert_close(old, module.running_mean)
    lengths = [1, 1]
    logits = frame(features[:, None], lengths)
    temporal_logits = TemporalClassifier(hidden_size=4)(features[:, None], lengths)
    assert logits.shape == temporal_logits.shape == (2, 1, 9)
    masked_cross_entropy(logits, torch.tensor([[0], [8]]), torch.ones((2, 1), dtype=torch.bool)).backward()
    assert all(p.grad is None for p in backbone.parameters())
    assert frame.classifier.weight.grad is not None
    assert backbone.provenance["swingsync_event_training"] is False
    assert "not-pretrained" in backbone.provenance["weights"]
    assert backbone.provenance["weights_url"] is None


def test_masked_weighted_loss_padding_has_no_loss_or_gradient():
    torch.manual_seed(19)
    scores = torch.randn(2, 5, 9, requires_grad=True)
    labels = torch.tensor([[0, 1, 8, -100, -100], [2, 8, -100, -100, -100]])
    mask = mask_from_lengths([3, 2], 5)
    with torch.no_grad():
        scores[~mask] = float("nan")
    weights = class_weights_from_counts(torch.tensor([2] * 8 + [100]))
    result = masked_cross_entropy(scores, labels, mask, weights)
    expected = torch.nn.functional.cross_entropy(scores[mask], labels[mask], weight=weights)
    torch.testing.assert_close(result, expected)
    result.backward()
    assert torch.isfinite(result)
    assert torch.all(scores.grad[~mask] == 0)
    assert torch.isfinite(scores.grad[mask]).all()


def test_counts_ignore_padded_labels_and_weights_address_imbalance():
    labels = torch.tensor([list(range(9)) + [-100], [8] * 10])
    mask = mask_from_lengths([9, 10], 10)
    counts = counts_from_labels(labels, mask)
    assert counts.tolist() == [1] * 8 + [11]
    weights = class_weights_from_counts(counts)
    assert weights.shape == (9,)
    assert weights.mean().item() == pytest.approx(1)
    assert weights[0] / weights[8] == pytest.approx(11)
    for bad in ([0] * 9, [1] * 8, torch.ones(9), [1] * 8 + [-1]):
        with pytest.raises(ValueError):
            class_weights_from_counts(bad)


@pytest.mark.parametrize("bad", [[0], [4], [1.5], [True], [], [[1]], [float("nan")]])
def test_invalid_lengths_fail_before_recurrence(bad):
    with pytest.raises(ValueError):
        TemporalClassifier(feature_dim=2)(torch.ones(1, 3, 2), bad)


def test_invalid_valid_features_masks_labels_weights_and_logits():
    model = TemporalClassifier(feature_dim=2)
    features = torch.ones(1, 3, 2)
    features[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(features, [3])
    logits, labels = torch.ones(1, 3, 9), torch.zeros((1, 3), dtype=torch.int64)
    with pytest.raises(ValueError, match="prefix"):
        masked_cross_entropy(logits, labels, torch.tensor([[True, False, True]]))
    with pytest.raises(ValueError, match="within class"):
        masked_cross_entropy(logits, labels - 1, torch.ones((1, 3), dtype=torch.bool))
    with pytest.raises(ValueError, match="positive"):
        masked_cross_entropy(logits, labels, torch.ones((1, 3), dtype=torch.bool), torch.zeros(9))
    logits[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        masked_cross_entropy(logits, labels, torch.ones((1, 3), dtype=torch.bool))
    with pytest.raises(ValueError, match="weights"):
        FrozenMobileNetV2(weights="DEFAULT")


def test_device_selection_errors_do_not_silently_fallback(monkeypatch):
    monkeypatch.setattr("backend.swingsync.research.models.detect_devices",
                        lambda: {"cuda": False, "mps": False, "cpu": True})
    assert select_device("auto").type == "cpu"
    for unavailable in ["cuda", "mps", "tpu"]:
        with pytest.raises(ValueError):
            select_device(unavailable)
