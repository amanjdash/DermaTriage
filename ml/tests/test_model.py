import torch
from ml.src.labels import CLASS_NAMES
from ml.src.model import build_model


def test_efficientnet_classifier_has_seven_finite_outputs():
    model = build_model(pretrained=False).eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 96, 96))
        probabilities = torch.softmax(logits, dim=1)
    assert logits.shape == (1, len(CLASS_NAMES))
    assert torch.isfinite(logits).all()
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1), atol=1e-5)

