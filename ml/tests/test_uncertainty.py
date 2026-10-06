import torch
from ml.src.metrics import bootstrap_macro_f1, expected_calibration_error
from ml.src.uncertainty import mc_dropout_probabilities, predictive_entropy, summarize_mc
from torch import nn


def test_predictive_entropy_orders_uniform_above_confident():
    uniform = torch.full((4, 1, 7), 1 / 7)
    confident = torch.zeros((4, 1, 7))
    confident[:, :, 0] = 0.94
    confident[:, :, 1:] = 0.01
    assert predictive_entropy(uniform).item() > predictive_entropy(confident).item()
    summary = summarize_mc(confident)
    assert summary["class_variance"].shape == (1, 7)


def test_mc_dropout_keeps_batchnorm_in_eval_and_restores_training_state():
    model = nn.Sequential(nn.BatchNorm1d(4), nn.Dropout(0.7), nn.Linear(4, 3))
    model.train()
    inputs = torch.ones((2, 4))
    running_mean = model[0].running_mean.clone()
    samples = mc_dropout_probabilities(model, inputs, passes=8)
    assert samples.shape == (8, 2, 3)
    assert torch.allclose(samples.sum(dim=-1), torch.ones((8, 2)))
    assert torch.equal(model[0].running_mean, running_mean)
    assert model.training and model[0].training and model[1].training
    assert samples.var(dim=0, unbiased=False).sum().item() > 0


def test_calibration_and_bootstrap_return_finite_summary():
    probabilities = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).numpy()
    targets = torch.tensor([0, 1]).numpy()
    assert expected_calibration_error(probabilities, targets) == 0
    interval = bootstrap_macro_f1(targets, targets, class_count=2, repeats=50)
    assert interval == [1.0, 1.0]

