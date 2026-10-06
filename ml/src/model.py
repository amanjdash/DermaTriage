"""EfficientNet-B3 factory and ImageNet preprocessing."""

from __future__ import annotations

import torch.nn as nn
from ml.src.labels import CLASS_NAMES
from torchvision import models, transforms


def build_model(num_classes: int = len(CLASS_NAMES), dropout: float = 0.30, pretrained: bool = False):
    weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b3(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[0] = nn.Dropout(p=dropout, inplace=True)
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_transforms(image_size: int = 300):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train, evaluation

