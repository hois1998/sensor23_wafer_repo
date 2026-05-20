from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from wafer_repro.core.registry import Registry
from wafer_repro.labels import PAPER_CLASSES


@dataclass(frozen=True)
class ModelSpec:
    name: str
    paper_name: str
    notes: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "resnet18": ModelSpec("resnet18", "ResNet18", "Reference baseline used in the paper."),
    "efficientnet_v2_s": ModelSpec("efficientnet_v2_s", "EfficientNetV2-S", "Largest lightweight model compared in the paper."),
    "shufflenet_v2_x1_0": ModelSpec("shufflenet_v2_x1_0", "ShuffleNetV2 1.0x", "Channel-shuffle lightweight model."),
    "shufflenet_v2_x0_5": ModelSpec("shufflenet_v2_x0_5", "ShuffleNetV2 0.5x", "Smallest ShuffleNetV2 variant from the paper."),
    "mobilenet_v2": ModelSpec("mobilenet_v2", "MobileNetV2", "Inverted residual and linear bottleneck model."),
    "mobilenet_v3_small": ModelSpec("mobilenet_v3_small", "MobileNetV3-Small", "Best trade-off model reported by the paper."),
    "cnn_wdi": ModelSpec("cnn_wdi", "CNN-WDI-style CNN", "Compact CNN approximation for the related-work comparison."),
    "small_cnn": ModelSpec("small_cnn", "Small smoke-test CNN", "Fast local sanity-check model, not part of the paper."),
    "timeseries_cnn": ModelSpec("timeseries_cnn", "Time-series CNN", "Small 1D CNN for time-series smoke tests."),
}

PAPER_MODEL_NAMES = (
    "resnet18",
    "efficientnet_v2_s",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x0_5",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "cnn_wdi",
)

MODEL_REGISTRY: Registry[Callable[..., nn.Module]] = Registry("model")


def _replace_classifier(model: nn.Module, model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name.startswith("shufflenet_v2"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "mobilenet_v2":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "mobilenet_v3_small":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "efficientnet_v2_s":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported torchvision classifier replacement for {model_name}")


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class CNNWDIStyle(nn.Module):
    """Compact CNN used as a practical stand-in for the CNN-WDI related-work baseline.

    The Sensors paper compares against CNN-WDI [5], but does not reproduce that
    architecture in full. This model keeps the intended comparison point: a
    straightforward multi-layer CNN with far fewer architectural tricks than the
    mobile backbones.
    """

    def __init__(self, num_classes: int = len(PAPER_CLASSES), dropout: float = 0.35):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 192),
            ConvBlock(192, 192),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = len(PAPER_CLASSES), dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 16),
            nn.MaxPool2d(2),
            ConvBlock(16, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeSeriesCNN(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@MODEL_REGISTRY.register("cnn_wdi")
def _build_cnn_wdi(
    num_classes: int = len(PAPER_CLASSES),
    pretrained: bool = False,
    dropout: float = 0.35,
) -> nn.Module:
    return CNNWDIStyle(num_classes=num_classes, dropout=dropout)


@MODEL_REGISTRY.register("small_cnn")
def _build_small_cnn(
    num_classes: int = len(PAPER_CLASSES),
    pretrained: bool = False,
    dropout: float = 0.2,
) -> nn.Module:
    return SmallCNN(num_classes=num_classes, dropout=dropout)


@MODEL_REGISTRY.register("timeseries_cnn")
def _build_timeseries_cnn(
    num_classes: int = 3,
    pretrained: bool = False,
    dropout: float = 0.2,
) -> nn.Module:
    return TimeSeriesCNN(num_classes=num_classes, dropout=dropout)


def _build_torchvision_classifier(
    model_name: str,
    num_classes: int,
    pretrained: bool,
) -> nn.Module:
    from torchvision import models

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        return _replace_classifier(models.resnet18(weights=weights), model_name, num_classes)
    if model_name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        return _replace_classifier(models.efficientnet_v2_s(weights=weights), model_name, num_classes)
    if model_name == "shufflenet_v2_x1_0":
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        return _replace_classifier(models.shufflenet_v2_x1_0(weights=weights), model_name, num_classes)
    if model_name == "shufflenet_v2_x0_5":
        weights = models.ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
        return _replace_classifier(models.shufflenet_v2_x0_5(weights=weights), model_name, num_classes)
    if model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        return _replace_classifier(models.mobilenet_v2(weights=weights), model_name, num_classes)
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        return _replace_classifier(models.mobilenet_v3_small(weights=weights), model_name, num_classes)
    raise ValueError(f"Unsupported torchvision model: {model_name}")


def _register_torchvision_model(model_name: str):
    @MODEL_REGISTRY.register(model_name)
    def builder(
        num_classes: int = len(PAPER_CLASSES),
        pretrained: bool = False,
        dropout: float = 0.35,
    ) -> nn.Module:
        return _build_torchvision_classifier(model_name, num_classes=num_classes, pretrained=pretrained)

    return builder


for _torchvision_model_name in PAPER_MODEL_NAMES:
    if _torchvision_model_name != "cnn_wdi":
        _register_torchvision_model(_torchvision_model_name)


def create_model(
    model_name: str,
    num_classes: int = len(PAPER_CLASSES),
    pretrained: bool = False,
    dropout: float = 0.35,
) -> nn.Module:
    model_name = model_name.lower()
    try:
        builder = MODEL_REGISTRY.get(model_name)
    except KeyError as exc:
        choices = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {choices}") from exc
    return builder(num_classes=num_classes, pretrained=pretrained, dropout=dropout)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location=map_location)
    config = checkpoint["config"]
    labels = tuple(checkpoint.get("labels", PAPER_CLASSES))
    model = create_model(
        model_name=config["model"],
        num_classes=len(labels),
        pretrained=False,
        dropout=float(config.get("dropout", 0.35)),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint
