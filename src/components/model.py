import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    """Simple CNN built from scratch — 4 conv blocks + classifier head."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_resnet18(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """
    Loads an ImageNet-pretrained ResNet18 and swaps the final layer for our
    number of classes. When freeze_backbone=True, only the new final layer
    is trainable — call unfreeze_all() later to fine-tune the whole network.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def unfreeze_all(model: nn.Module) -> nn.Module:
    """Unfreezes every parameter in the model — used for the fine-tuning phase."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def build_model(model_type: str, num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """Factory — returns the requested architecture by name from config."""
    if model_type == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    elif model_type == "resnet18":
        return build_resnet18(num_classes=num_classes, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Use 'baseline_cnn' or 'resnet18'.")
