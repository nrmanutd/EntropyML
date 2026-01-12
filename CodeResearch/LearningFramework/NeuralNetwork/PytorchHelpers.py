import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, nFeatures, nClasses, dense):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(nFeatures, dense),
            nn.ReLU(),
            nn.Linear(dense, dense),
            nn.ReLU(),
            nn.Linear(dense, nClasses)
        )

    def forward(self, x):
        return self.layers(x)

# =========================
# MNIST (1x28x28)
# =========================

class MNISTTargetNet(nn.Module):
    """
    Target-модель для MNIST: небольшой, но "нормальный" ConvNet.
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 28 -> 14
            nn.Dropout(dropout),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 14 -> 7
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class MNISTScoringNet(nn.Module):
    """
    Scoring-модель для MNIST: сильно уменьшенная (быстрее для easiness/diversity).
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 28 -> 14
            nn.Dropout(dropout),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# =========================
# CIFAR ResNet blocks (3x32x32)
# =========================

def _conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)

class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = _conv3x3(out_ch, out_ch, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            x = self.down(x)
        out = F.relu(out + x, inplace=True)
        return out


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes: int, width_mult: float = 1.0):
        super().__init__()
        base = int(64 * width_mult)
        self.in_ch = base

        self.conv1 = nn.Conv2d(3, base, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base)

        self.layer1 = self._make_layer(base,   blocks=2, stride=1)
        self.layer2 = self._make_layer(base*2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base*4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base*8, blocks=2, stride=2)

        # Ключевое: единый интерфейс
        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(inplace=True),
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(base*8, num_classes)

    def _make_layer(self, out_ch, blocks, stride):
        layers = [_BasicBlock(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.in_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# =========================
# CIFAR "CIFAR-ResNet-(6n+2)" (ResNet20/32/56) — хорошая scoring-сеть
# =========================

class CifarResNet6n2(nn.Module):
    def __init__(self, num_classes: int, n: int = 3, width_mult: float = 0.5):
        super().__init__()
        base = int(16 * width_mult)
        base = max(base, 8)
        self.in_ch = base

        self.conv1 = _conv3x3(3, base, 1)
        self.bn1 = nn.BatchNorm2d(base)

        self.layer1 = self._make_layer(base,   blocks=n, stride=1)
        self.layer2 = self._make_layer(base*2, blocks=n, stride=2)
        self.layer3 = self._make_layer(base*4, blocks=n, stride=2)

        # Ключевое: единый интерфейс
        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(inplace=True),
            self.layer1,
            self.layer2,
            self.layer3,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(base*4, num_classes)

    def _make_layer(self, out_ch, blocks, stride):
        layers = [_BasicBlock(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.in_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# =========================
# Dataset-specific wrappers (как ты просил: отдельные классы для каждого датасета)
# =========================

class CIFAR10TargetNet(ResNet18CIFAR):
    def __init__(self, num_classes: int = 10):
        super().__init__(num_classes=num_classes, width_mult=1.0)

class CIFAR10ScoringNet(CifarResNet6n2):
    """
    Scoring для CIFAR-10: ResNet20 (n=3), узкий.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__(num_classes=num_classes, n=3, width_mult=0.5)

class CIFAR100TargetNet(ResNet18CIFAR):
    """
    Target для CIFAR-100: ResNet18 CIFAR-style.
    Можно сделать width_mult=1.0 (стандарт) или 1.5 (если хочешь сильнее).
    """
    def __init__(self, num_classes: int = 100):
        super().__init__(num_classes=num_classes, width_mult=1.0)

class CIFAR100ScoringNet(CifarResNet6n2):
    """
    Scoring для CIFAR-100: чуть "сильнее" чем для CIFAR-10, но всё ещё лёгкая.
    """
    def __init__(self, num_classes: int = 100):
        super().__init__(num_classes=num_classes, n=3, width_mult=0.75)


# =========================
# (опционально) helper: число параметров
# =========================
def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)