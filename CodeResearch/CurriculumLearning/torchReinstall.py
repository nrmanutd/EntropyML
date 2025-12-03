import subprocess
import sys

print("Установка PyTorch с GPU...")

# Удаление
subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"])

# Установка
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "torch==2.7.1+cu128",
    "torchvision==0.22.1+cu128",
    "torchaudio==2.7.1+cu128",
    "--index-url", "https://download.pytorch.org/whl/cu128"
])

# Проверка
import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")