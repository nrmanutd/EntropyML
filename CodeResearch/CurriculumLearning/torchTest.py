import torch
import subprocess
import sys

print("=" * 60)
print("ДИАГНОСТИКА PYTORCH + CUDA")
print("=" * 60)

# 1. Проверка PyTorch
print(f"PyTorch version: {torch.__version__}")

# 2. Проверка CUDA в PyTorch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (in PyTorch): {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")

# 3. Проверка через nvidia-smi
print("\nПроверка NVIDIA драйверов:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ nvidia-smi работает")
        lines = result.stdout.strip().split('\n')
        for i in range(min(5, len(lines))):
            print(f"  {lines[i]}")
    else:
        print("❌ nvidia-smi не работает")
except FileNotFoundError:
    print("❌ nvidia-smi не установлен")

# 4. Проверка установленных пакетов
print("\nУстановленные CUDA пакеты PyTorch:")
import pkg_resources
for pkg in pkg_resources.working_set:
    pkg_name = pkg.key.lower()
    if 'cuda' in pkg_name or 'torch' in pkg_name:
        print(f"  {pkg.key}=={pkg.version}")