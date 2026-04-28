import os
import sys

import torch
print(f"PyTorch: {torch.__version__}")

import intel_extension_for_pytorch as ipex
print(f"IPEX: {ipex.__version__}")

# Add Intel runtime libraries to PATH
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.environ['PATH'] = f"{torch_lib};" + os.environ['PATH']

print("PATH updated")
print(f"Has XPU: {hasattr(torch, 'xpu')}")

if hasattr(torch, 'xpu'):
    print(f"XPU available: {torch.xpu.is_available()}")
    if torch.xpu.is_available():
        print(f"XPU device count: {torch.xpu.device_count()}")
        for i in range(torch.xpu.device_count()):
            print(f"XPU device {i}: {torch.xpu.get_device_name(i)}")
        print(f"Current XPU device: {torch.xpu.current_device()}")
else:
    print("XPU is not available in this PyTorch build")
