import subprocess, sys

# Check GPU via nvidia-smi
try:
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader"], capture_output=True, text=True)
    print("=== nvidia-smi ===")
    print(r.stdout.strip())
except Exception as e:
    print("nvidia-smi error:", e)

# Check TF with GPU
print("\n=== TensorFlow ===")
import tensorflow as tf
print("TF version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())

# Check JAX
print("\n=== JAX ===")
try:
    import jax
    print("JAX version:", jax.__version__)
    print("JAX devices:", jax.devices())
    print("JAX backend:", jax.default_backend())
except Exception as e:
    print("JAX error:", e)

# Check PyTorch
print("\n=== PyTorch ===")
try:
    import torch
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
except Exception as e:
    print("Torch error:", e)

# Check CUDA toolkit
print("\n=== CUDA toolkit ===")
try:
    r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    print(r.stdout.strip())
except Exception as e:
    print("nvcc error:", e)
