import tensorflow as tf
print("TF version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs disponibles:", gpus)
print("CPU count:", tf.config.threading.get_intra_op_parallelism_threads())

# Ver también si torch tiene GPU
try:
    import torch
    print("PyTorch CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
except: pass

# Escala del entrenamiento
n_interactions = 7_762_197
batch_size = 1024
steps_per_epoch = n_interactions // batch_size
print(f"\nEscala estimada:")
print(f"  Interacciones: {n_interactions:,}")
print(f"  Batch size:    {batch_size}")
print(f"  Steps/epoch:   {steps_per_epoch:,}")
print(f"  5 épocas:      {steps_per_epoch*5:,} steps")
