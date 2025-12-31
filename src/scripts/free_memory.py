import torch
import gc

def print_gpu_memory(tag=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved  = torch.cuda.memory_reserved() / 1024**2
    print(f"{tag} | Allocated: {allocated:.2f} MB | Reserved (cache): {reserved:.2f} MB")

# BEFORE cleanup
print_gpu_memory("Before cleanup")

# Free objects
# del model
# del optimizer
# del loss
gc.collect()
torch.cuda.empty_cache()

# AFTER cleanup
print_gpu_memory("After cleanup")
