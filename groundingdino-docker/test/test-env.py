import torch
import os
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


print("env")
print(os.environ["CUDA_HOME"])
print("torch")
print(torch.cuda.is_available())
print("home")
print(CUDA_HOME)
print("end")
import time
time.sleep(10)
