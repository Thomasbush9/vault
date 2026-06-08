In a container we install:
- Torch + transformer engine -> built on cu128
- cuDNN-> libcudnn.so
- CUDA runtime + toolkit 

Host has: 
- libcuda.so nvidia-smi=> user space driver
- Kernel module 
- GPU hardware 
Driver: it's what talks with the silicon, libcuda.so is part of the driver not of CUDA
CUDA: it's a capability label, it means that our driver can support CUDA runtimes up to 12.9-> it's a ceiling. 

Cuda toolkit: libcudart.so (runtime your code calls), nvcc (compiler) 

