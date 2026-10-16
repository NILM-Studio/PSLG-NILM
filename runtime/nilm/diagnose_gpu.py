import ctypes
import glob
import json
import os
import socket
import subprocess
import torch

driver = ctypes.CDLL('libcuda.so.1')
init = driver.cuInit(0)
count = ctypes.c_int()
count_result = driver.cuDeviceGetCount(ctypes.byref(count))
print(json.dumps({'host': socket.gethostname(), 'env': {k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','SLURM_JOB_GPUS','SLURM_STEP_GPUS','LD_LIBRARY_PATH']}, 'cuInit':init,'cuDeviceGetCount':count_result,'count':count.value,'torch_cuda':torch.cuda.is_available()}, indent=2))
subprocess.run(['ls','-l',*glob.glob('/dev/nvidia*')])
