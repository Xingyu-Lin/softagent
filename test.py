import numpy as np
import torch
print(torch.__version__)
for i in range(1000):
    gpu_arr = torch.randn([2, 800], dtype=torch.float64).cuda()
    # success = (gpu_arr[1].sum() == gpu_arr.sum(dim=1)[1])
    success = torch.allclose(gpu_arr[1].sum(), gpu_arr.sum(dim=1)[1], atol=1e-7)
    if not success:
        print("Test failed!")
        print(gpu_arr[1].sum(), gpu_arr.sum(dim=1)[1])
        exit()
print("Test passes!")
