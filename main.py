import torch
from torch.nn import functional as F

import vae.encoder as encoder


def main():
    betas_start = 0.0
    betas_end = 10.0
    num_training_steps = 10
    arr = torch.linspace(betas_start ** 0.5, betas_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
    arr2 = torch.linspace(betas_start, betas_end, num_training_steps, dtype=torch.float32)
    print(arr)
    print(arr2)
    print(arr == arr2)

    for i in range(num_training_steps):
        # print(f"i: {i}, arr[i]: {arr[i]}, arr2[i]: {arr2[i]}, arr[i] == arr2[i]: {arr[i] == arr2[i]}")
        print(f"({i},{arr[i]})", end=", ")
    print()
    for i in range(num_training_steps):
        print(f"({i},{arr2[i]})", end=", ")
if __name__ == "__main__":
    main()
