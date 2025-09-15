import os
import numpy as np
import torch

def main():
    npz_path = "sample3.npz"
    data = np.load(npz_path)
    data_copy = {k: v.copy() for k, v in data.items()}
    data_copy['local_rot_quat'][:]=0
    data_copy['root_trans'][:]=0
    np.savez("sample3_zero.npz", **data_copy)
if __name__ == "__main__":
    main()
