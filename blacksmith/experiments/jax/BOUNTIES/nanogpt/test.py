import numpy as np
import torch
import torch.optim as optim
from model import GPT as GPT_PT, GPTConfig as Config_PT
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


def main():
    use_tt = True
    if use_tt:
        xr.set_device_type("TT")
        device = torch_xla.device()
        xr.set_device_type("TT")
        print("device: ", device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    a_pt = torch.randn((4, 512, 512), device=device)
    b_pt = torch.randn((4, 512, 512), device=device)
    c = a_pt @ b_pt.transpose(-2, -1)
    print("Result:", c)


if __name__ == "__main__":
    main()