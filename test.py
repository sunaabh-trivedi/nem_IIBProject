import torch
import numpy as np

def main():

    max_epochs = 8
    for i in range(max_epochs):
        frac = i/max_epochs

        x = (1-frac)*-5 + ((1-frac)*0.5 + frac*2.0)*torch.randn((10,))
        x = torch.exp(x)

        x = torch.clamp(x, min=0.0, max=1.0)

        print(f"Iteration {i}: {x}")

main()