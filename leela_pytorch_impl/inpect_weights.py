
import os
import torch


PATH = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(PATH, "lc0_weights_791556.pt")


model_state_dict = torch.load(WEIGHTS)
for name, tensor in model_state_dict.items():
    print(f"{name}: {tensor.shape} dtype={tensor.dtype}")

