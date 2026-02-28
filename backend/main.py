import torch
print(torch.backends.mps.is_available())

device = torch.device("mps")