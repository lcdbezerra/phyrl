import torch
import torch.nn as nn

def init_params(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
        
def init_sindy(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.ones_(m.weight)
        
if __name__ == "__main__":
    enc = nn.Sequential(nn.Linear(10,5), nn.ReLU(), nn.Linear(5,20))
    enc.apply(init_params)
    breakpoint()