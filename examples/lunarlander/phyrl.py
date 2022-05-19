from builtins import breakpoint
import sys
sys.path.append("../../src_torch")
import os
import wandb
import numpy as np
from autoencoder import *
from training import train_network
# from example_pendulum import get_pendulum_data
from env_dataset import load_data

from torch.utils.data import TensorDataset, DataLoader, random_split

IBEX = True if os.path.exists("/ibex/scratch/camaral/") else False
online = None
scratch_dir = "/ibex/scratch/camaral/runs/" if IBEX \
    else "/home/camaral/scratch/runs/"
base_dir = "/home/camaral/code/SindyAutoencoders/examples/lunarlander/"
dataset_base = "/ibex/scratch/camaral/phyrl/dataset/" if IBEX \
    else "/home/camaral/scratch/phyrl/dataset/"
# ds_name = "dataset_debug" #####
ds_name = "dataset_trained_gstate" #####
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_ds():
    # Get dataset from the Lunar Lander env.
    path_data = dataset_base+ds_name+"_train.npz"
    x, dx, ddx, u, r, d = load_data(path_data)
    d = np.logical_not(d).astype(int)
    d = np.eye(2)[d]
    r = r.reshape((-1,1))
    train_ds = TensorDataset(*[torch.Tensor(v) for v in [x,dx,ddx,u,r,d]])
    
    path_data = dataset_base+ds_name+"_test.npz"
    x, dx, ddx, u, r, d = load_data(path_data)
    d = np.logical_not(d).astype(int)
    d = np.eye(2)[d]
    r = r.reshape((-1,1))
    test_ds = TensorDataset(*[torch.Tensor(v) for v in [x,dx,ddx,u,r,d]])
    
    return train_ds, test_ds

train_ds, test_ds = get_ds()

def train(config=None):
    mode = "online" if online else "offline"
    with wandb.init(config=config, mode=mode) as run:
        config = wandb.config
        np.random.seed(config.seed)
        torch_seed = torch.Generator().manual_seed(config.seed)
        config.device = device

        train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=True)
        test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4, drop_last=True)
        
        AE = Autoencoder(config).to(device)
        wandb.watch(AE, log="all")
        
        train_network(AE, train_dl, test_dl, config)
        
        # Save network
        save_path = scratch_dir + run.id + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(AE.state_dict(), save_path+"model.h5")
        wandb.save(save_path+"model.h5")

wandb_root = "lucascamara/PhyRL/"
# sweep_id = {
#     "0": "bwtxt4mg", # REGULAR
#     "1": "zw8dz05t", # CONSTRAINED 1
#     "2": "2pbv8gut", # CONSTRAINED 2
#     "3": "y5zuf1bc", # CONSTRAINED 3
#     "4": "ju9eq13d", # CONSTRAINED 4
# }
sweep_id = {
    "4": "bwtxt4mg", # REGULAR
    "1": "zw8dz05t", # CONSTRAINED 1
    "2": "2pbv8gut", # CONSTRAINED 2
    "3": "ju9eq13d", # CONSTRAINED 4
}

if __name__ == "__main__":
    import sys
    assert len(sys.argv)==2, "Usage: python phyrl.py [Physics Constraint Index]"
    phy_constraint = sys.argv[1]
    sweep_id = wandb_root + sweep_id[phy_constraint]

    online = True #if IBEX else False
    # online = True if IBEX else False
    wandb.agent(sweep_id, train)