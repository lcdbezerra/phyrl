import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from envs import *
from datetime import datetime
from d3rlpy.dataset import MDPDataset


def build_dataset(env, filename=None, **kwargs):
    if filename is None:
        N_eps = int(1e3)
        steps_per_eps = 50
        # env = SimplePhyEnv(x0=np.random.randn(4,1)*2)
        
        dataset = collect_data(env, N_eps, steps_per_eps)
        dt = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        filename = "data/Dataset_"+dt+".h5"
        dataset.dump(filename)
    else:
        dataset = MDPDataset.load(filename)
        
    # Split episodes in N_frames samples
    L = 5
    S = 5
    
    obs = np.concatenate(tuple([torch.from_numpy(e.observations).unfold(0,L,S).numpy().transpose(0,2,1)
           for e in dataset.episodes]))
    act = np.concatenate(tuple([torch.from_numpy(e.actions).unfold(0,L,S).numpy().transpose(0,2,1)
           for e in dataset.episodes]))
    rwd = np.concatenate(tuple([torch.from_numpy(e.rewards).unfold(0,L,S).numpy()
           for e in dataset.episodes]))
    
    return obs, act, rwd
    
def build_dataloader(obs,act,rwd):
    ds = TensorDataset(torch.Tensor(obs), torch.Tensor(act), torch.Tensor(rwd))
    # Split between training and test
    L = len(ds)
    trainL = int(np.floor(.8*L))
    train_ds, test_ds = random_split(ds, [trainL, L-trainL]) # later define rng generator!
    # Create dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    test_dataloader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
    
    return train_dataloader, test_dataloader
        
def collect_data(env, N_eps, steps_per_eps):
    observations = np.zeros((N_eps*steps_per_eps, env.observation_space.shape[0]))
    actions = np.zeros((N_eps*steps_per_eps, env.action_space.shape[0]))
    rewards = np.zeros((N_eps*steps_per_eps,))
    terminals = np.zeros((N_eps*steps_per_eps,))

    for e in range(N_eps):
        x0 = np.random.randn(4,1)*2 # std-dev=2
        x = env.reset(x0=x0)
        for s in range(steps_per_eps):
            ind = e*steps_per_eps + s
            observations[ind,:] = x.T

            a = env.action_space.sample()
            x, r, _, _  = env.step(a)

            actions[ind,:] = a.T
            rewards[ind] = r
        terminals[ind] = 1
        
    return MDPDataset(observations, actions, rewards, terminals)