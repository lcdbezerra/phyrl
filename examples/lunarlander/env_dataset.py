from gc import callbacks
import string
import gym
from gym import spaces
from PIL import Image
import numpy as np
import pysindy as ps
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from tqdm.auto import tqdm
from stable_baselines3.common.env_checker import check_env

class StateToImgWrapper(gym.ObservationWrapper):
    def __init__(self, env, img_sz):
        super().__init__(env)
        self.env = env
        self.img_sz = img_sz
        assert(type(img_sz) is tuple)
        assert(img_sz[0] >= img_sz[1])
        # IMG_SZ: Width x Height
        
        # self.observation_shape = (400,600,3)
        self.observation_shape = (*img_sz,3)
        self.action_space = env.action_space
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape),
                                            dtype = np.uint8)
        self.env.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape),
                                            dtype = np.uint8)
        
        
    def observation(self, obs):
        obs = self.env.render(mode="rgb_array")
        # obs = obs.transpose((1,0,2))
        img = Image.fromarray(obs)
        img = img.resize(self.img_sz, Image.BILINEAR)
        return np.asarray(img).transpose((1,0,2))/255.
        # return obs
        
        # next_state, reward, done, info = self.env.step(action)
        # next_state = self.env.render(mode="rgb_array")/255


def sample_from_env(steps):
    steps = int(steps)
    
    env = gym.make("LunarLander-v2")
    env = StateToImgWrapper(env)  
    
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    
    ds_obs = np.zeros((steps,*obs_shape))
    ds_rwd = np.zeros((steps,1))
    ds_act = np.zeros((steps,*act_shape))
    
    s = 0
    obs = env.reset()
    done = False
    
    while s < steps:
        act = env.action_space.sample()
        
        ds_obs[s] = obs
        ds_act[s] = act
        
        obs, reward, done, _  = env.step(act)
        
        ds_rwd[s] = reward
        
        if done:
            obs = env.reset()
        s += 1
        
    return ds_obs, ds_act, ds_rwd

#####################################################################################
### COLLECT DATA FROM AGENT TRAINING

class TqdmCallback(BaseCallback):
    def __init__(self, update_every):
        super().__init__()
        self.progress_bar = None
        self.update_every = update_every
    
    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.locals['total_timesteps'])
    
    def _on_step(self):
        self.progress_bar.update(self.update_every)
        return True

    def _on_training_end(self):
        self.progress_bar.close()
        self.progress_bar = None

class ExperienceDatasetWrapper(gym.Wrapper):
    def __init__(self, env, steps):
        super(ExperienceDatasetWrapper, self).__init__(env)
        self.env = env
        self.steps = steps

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.obs_shape = env.observation_space.shape
        self.act_shape = env.action_space.shape

        self.ds_obs = np.zeros((steps,*self.obs_shape))
        self.ds_rwd = np.zeros((steps,))
        self.ds_act = np.zeros((steps,*self.act_shape))
        self.ds_don = np.zeros((steps,), bool)
        self.step_count = 0

    def reset(self):
        obs = self.env.reset()
        if self.step_count < self.steps:
            self.ds_obs[self.step_count] = obs
        # This will overwrite the last observation of an episode
        return obs

    def step(self, act):
        if self.step_count >= self.steps:
            # raise RuntimeError(f"Dataset full. Step count: {self.step_count}")
            # print(f"Dataset full. Step count: {self.step_count}")
            self.step_count += 1
            return self.env.step(act)


        obs, rwd, don, info = self.env.step(act)

        self.ds_rwd[self.step_count] = rwd
        self.ds_act[self.step_count] = act
        self.ds_don[self.step_count] = don
        self.step_count += 1
        if self.step_count < self.steps:
            self.ds_obs[self.step_count] = obs
        
        return obs, rwd, don, info

    def get_dataset(self):
        # assert self.step_count == self.steps, self.step_count
        # return self.ds_obs, self.ds_act, self.ds_rwd, self.ds_don, self.step_count
        return self.ds_obs, self.ds_act, self.ds_rwd, self.ds_don

class JointExperienceDatasetWrapper(gym.Wrapper):
    def __init__(self, env, img_sz, steps):
        super(JointExperienceDatasetWrapper, self).__init__(env)
        self.env = env
        self.steps = steps
        self.img_sz = img_sz
        assert(type(img_sz) is tuple)
        assert(img_sz[0] >= img_sz[1])

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.obs_shape = env.observation_space.shape
        self.act_shape = env.action_space.shape

        self.ds_obs = np.zeros((steps,*img_sz, 3))
        self.ds_rwd = np.zeros((steps,))
        self.ds_act = np.zeros((steps,*self.act_shape))
        self.ds_don = np.zeros((steps,), bool)
        self.step_count = 0

    def get_img(self):
        img = self.env.render(mode="rgb_array")
        # obs = obs.transpose((1,0,2))
        img = Image.fromarray(img)
        img = img.resize(self.img_sz, Image.BILINEAR)
        return np.asarray(img).transpose((1,0,2))/255.

    def reset(self):
        obs = self.env.reset()
        if self.step_count < self.steps:
            self.ds_obs[self.step_count] = self.get_img()
        # This will overwrite the last observation of an episode
        return obs

    def step(self, act):
        if self.step_count >= self.steps:
            # raise RuntimeError(f"Dataset full. Step count: {self.step_count}")
            # print(f"Dataset full. Step count: {self.step_count}")
            self.step_count += 1
            return self.env.step(act)

        obs, rwd, don, info = self.env.step(act)

        self.ds_rwd[self.step_count] = rwd
        self.ds_act[self.step_count] = act
        self.ds_don[self.step_count] = don
        self.step_count += 1
        if self.step_count < self.steps:
            self.ds_obs[self.step_count] = self.get_img()
        
        return obs, rwd, don, info

    def get_dataset(self):
        # assert self.step_count == self.steps, self.step_count
        # return self.ds_obs, self.ds_act, self.ds_rwd, self.ds_don, self.step_count
        return self.ds_obs, self.ds_act, self.ds_rwd, self.ds_don

def sample_from_env_under_training(steps):
    steps = int(steps)
    
    env = gym.make("LunarLander-v2")
    # env = StateToImgWrapper(env, img_sz=(60,40))
    # env = ExperienceDatasetWrapper(env, steps)
    env = JointExperienceDatasetWrapper(env, img_sz=(60,40), steps=steps)
    # check_env(env)

    s = 0
    obs = env.reset()
    done = False

    # cb = TqdmCallback()
    update_every = 100
    cb = EveryNTimesteps(update_every, TqdmCallback(update_every))
    
    model = sb3.PPO("MlpPolicy", env)
    # model = sb3.PPO("CnnPolicy", env)
    model.learn(total_timesteps=steps, callback=cb)

    return env.get_dataset()
######################################################################

# def estimate_derivatives(x):
    
#     dx = ps.SmoothedFiniteDifference(axis=0, order=4, d=1)._differentiate(x,1)
#     ddx = ps.SmoothedFiniteDifference(axis=0, order=4, d=2)._differentiate(x,1)
    
#     return x, dx, ddx

def derivatives(x, order=4):
    d  = ps.SmoothedFiniteDifference(axis=0, order=order, d=1, is_uniform=True)._differentiate(x,1)
    dd = ps.SmoothedFiniteDifference(axis=0, order=order, d=2, is_uniform=True)._differentiate(x,1)
    return d,dd

def estimate_derivatives(obs, done):

    ep_ends = np.argwhere(done).reshape((-1,))
    ep_ends = list(ep_ends.reshape((-1,)) + 1)
    eps, ds, dds = [], [], []

    ep_obs = obs[:ep_ends[0]]
    eps.append(ep_obs)
    d, dd = derivatives(ep_obs)
    ds.append(d)
    dds.append(dd)

    for i in range(len(ep_ends)-1):
        ep_obs = obs[ep_ends[i] : ep_ends[i+1]]
        eps.append(ep_obs)
        d, dd = derivatives(ep_obs)
        ds.append(d)
        dds.append(dd)
    
    ep_obs = obs[ep_ends[-1]:]
    if ep_obs.shape[0] > 10: # not sure why, but this is needed
        eps.append(ep_obs)
        d, dd = derivatives(ep_obs)
        ds.append(d)
        dds.append(dd)

    eps = np.concatenate(eps, axis=0)
    ds  = np.concatenate(ds,  axis=0)
    dds = np.concatenate(dds, axis=0)
    
    return eps, ds, dds


def get_dataset(steps):

    ds_obs, ds_act, ds_rwd, ds_don = sample_from_env_under_training(steps)
    x, dx, ddx = estimate_derivatives(ds_obs, ds_don)

    # Sometimes not all derivatives can be computed
    L = dx.shape[0]
    ds_act = ds_act[:L]
    ds_rwd = ds_rwd[:L]
    ds_don = ds_don[:L]

    ds_act = ds_act.reshape((-1,1))
    
    # One-hot action representation
    act = np.zeros((ds_act.shape[0], 4), int)
    ds_act = ds_act.reshape((-1,))
    for i in range(4):
        act[ds_act==i, i] = 1
    
    return x, dx, ddx, act, ds_rwd, ds_don

def save_data(path, *data):
    np.savez(path, *data)

def load_data(path):
    f = np.load(path)
    files = (f["arr_0"], f["arr_1"], f["arr_2"], f["arr_3"], f["arr_4"], f["arr_5"])
    f.close()
    return files

def fix_derivatives(path):
    # Recompute derivatives by separating episodes
    d = load_data(path)
    x, dx, ddx = estimate_derivatives(d[0], d[-1]) # obs, done
    d = d[0], dx, ddx, *d[3:]
    L = dx.shape[0]
    d = tuple([x[:L] for x in d])
    save_data("dataset_fix_img.npz", *d)
    # save_data("dataset_fix_gstate.npz", *d)

def train_test_split(data):
    if type(data)==str:
        d = load_data(data)
    else:
        d = data
    
    done = d[-1]
    ep = np.argwhere(done).reshape((-1,))
    ep = list(ep.reshape((-1,)) + 1)

    inds = np.arange(len(done))
    inds = [inds[0:ep[0]], *[ inds[ep[i]:ep[i+1]] for i in range(len(ep)-1) ] , inds[ep[-1]:] ]
    # for i in range(len(ep)+1):
    #     print(len(inds[i]), inds[i])
    
    # Shuffle episodes
    ratio = .8
    L = int(np.ceil(ratio*len(ep)))
    ep_inds = np.arange(len(ep))
    np.random.shuffle(ep_inds)
    # Split between train and test episodes
    train_ep = ep_inds[:L]
    train_inds = np.concatenate([inds[i] for i in train_ep])

    test_ep = ep_inds[L:]
    test_inds = np.concatenate([inds[i] for i in test_ep])
    
    train_d = [d[i][train_inds] for i in range(len(d))]
    test_d  = [d[i][test_inds] for i in range(len(d))]

    print("TRAIN SHAPES")
    for i in range(len(d)):
        print(train_d[i].shape)
    print("TEST SHAPES")
    for i in range(len(d)):
        print(test_d[i].shape)

    if type(data) == str:
        save_data( data.split(".")[0] + "_train.npz", *train_d)
        save_data( data.split(".")[0] + "_test.npz", *test_d)
    return train_d, test_d
    
def join_dataset(data1,data2):
    raise NotImplementedError("Join dataset")

if __name__ == "__main__":
    import sys
    assert len(sys.argv)==2, "Usage: python env_dataset.py FILENAME.npz"
    filename = sys.argv[1]

    # os.putenv('SDL_VIDEODRIVER', 'dummy')
    # import pygame
    # pygame.display.init()
    

    # x, dx, ddx, u, r, d = load_data("/home/camaral/scratch/phyrl/dataset/dataset_trained_gstate.npz")
    # breakpoint()

    d = get_dataset(1e5)
    # print("Save to? ______.npz")
    # filename = input()+".npz"
    save_data(filename, *d)
    train_test_split(filename)

    # x, dx, ddx, u, r, d = load_data(filename)
    # d = np.logical_not(d).astype(int)
    # d = np.eye(2)[d]
    # breakpoint()

    # train_test_split(filename)

    # fix_derivatives(filename)
    
    # d = get_dataset(1e4)
    # save_data("dataset_test_cnn.npz", *d)