import gym
from gym import spaces
import dreamerv2.api as dv2
import numpy as np

import os
os.putenv('SDL_VIDEODRIVER', 'dummy')
import pygame
pygame.display.init()
from datetime import datetime

import wandb

from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

IBEX = True if os.path.exists("/ibex/scratch/camaral/") else False

class StateToImgWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
        self.observation_shape = (400,600,3)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = 255*np.ones(self.observation_shape),
                                            dtype = np.uint8)
        self.env.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = 255*np.ones(self.observation_shape),
                                            dtype = np.uint8)
        
        # self.env.screen = "something" # prevent creation of new window
        
        
        
    def observation(self, obs):
        obs = self.env.render(mode="rgb_array")#/255.
        return obs


if __name__=="__main__":
    env = gym.make("LunarLander-v2")
    # env = StateToImgWrapper(env, img_sz=(60,40))
    env = StateToImgWrapper(env)
    env.reset()

    scratch_dir = "/ibex/scratch/camaral/dreamerv2/" if IBEX \
        else "/home/camaral/scratch/dreamerv2/"
    # config = dv2.defaults.update( dv2.configs["dmc_vision"] )
    config = dv2.defaults.update( dv2.configs["atari"] )
    print(config)

    run = wandb.init(entity="lucascamara", project="PhyRL_Dreamerv2", config=config)
    save_path = scratch_dir + run.id
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config = config.update({
        'logdir': save_path,
        # 'discount': 0.999,
    })

    dv2.train(env, config)

    run.finish()