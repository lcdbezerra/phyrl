import numpy as np
import gym
from gym import spaces

class SimplePhyEnv(gym.Env):
    def __init__(self, x0, timestep=1, mass=1):
        super(SimplePhyEnv,self).__init__()
        
        assert(x0.shape == (4,1))
        self.x0 = x0
        self.ts = timestep
        self.m = mass
        
        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(2,1))
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf,
                                            shape=(2,1))
        
        # Initial state
        self.reset()
        # State-space representation
        self.A = np.matrix([[0,0,1,0],
                            [0,0,0,1],
                            [0,0,0,0],
                            [0,0,0,0]])
        self.B = np.matrix([[0,0],
                            [0,0],
                            [1/mass,0],
                            [0,1/mass],])
        
    def reset(self, x0=None):
        if x0 is None:
            self.x = self.x0
        else:
            self.x = x0
        return self.x[:2]
    
    def step(self, action):
        x_dot = self.A*self.x + self.B*np.matrix(action).reshape((2,1))
        self.x = self.x + x_dot*self.ts
        
        rwd = 0
        done = False
        info = None
        return self.x[:2], rwd, done, info
    
class ImgPhyEnv(gym.Env):
    def __init__(self, x0, timestep=1, mass=1):
        super(ImgPhyEnv,self).__init__()
        
        assert(x0.shape == (4,1))
        self.x0 = x0
        self.ts = timestep
        self.m = mass
        
        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(2,1))
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(32,32))
        
        # Initial state
        self.reset()
        # State-space representation
        self.A = np.matrix([[0,0,1,0],
                            [0,0,0,1],
                            [0,0,0,0],
                            [0,0,0,0]])
        self.B = np.matrix([[0,0],
                            [0,0],
                            [1/mass,0],
                            [0,1/mass],])
        
    def reset(self, x0=None):
        if x0 is None:
            self.x = self.x0
        else:
            self.x = x0
        return self.x[:2]
    
    def step(self, action):
        x_dot = self.A*self.x + self.B*np.matrix(action).reshape((2,1))
        self.x = self.x + x_dot*self.ts
        
        rwd = 0
        done = False
        info = None
        return self.x[:2], rwd, done, info