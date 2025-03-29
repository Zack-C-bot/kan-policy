from typing import List, Dict, Optional, Optional
import numpy as np
import gym
from gym.spaces import Box
from diffusion_policy.env.kitchen.base import KitchenBase
import os
import pandas as pd

def save_tradata(data,path_id):
    id=int(path_id.split('/')[-1])
    name=path_id.split('/')[-2]
    path='/home/data/chenzikang/diffusion_policy/data/ttra/'+name
    path1='/home/data/chenzikang/diffusion_policy/data/tra/'+name
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path1):
        os.mkdir(path1)
    path=path+'/tra{}.csv'.format(id+100000)
    path1=path1+'/tra{}.csv'.format(id+100000)
    data_1=[[i] for i in data.reshape(-1, 3)]
    data=data.reshape(-1, 3)
    if os.path.exists(path):
        df = pd.DataFrame(data)
        df.to_csv(path, mode='a', header=False, index=False)
        #pass
    else:
        df = pd.DataFrame(data)
        df.to_csv(path,index=False)
    
    if os.path.exists(path1):
        df = pd.DataFrame(data_1)
        df.to_csv(path1, mode='a', header=False, index=False)
        #pass
    else:
        df = pd.DataFrame(data_1)
        df.to_csv(path1,index=False)


class KitchenLowdimWrapper(gym.Env):
    def __init__(self,
            env: KitchenBase,
            init_qpos: Optional[np.ndarray]=None,
            init_qvel: Optional[np.ndarray]=None,
            render_hw = (240,360),
            output=None
        ):
        self.env = env
        self.init_qpos = init_qpos
        self.init_qvel = init_qvel
        self.render_hw = render_hw
        self.output=output
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return self.env.observation_space

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        if self.init_qpos is not None:
            # reset anyway to be safe, not very expensive
            _ = self.env.reset()
            # start from known state
            self.env.set_state(self.init_qpos, self.init_qvel)
            obs = self.env._get_obs()
            return obs
            # obs, _, _, _ = self.env.step(np.zeros_like(
            #     self.action_space.sample()))
            # return obs
        else:
            return self.env.reset()

    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode, width=w, height=h)
    
    def step(self, a):
        _,_,_,env_info=self.env.step(a)
        
        if self.output is not None:
            save_tradata(env_info['obs_dict']['qp'][-3:],self.output)
        return self.env.step(a)
