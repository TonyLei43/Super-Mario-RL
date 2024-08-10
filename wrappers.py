import numpy as np
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip # number of frames to skip
    
    def step(self, action):
        #================================================#
        # Accumulate the rewards for 4 frames into a total reward 
        # Parameters:
        # - action: an action the agent will take
        # Return:
        # - next_state : next state of the agent
        # - total_rewardL: the total accumulated reward for 4 frames
        # - done: flag for when agent reaches terminal state
        # - trunc: flag for whether the truncation condition is met
        # - info: info for debugging
        # HINT: use a for loop to accuulate the rewards. Use the built in 
        # env.step(action) function! Check the documentation to see what it does
        #=================================================#
        pass

        return next_state, total_reward, done, trunc, info
    

    

def apply_wrappers(env):
    env = SkipFrame(env) # Num of frames to apply one action to (! NEEDS A PARAMETER)
    env = ResizeObservation(env) # Resize frame from 240x256 to 84x84 (! NEEDS A PARAMTER)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True) # May need to change lz4_compress to False if issues arise
    return env
