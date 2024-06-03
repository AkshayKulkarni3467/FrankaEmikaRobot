import gymnasium as gym
import panda_gym
from sac import SAC
import numpy as np
import time

env_name = 'PandaReachDense-v3'
# Environment
env = gym.make(env_name,render_mode = 'human')

ag_size = env.observation_space['achieved_goal'].shape[0]
dg_size = env.observation_space['desired_goal'].shape[0]
obs_size =env.observation_space['observation'].shape[0]

agent = SAC(obs_size+dg_size, env.action_space)
agent.load_checkpoint('final_checkpoints/sac_checkpoint_sacher_')

episodes = 200
max_timesteps = 70

for _ in range(episodes):
    done = False
    state = env.reset()[0]
    timesteps = 0
    while not done and timesteps < max_timesteps:
        desired_state = np.hstack([state['observation'], state['desired_goal']])
        achieved_state = np.hstack([state['observation'], state['achieved_goal']])
        timesteps+=1
        
        action = agent.select_action(desired_state,evaluate=True)
        next_state, reward, terminated,truncated, info = env.step(action) 
        done = terminated or truncated
        state = next_state
        time.sleep(0.05)

env.close()