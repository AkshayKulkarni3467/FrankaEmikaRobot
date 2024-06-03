import argparse
import datetime
import gymnasium as gym
import numpy as np
import itertools
import torch
import panda_gym
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory






env_name = 'PandaReachDense-v3'
# Environment
env = gym.make(env_name,render_mode = 'human')

ag_size = env.observation_space['achieved_goal'].shape[0]
dg_size = env.observation_space['desired_goal'].shape[0]
obs_size =env.observation_space['observation'].shape[0]


# Agent
agent = SAC(obs_size+dg_size, env.action_space)
# agent.load_checkpoint(ckpt_path='final_checkpoints/sac_checkpoint_humanoid_final')
#Tesnorboard
writer = SummaryWriter()


# Memory
replay_size = 100000
memory = ReplayMemory(capacity=replay_size,her_probability=0.8)

# Training Loop
total_numsteps = 0
numsteps = 1000001
updates = 0
batch_size = 256
updates_per_step =1
start_steps = 1000
eval1 = True
max_timesteps = 1000

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()[0]
    time_steps = 0
    while not done and time_steps<max_timesteps:
        desired_state = np.hstack([state['observation'], state['desired_goal']])
        achieved_state = np.hstack([state['observation'], state['achieved_goal']])
        time_steps+=1
        if start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(desired_state)  # Sample action from policy

        if len(memory.her_buffer) and len(memory.buffer) > batch_size:
            # Number of updates per step in environment
            for i in range(updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                updates += 1

        next_state, reward, terminated,truncated, info = env.step(action) # Step
        done = terminated or truncated
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        next_desired_state = np.hstack([next_state['observation'], next_state['desired_goal']])
        next_achieved_state = np.hstack([next_state['observation'], next_state['achieved_goal']])
        memory.push(desired_state, action, reward, next_desired_state, mask) 

        reward = env.compute_reward(next_state['desired_goal'], next_state['achieved_goal'], info)
        exp = (achieved_state, action, reward, done, next_achieved_state)
        memory.push(achieved_state, action, reward, next_achieved_state,done, her=True)
        state = next_state

    if total_numsteps > numsteps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    if i_episode % 100 == 0:
        agent.save_checkpoint(env_name='sacher')

    if i_episode % 10 == 0 and eval1 is True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()[0]
            episode_reward = 0
            done = False
            time_steps1 = 0
            while not done and time_steps1<max_timesteps:
                time_steps1+=1
                desired_state = np.hstack([state['observation'], state['desired_goal']])
                action = agent.select_action(desired_state, evaluate=True)

                next_state, reward, terminated, truncated,_ = env.step(action)
                done = terminated or truncated
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()

