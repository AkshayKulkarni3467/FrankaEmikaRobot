# Franka Emika Robot with SAC+HER

## Overview:
This project trains the franka emika robot using SAC + HER algorithm.

## Overview of algorithm:
- Humans have the ability to learn as much from achieving an undesired outcome as from the desired one.
- HER allows the algorithm to perform exactly this kind of reasoning and can be combined with any off-policy RL algorithm.
- The idea behind HER is to replay each episode with a different goal than the one the agent was trying to achieve.

## Before Training GIF:


## After Training GIF:



## Loss and Reward curves:
- Actor Loss
  
![Actor](https://github.com/AkshayKulkarni3467/FrankaEmikaRobot/assets/129979542/80f37368-545a-4f5e-bbb3-758e3606f11a)


- Critic Loss

![Critic](https://github.com/AkshayKulkarni3467/FrankaEmikaRobot/assets/129979542/c52306ee-4396-45cf-9cce-6333a7af4e61)


- Reward Curve

![Reward](https://github.com/AkshayKulkarni3467/FrankaEmikaRobot/assets/129979542/c5d7a351-1fef-48ea-8f97-00df003b60bc)

