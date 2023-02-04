import os
import argparse

import torch
import gymnasium as gym

from aicourse.dqn import QFunc
from aicourse.agent import LunarLanderAgent

parser = argparse.ArgumentParser()
parser.add_argument('--model-ckpt', type=str)
parser.add_argument('--episodes', type=int, default=1)
args = parser.parse_args()

episodes = args.episodes
model_checkpoints = args.model_ckpt
assert os.path.exists(model_checkpoints), 'Invalid model checkpoints'

env = gym.make("LunarLander-v2", render_mode="human")   

qfunc = QFunc(4, 8, 512, 512).to('cuda')
qfunc.load_state_dict(torch.load(model_checkpoints))
agent = LunarLanderAgent(qfunc, env)

for _ in range(episodes): 
    agent.play_episode([], True)