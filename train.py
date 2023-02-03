import os
import argparse
from copy import deepcopy
from collections import deque

import numpy as np
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter # type: ignore
from tqdm import tqdm

from aicourse.dqn import QFunc
from aicourse.agent import LunarLanderAgent
from aicourse.train import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--dest',         type=str)
parser.add_argument('--epochs',       type=int, default=50)
parser.add_argument('--episodes',     type=int, default=50)
parser.add_argument('--batch-size',   type=int, default=512)
parser.add_argument('--capacity',     type=int, default=500_000)
parser.add_argument('--sync-rate',    type=int, default=10)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dest = args.dest
assert os.path.exists(dest), 'invalid destination'

epochs = args.epochs
episodes = args.episodes
batch_size = args.batch_size
capacity = args.capacity
sync_rate = args.sync_rate

output_path = os.path.join(dest, f'll-cap_{capacity}_sr_{sync_rate}')
models_path = os.path.join(output_path, 'models')
os.makedirs(models_path)

env_train = gym.make('LunarLander-v2')

policy_net = QFunc(4, 8, 512, 512).to(device)
target_net = QFunc(4, 8, 512, 512).to(device)
target_net.load_state_dict(policy_net.state_dict())

agent = LunarLanderAgent(policy_net, env_train)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
epsilon_fun = lambda e: max(1 - (e / epochs) * 2, 0.1)
replay_buffer = deque(maxlen=capacity)

trainer = Trainer(policy_net=policy_net, 
                  target_net=target_net, 
                  batch_size=batch_size, 
                  optimizer=optimizer,
                  sync_rate=sync_rate)

writer = SummaryWriter(os.path.join(output_path, 'logs'))


current_best = None
current_best_ret = -1e10

epoch_idx = 0
for epoch_idx in tqdm(range(epochs)):

    curr_eps = epsilon_fun(epoch_idx)

    returns = [ agent.play_episode(replay_buffer, eps=curr_eps) for _ in range(episodes) ]    
    
    average_return = np.mean(returns)
    average_loss = trainer.run(replay_buffer)
                
    writer.add_scalar('train/reward',  average_return, epoch_idx)
    writer.add_scalar('train/loss',    average_loss, epoch_idx)
    writer.add_scalar('train/epsilon', curr_eps, epoch_idx)

    if average_return > current_best_ret:
        current_best_ret = average_return
        current_best = deepcopy(policy_net.state_dict()) 

    with torch.no_grad():
        total_reward = agent.play_episode([], validation=True)
        writer.add_scalar('valid/reward', total_reward, epoch_idx)

    if epoch_idx % 5 == 0:
        if current_best is None: current_best = policy_net.state_dict()
        torch.save(current_best, os.path.join(models_path, f'll-ep{epoch_idx}.ckpt'))        
        current_best = None
        current_best_ret = -1e10
        
last_idx = epoch_idx + 1
torch.save(current_best, os.path.join(models_path, f'll-ep{last_idx}.ckpt'))        

print('training finished.')