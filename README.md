
# 

<h1 align="center">
  <br>
  Lunar Lander - DQN
  <br>
</h1>



<p align="center">
  <img src="docs/assets/lunar_lander.gif" alt="animated" />
</p>

<h4 align="center">A simple PyTorch implementation of the Deep Q-Learning algorithm to solve <a href="https://gymnasium.farama.org/environments/box2d/lunar_lander/">Lunar Lander environment from Gymnasium</a>.</h4>

******

<p align="center">AI Course Project - UniCT, DMI.</p>

## Pretrained models

| Replay Memory Capacity | Target-network sync-rate |                  Pre-trained models folder                   |
| :--------------------: | :----------------------: | :----------------------------------------------------------: |
|         500000         |            10            | [link](https://github.com/LemuelPuglisi/Lunar-Lander-DQN/tree/main/models/ll-cap_500000_sr_10) |
|         50000          |            10            | [link](https://github.com/LemuelPuglisi/Lunar-Lander-DQN/tree/main/models/ll-cap_50000_sr_10) |
|          5000          |            10            | [link](https://github.com/LemuelPuglisi/Lunar-Lander-DQN/tree/main/models/ll-cap_5000_sr_10) |
|         500000         |            1             | [link](https://github.com/LemuelPuglisi/Lunar-Lander-DQN/tree/main/models/ll-cap_500000_sr_1) |
|         500000         |           100            | [link](https://github.com/LemuelPuglisi/Lunar-Lander-DQN/tree/main/models/ll-cap_500000_sr_100) |

## Installation

Download and install `Anaconda` or `miniconda` from the [official website](https://www.anaconda.com/products/distribution). Make it works by opening a shell and running:

```bash
$ conda env list
```

This should prompt the list of your conda environments. Now create a new environment: 

```bash
$ conda create -n dqn python=3.9
```

And activate the new env:

```bash
$ conda activate dqn
```

Finally, clone the repository and install the `aicourse` module running the following inside the project folder:

```bash
$ pip install -e .
```

## Run Locally

After installing the project, start the training using `train.py` script: 

```bash
(your_env) <Lunar-Lander-DQN> python train.py --help

usage: train.py [-h] [--dest DEST] [--epochs EPOCHS] [--episodes EPISODES] [--batch-size BATCH_SIZE] [--capacity CAPACITY]
                [--sync-rate SYNC_RATE]

optional arguments:
  -h, --help                show this help message and exit
  --dest DEST               destination folder
  --epochs EPOCHS           number of epochs
  --episodes EPISODES       number of episodes to play in an epoch
  --batch-size BATCH_SIZE   batch size on trainin phase
  --capacity CAPACITY       capacity of the replay memory
  --sync-rate SYNC_RATE     sync rate of the target network
```

Visually validate the agent using the `play.py` script: 

```bash
(your_env) <Lunar-Lander-DQN> python play.py --help

usage: play.py [-h] [--model-ckpt MODEL_CKPT] [--episodes EPISODES]

optional arguments:
  -h, --help            show this help message and exit
  --model-ckpt MODEL_CKPT
  --episodes EPISODES
```

## Authors

- [@lemuelpuglisi](https://www.github.com/lemuelpuglisi)