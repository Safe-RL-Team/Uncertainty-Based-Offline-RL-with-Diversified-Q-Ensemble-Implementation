# Inspired by:
# 1. paper for EDAC: https://arxiv.org/abs/2110.01548
# 2. official implementation: https://github.com/snu-mllab/EDAC
# 3. CORL implementation: https://github.com/tinkoff-ai/CORL
import os
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable
import pyrallis
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import wandb
from tqdm import trange
import time
import random
import d4rl
import gym



@dataclass
class TrainConfig:
    epochs: int = 200  # number of total updates
    updates_per_epoch: int = 500  # number of updates per epoch
    eval_episodes: int = 5  # number of episodes for evaluation
    batch_size: int = 2048  # batch size (per update)
    lr_actor: float = 0.0003  # learning rate for actor
    lr_critic: float = 0.0003  # learning rate for critic
    env: str = 'halfcheetah-medium-v2'  # environment name

    num_critics: int = 5  # number of critics
    beta : float = 0.1  # factor for action log probability for the actor loss
    eta: float = 1.0  # diversity loss factor
    gamma: float = 0.99  # discount factor
    tau: float = 0.005  # target network update factor

    name: str = 'edac'  # wandb name of the experiment
    group: str = 'edac'  # wandb group name
    project: str = 'edac_reimplementation'  # wandb project name
    seed: int = 0  # seed (0 for random seed)
    device: str = 'auto'  # device to use (auto, cuda or cpu)

    save_path: str = 'ckp'  # where to save the model weights and config
    save_every: int = 10  # save the model every x epochs
    continue_from: str = ''  # continue training from a checkpoint file (config has to be loaded separately)

    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.name_full = f'{self.name}-{self.env}-{time.strftime("%y%m%d-%H%M%S")}'
        if self.save_path:
            self.save_path_full = Path(self.save_path) / self.name_full


class ReplayBuffer:
    '''buffer for the offline RL dataset'''
    def __init__(self, dataset, batch_size : int, device : str):
        size = dataset["observations"].shape[0]
        if size < batch_size:
            raise ValueError(f'The batch_size ({batch_size}) cannot be larger than the dataset size ({size}).')
        self.batch_size = batch_size
        self.states, self.actions, self.rewards, self.next_states, self.dones = (
            torch.tensor(dataset[x], dtype=torch.float32, device=device)
            for x in ['observations', 'actions', 'rewards', 'next_observations', 'terminals']
        )

    def sample(self):
        idx = np.random.randint(0, len(self.states), size=self.batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


class Actor(nn.Module):
    def __init__(self, layer_sizes : list[int], max_action : float = 1.0):
        super().__init__()
        assert len(layer_sizes) >= 3, f'layer_sizes must have at least 3 elements (input, hidden, output), got {layer_sizes}'
        self.max_action = max_action
        # setup hidden layers based on the given layer sizes
        self.hidden = nn.Sequential(*(
            x for i in range(len(layer_sizes) - 2) for x in [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                nn.ReLU()
            ]
        ))
        # create output and output uncertainty layers
        self.output = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.output_uncertainty = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        # init as in the EDAC paper
        for layer in self.hidden[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)
        torch.nn.init.uniform_(self.output.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.output.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.output_uncertainty.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.output_uncertainty.bias, -1e-3, 1e-3)


    def forward(self, state : torch.Tensor):
        x_hidden = self.hidden(state)
        x_mean = self.output(x_hidden)
        x_std = torch.exp(torch.clip(self.output_uncertainty(x_hidden), -5, 2))
        policy_dist = Normal(x_mean, x_std)
        action_linear = policy_dist.rsample()
        action = torch.tanh(action_linear) * self.max_action
        action_log_prob = policy_dist.log_prob(action_linear).sum(-1)
        # TODO: maybe subtract `torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)` from action_log_prob, as in CORL
        return action, action_log_prob


class VectorCritic(nn.Module):
    def __init__(self, layer_sizes: list[int], num_critics: int):
        super().__init__()
        self.models = nn.ModuleList([
            nn.Sequential(*[
                x for i in range(len(layer_sizes) - 1) for x in [
                    nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                    nn.ReLU()
                ]
        ][:-1]) for _ in range(num_critics)
        ])
        for model in self.models:
            # init as in the EDAC paper
            for layer in model[::2]:
                torch.nn.init.constant_(layer.bias, 0.1)
            torch.nn.init.uniform_(model[-1].weight, -3e-3, 3e-3)
            torch.nn.init.uniform_(model[-1].bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return torch.cat([model(torch.cat([state, action], dim=-1)) for model in self.models], dim=-1)



def train(config: TrainConfig, display_video_callback: Callable[[list[np.array]], None] = None) -> None:
    # init env
    eval_env = gym.make(config.env)
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    d4rl_dataset = d4rl.qlearning_dataset(eval_env)
    buffer = ReplayBuffer(d4rl_dataset, config.batch_size, config.device)

    # set seed
    if config.seed == 0:
        config.seed = np.random.randint(0, 100000)
    else:
        # TODO: if a seed is given, try to be deterministic (might be slower)
        #       but does this make sense?
        # torch.use_deterministic_algorithms(True)
        pass
    eval_env.seed(config.seed)
    eval_env.action_space.seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # init model
    actor = Actor([state_dim, 256, 256, action_dim], max_action=eval_env.action_space.high[0]).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.lr_actor)
    critic = VectorCritic([state_dim + action_dim, 256, 256, 1], num_critics=config.num_critics).to(config.device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.lr_critic)
    with torch.no_grad(): target_critic = deepcopy(critic)  # make training more stable by using a soft updated target critic
    # TODO: maybe also train config.beta, as in the CORL implementation (there its called alpha),
    #       but the paper pseudocode doesn't do that.
    #       the official implementaiton does that only if `use_automatic_entropy_tuning` is enabled

    # continue from checkpoint
    start_epoch = 0
    if config.continue_from:
        continue_file = Path(config.continue_from)
        if not continue_file.exists():
            raise FileNotFoundError(f'File `{continue_file}` does not exist.')
        checkpoint = torch.load(continue_file)
        actor.load_state_dict(checkpoint['actor'])
        critic.load_state_dict(checkpoint['critic'])
        target_critic.load_state_dict(checkpoint['target_critic'])
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        if start_epoch >= config.epochs:
            print(f'Already finished training: {start_epoch}/{config.epochs} epochs')
            return
        print(f'Continuing from epoch {start_epoch}, {config.epochs - start_epoch} epochs left.')

    # save config
    if config.save_path:
        config.save_path_full.mkdir(parents=True, exist_ok=True)
        with (config.save_path_full / 'config.yaml').open("w") as f:
            pyrallis.dump(config, f)

    # init wandb logging
    wandb_run = wandb.init(name=config.name_full, group=config.group, project=config.project, config=config)

    # main training loop
    for epoch in trange(start_epoch, config.epochs, desc='Epoch'):
        for _ in trange(config.updates_per_epoch, desc='Training Update', leave=False):
            # [batch_size, ...]
            state, action, reward, next_state, done = buffer.sample()
            with torch.no_grad():
                # [batch_size, action_dim], [batch_size]
                next_action, next_action_log_prob = actor(next_state)
                # [batch_size]
                q_next = target_critic(next_state, next_action).min(-1).values - config.beta * next_action_log_prob
                # [batch_size]
                q_target = reward + (1 - done) * config.gamma * q_next

            # update critcs
            # [1] <- ([batch_size, num_critics] - [batch_size, 1])
            critic_losses = (critic(state, action) - q_target.unsqueeze(-1)).pow(2).sum(dim=1).mean(dim=0)
            # [batch_size, num_critics, *_dim]
            action_tmp = action.unsqueeze(1).repeat_interleave(config.num_critics, dim=1).requires_grad_()
            # [1]
            critic_q_values_for_grad = torch.cat([critic.models[i](torch.cat([state, action_tmp[:,i]], dim=-1)) for i in range(config.num_critics)]).sum()
            # [batch_size, num_critics, action_dim]
            q_gradients_tmp = torch.autograd.grad(critic_q_values_for_grad, action_tmp, create_graph=True)[0]
            q_gradients = q_gradients_tmp / (q_gradients_tmp.norm(p=2,dim=-1).unsqueeze(-1) + 1e-10)
            # [batch_size, num_critics, num_critics]
            q_cosine_similarity_matrix = (q_gradients @ q_gradients.transpose(1,2)) * (1 - torch.eye(config.num_critics, device=config.device))
            # [1]
            diversity_loss = (q_cosine_similarity_matrix.sum(dim=(1,2))).mean() / (config.num_critics - 1)
            critic_loss = critic_losses + config.eta * diversity_loss
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # update actor
            actor_action, actor_action_log_prob = actor(state)
            actor_q_values = critic(state, actor_action)
            actor_loss = -(actor_q_values.min(-1).values - config.beta * actor_action_log_prob).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # update target critic
            with torch.no_grad():
                for target_param, source_param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_((1 - config.tau) * target_param.data + config.tau * source_param.data)

        # eval
        actor.eval()
        with torch.no_grad():
            if display_video_callback:
                video = []
                state = eval_env.reset()
                done = False
                while not done:
                    action, _ = actor(torch.tensor(state, dtype=torch.float32, device=config.device))
                    state, reward, done, _ = eval_env.step(action.cpu().numpy())
                    video.append(eval_env.render(mode='rgb_array'))
                display_video_callback(video)
            rewards = np.zeros(config.eval_episodes)
            for i in trange(config.eval_episodes, desc='Eval Episode', leave=False):
                state = eval_env.reset()
                done = False
                while not done:
                    action, _ = actor(torch.tensor(state, dtype=torch.float32, device=config.device))
                    state, reward, done, _ = eval_env.step(action.cpu().numpy())
                    rewards[i] += reward
        if config.save_path and (epoch % config.save_every == 0 or epoch == config.epochs - 1):
            torch.save(dict(
                actor = actor.state_dict(),
                critic = critic.state_dict(),
                actor_optimizer = actor_optimizer.state_dict(),
                critic_optimizer = critic_optimizer.state_dict(),
                target_critic = target_critic.state_dict(),
                epoch = epoch,
            ), config.save_path_full / f'edac-{epoch}.pt')
        actor.train()

        # log
        wandb.log({
            "epoch": epoch,
            "critic/loss": critic_loss.item(),
            "critic/base_loss": critic_losses.item(),
            "critic/diversity_loss": diversity_loss.item(),
            "critic/weight_std": torch.stack([torch.cat([p.flatten() for p in c.parameters()]) for c in critic.models]).std(dim=0).mean().item(),
            "actor/loss": actor_loss.item(),
            "actor/entropy": -actor_action_log_prob.mean().item(),
            "actor/q_value_mean": actor_q_values.mean().item(),
            "actor/q_value_std": actor_q_values.std().item(),
            "eval/reward_mean": np.mean(rewards),
            "eval/reward_std": np.std(rewards),
        })

    torch.save(actor.state_dict(), config.save_path_full / f'actor-final.pt')
    wandb_run.finish()



@pyrallis.wrap()
def main(config: TrainConfig) -> None:
    train(config)



if __name__ == '__main__':
    main()
