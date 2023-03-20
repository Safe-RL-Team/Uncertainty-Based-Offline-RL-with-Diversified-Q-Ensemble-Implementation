# Ensemble Diversified Actor Critic (EDAC)

This is a reimplementation of the EDAC algorithm in PyTorch. The original paper is [Uncertainty-Based-Offline-RL-with-Diversified-Q-Ensemble](https://arxiv.org/abs/2110.01548), and the official implementation can be found [here](https://github.com/snu-mllab/EDAC).

This implementation is heavily inspired by [CORL](https://github.com/tinkoff-ai/CORL/blob/main/algorithms/edac.py).


## Getting started

This assumes you are running Ubuntu. First install [mujoco-py](https://github.com/openai/mujoco-py#install-mujoco), i.e. download the [Mujoco binaries](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) and extract them to `~/.mujoco/mujoco210`. Then:

```bash
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3  # requirements for mujoco
sudo apt install build-essential  # for gcc for pybullet
# TODO: clone this repo and cd into it
conda create -n edac_r python=3.10  # if you want to use a conda env
pip install -r requirements.txt
```

Then run `edac.py` using python. Add `--help` to show the available options.

```bash
conda activate edac_r
python edac.py --project=MyWandbEdacProject --epochs=42
```

By default, the results are synced to [Weights & Biases](https://wandb.ai/). This can be disabled with `wandb offline` and enabled again with `wandb online`.

## Options

You can see all options by running `python edac.py --help`.

```
--epochs int          number of total updates (default: 200)
--updates_per_epoch int
                      number of updates per epoch (default: 500)
--eval_episodes int   number of episodes for evaluation (default: 5)
--batch_size int      batch size (per update) (default: 2048)
--lr_actor float      learning rate for actor (default: 0.0003)
--lr_critic float     learning rate for critic (default: 0.0003)
--lr_beta float       learning rate for beta (default: 0.0003)
--env str             environment name (default: halfcheetah-medium-v2)
--num_critics int     number of critics (default: 5)
--critic_reduction str
                    reduction method for critics (min, mean, mean-[float]) (default: min)
--beta float          factor for action log probability for the actor loss (default: 0.1)
--eta float           diversity loss factor (default: 1.0)
--gamma float         discount factor (default: 0.99)
--tau float           target network update factor (default: 0.005)
--name str            wandb name of the experiment (default: edac)
--group str           wandb group name (default: edac)
--project str         wandb project name (default: edac_reimplementation)
--seed int            seed (0 for random seed) (default: 0)
--device str          device to use (auto, cuda or cpu) (default: auto)
--save_path str       where to save the model weights and config, None for no saving (default: ckp)
--save_every int      save the model every x epochs (default: 10)
--continue_from str   continue training from a checkpoint file (config has to be loaded separately) (default: )
```

By default, the model is periodically saved to `ckp`. You can load a model by specifying `--load_path=ckp/DIR_NAME/MODEL_NAME.pt`. Note, that the config is saved separetely and can be loaded, if needed, by specifying `--config_path=ckp/DIR_NAME/config.yaml`. Remember to increase `--epochs`, if you want to continue a finished training run.
