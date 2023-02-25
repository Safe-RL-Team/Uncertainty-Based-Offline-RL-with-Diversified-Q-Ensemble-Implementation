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
python edac.py --project=MyWandbEdacProject --total_updates=69420
```

By default, the results are synced to [Weights & Biases](https://wandb.ai/). This can be disabled with `wandb offline` and enabled again with `wandb online`.

