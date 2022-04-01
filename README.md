# collision_net

Checking collision with a neural network.

![The results from running `python run.py --n-objects=10 --n-iters=1000 --n-batch=1000`](figures/collision_net_metrics.png)

We want to use neural networks to check collision for a few reasons

 * They might be faster than mesh-based collision checkers
 * They can produce collision gradients

## Running stuff

 * `python run.py` will train a collision net and plot some performance metrics
 * `python tune.py` will perform a hyperparameter search 
 * `python data.py` will generate some training data and histgram the labels

The neural network is defined in `model.py` and training utils are defined in `train.py`.

## Setup

I always recommend a virtaul environment

```
virtualenv -p python3 venv
./venv/bin/activate
```

Then you can install the requirements.

```
pip install torch ray tqdm matplotlib numpy scipy python-fcl
```

 * `scipy` is used for generating random rotations and finite difference gradients
 * `ray` is used for hyperparameter tuning
 * `tqdm` makes pretty progress bars during training
 * `python-fcl` is a collision checking library used for creating label data
