import numpy as np
from torch import nn, optim

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch

from data import random_object_params, objects_from_params, get_batch
from model import CollisionNet, SDF
from train import training_step, DictEMA, evaluate

obj_params = random_object_params(50)

# 1. Define an objective function.
def objective(config):

    sdf = SDF(x_dim=config["x_dim"],
              apply_transform=config["apply_transform"],
              h_dims=[config["l1"], config["l2"]],
              nonlin=config["nonlin"])
    model = CollisionNet(config["n_objs"],
                         sdf=sdf,
                         mirror=config["mirror"])

    objs = objects_from_params(obj_params)
    batch_fn = lambda: get_batch(objs, config["n_batch"])
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(),
                           lr=config["lr"],
                           weight_decay=config["weight_decay"])

    ema = DictEMA(0.01)

    for i in range(config["n_iters"]):
        loss = training_step(model, batch_fn, loss_fn, optimizer)
        metrics = evaluate(model, get_batch(objs, 100))
        tune.report(iterations=i,
                    **ema.update(metrics))


# 2. Define a search space.
search_space = {
    "lr": tune.loguniform(1e-6, 1e-0),
    "weight_decay": tune.loguniform(1e-8, 1e-2),
    "n_batch": tune.lograndint(1e1, 5e4),
    "mirror": tune.choice([True, False]),
    "apply_transform": tune.choice([True, False]),
    "x_dim": tune.qlograndint(3, 150, 3),
    "l1": tune.lograndint(8, 257),
    "l2": tune.lograndint(8, 257),
    "nonlin": tune.choice([nn.ReLU(), nn.Sigmoid(), nn.SiLU(), nn.Tanh()]),
    # These two parameters are not tuned.
    "n_objs": len(obj_params),
    "n_iters": 1000,
}

# These are settings I've found to work well. Uncomment these lines, and the
# hyperparams will not be tuned.
# search_space["lr"] = 1e-3
# search_space["weight_decay"] = 1e-8
# search_space["n_batch"] = 1000
# search_space["mirror"] = True
# search_space["apply_transform"] = True
# search_space["x_dim"] = 30
# search_space["l1"] = 128
# search_space["l2"] = 128
# search_space["nonlin"] =  nn.ReLU()

scheduler = ASHAScheduler(max_t=search_space["n_iters"],
                          grace_period=10,
                          reduction_factor=2)

algo = ConcurrencyLimiter(HyperOptSearch(), max_concurrent=14)
# 3. Start a Tune run and print the best result.
analysis = tune.run(objective,
                    config=search_space,
                    metric="acc",
                    mode="max",
                    scheduler=scheduler,
                    search_alg=algo,
                    num_samples=100,
                    verbose=1)

print(analysis.get_best_config(metric="acc", mode="max"))
# keys = ["score", "iterations"] + ["config."+k for k in search_space.keys()]
print(analysis.results_df.sort_values(by="acc", ascending=False)[:20])
