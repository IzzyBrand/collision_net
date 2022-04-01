import argparse
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
from torch import nn, optim

from data import random_objects, get_batch
from model import CollisionNet, SDF
from train import train, EMA, DictEMA, evaluate


# def sweep_x_dim(args):
#     objs = random_objects(args.n_objs)
#     batch_fn = lambda: get_batch(objs, args.n_batch)
#     loss_fn = nn.BCEWithLogitsLoss()

#     x_dims = (np.logspace(0,2,20).astype(int) * 3)
#     final_accs = []

#     for x_dim in x_dims:
#         print(f"Training a model with x_dim = {x_dim}")
#         model = CollisionNet(args.n_objs, sdf=SDF(x_dim=x_dim))
#         accs = train(model, args.n_iters, batch_fn, loss_fn,
#             optim.Adam(model.parameters(), lr=1e-5))
#         final_accs.append(accs[-1])

#     plt.plot(x_dims, final_accs)
#     plt.xlabel("x_dim")
#     plt.ylabel("Accuracy")
#     plt.show()


# def compare_apply_transform(args):
#     objs = random_objects(args.n_objs)
#     batch_fn = lambda: get_batch(objs, args.n_batch)
#     loss_fn = nn.BCEWithLogitsLoss()

#     for apply_transform in [True, False]:
#         print(f"Training a model with apply_transform = {apply_transform}")
#         model = CollisionNet(args.n_objs, sdf=SDF(apply_transform=apply_transform))
#         losses = train(model, args.n_iters, batch_fn, loss_fn,
#             optim.Adam(model.parameters(), lr=1e-5))
#         plt.plot(losses, label=f"apply_transform = {apply_transform}", alpha=0.5)

#     plt.xlabel("Batch")
#     plt.ylabel("Accuracy")
#     plt.legend()
#     plt.show()


def plot_training_run(args):
    objs = random_objects(args.n_objs)
    model = CollisionNet(args.n_objs)
    eval_batch_fn = lambda: get_batch(objs, args.n_eval, return_gradient=(not args.no_fcl_gradient))

    losses, metrics = train(model=model,
                            n_iters=args.n_iters,
                            batch_fn=lambda: get_batch(objs, args.n_batch),
                            loss_fn=nn.MSELoss(),
                            optimizer=optim.Adam(model.parameters(), lr=3e-3),
                            eval_fn=lambda model: evaluate(model, eval_batch_fn()))
    # smooth out the metrics
    smoothed = DictEMA(0.1).run(metrics)

    n_plots = len(metrics[0])
    n_rows = 2
    n_cols = int(np.ceil(n_plots / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols)
    ax = axes.flat

    # Plot each of them
    for i, (c, k) in enumerate(zip(colors.TABLEAU_COLORS, metrics[0])):
        ax[i].plot([d[k] for d in metrics], c=c, alpha=0.2)
        ax[i].plot([d[k] for d in smoothed], c=c)
        ax[i].set_xlabel(f"Batch (batch_size = {args.n_batch})")
        ax[i].set_ylabel(k)

    fig.suptitle(f"CollisionNet training on {args.n_objs} objects.")
    plt.show()


def main(args):
    try:
        globals()[args.experiment](args)
    except KeyError:
        print(f"No experiment named {args.experiment}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a CollisionNet')
    parser.add_argument("--n-objs", type=int, default=50)
    parser.add_argument("--n-batch", type=int, default=5000)
    parser.add_argument("--n-iters", type=int, default=1000)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--no-fcl-gradient", action="store_true")
    parser.add_argument("--experiment", type=str, default="plot_training_run")
    args = parser.parse_args()
    main(args)