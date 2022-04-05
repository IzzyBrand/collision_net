import argparse
from jax import random, value_and_grad, jit, grad
import jax.numpy as jnp
import optax
from matplotlib import pyplot as plt

from data import random_objects
from experimental.model_and_train import *

def supervise_distance(args):
    d_embed = 30

    objs = random_objects(args.n_objs)
    batch_fn = lambda: jnp_batch(objs, args.n_batch)
    params = collision_net_params([64, 64], args.n_objs, d_embed) # Init the network

    def loss_fn(params, batch):
        inputs = batch[:3]
        labels = batch[3]
        preds = collision_net(params, *inputs)
        return l2(preds, labels)

    def eval_fn(params):
        batch = jnp_batch(objs, args.n_eval)
        inputs = batch[:3]
        labels = batch[3]
        preds = collision_net(params, *inputs)
        acc = (jnp.sign(labels) == jnp.sign(preds)).mean()
        return acc.item()

    optimizer = optax.adam(1e-3)

    params, losses, metrics = train(params, args.n_iters, batch_fn, loss_fn, optimizer, eval_fn)

    plt.plot(losses)
    plt.plot(metrics)
    plt.show()

def supervise_gradients(args):
    d_embed = 30
    alpha = 0.1 # Weighting between gradient loss and distance loss

    objs = random_objects(args.n_objs)
    batch_fn = lambda: jnp_batch(objs, args.n_batch, grad=True)
    params = collision_net_params([64, 64], args.n_objs, d_embed) # Init the network

    summed_net = lambda *a: collision_net(*a).sum()
    grad_net = jit(grad(summed_net, argnums=3)) # Returns d_pred/d_T

    def loss_fn(params, batch):
        inputs = batch[:3]
        labels, label_grads = batch[3:]
        preds = collision_net(params, *inputs)
        pred_grads = grad_net(params, *inputs)
        return (1. - alpha) * l2(preds, labels) + alpha * l2(pred_grads, label_grads)

    def eval_fn(params):
        batch = jnp_batch(objs, args.n_eval, grad=False)
        inputs = batch[:3]
        labels = batch[3]
        preds = collision_net(params, *inputs)
        acc = (jnp.sign(labels) == jnp.sign(preds)).mean()
        return acc.item()

    optimizer = optax.adam(1e-3)

    params, losses, metrics = train(params, args.n_iters, batch_fn, loss_fn, optimizer, eval_fn)
    plt.plot(losses)
    plt.plot(metrics)
    plt.show()

def main(args):
    exp = args.experiment.replace("-", "_")
    if exp == "supervise_gradients":
        supervise_gradients(args)
    elif exp == "supervise_distance":
        supervise_distance(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a CollisionNet in Jax')
    parser.add_argument("--n-objs", type=int, default=10)
    parser.add_argument("--n-batch", type=int, default=1000)
    parser.add_argument("--n-iters", type=int, default=100)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--experiment", type=str, default="supervise_distance")
    args = parser.parse_args()
    main(args)
