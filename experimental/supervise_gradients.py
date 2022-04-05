from jax import random, value_and_grad, jit, grad
import jax.numpy as jnp
from jax.nn import sigmoid, relu, silu, initializers
from matplotlib import pyplot as plt

from data import get_batch, random_objects


def l2(x, y):
    if x.ndim == 2:
        return jnp.linalg.norm(x - y, axis=1).mean()
    elif x.ndim == 3:
        return jnp.linalg.norm(x - y, axis=(1, 2)).mean()

def mlp_params(layer_dims, seed=0):
    """Get initial parameters for a MLP"""
    params = []
    xavier = initializers.glorot_normal()
    for i, (l, ll) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        # Get a key for the random number generator
        key = random.PRNGKey(i+seed)
        # Initialize weights with Glorot (Xavier)
        w = xavier(key, (ll, l), jnp.float32)
        # Intialize bias to zero
        b = jnp.zeros(ll)
        # Save the parameter vectors
        params.append((w, b))

    return params

def mlp_forward(params, x, nonlin=silu):
    """Implements a forward pass of a MLP"""
    n = len(params)
    for i, (w, b) in enumerate(params):
        # Linear
        x = (w @ x.T).T + b
        # Don't nonlin the last layer
        if i == n - 1: return x
        else: x = nonlin(x)

def collision_net_params(h_dims, n_objs, d_embed, seed=0):
    """Get initial parameters for the collision network"""
    mlp = mlp_params([d_embed * 2 + 16] + h_dims + [1], seed=seed)
    embeddings = random.normal(random.PRNGKey(seed), shape=[n_objs, d_embed])
    return mlp, embeddings

def collision_net(params, i, j, T):
    """Implements a forward pass of the collision network"""
    mlp_params, embeddings = params
    x = jnp.concatenate([embeddings[i],
                         embeddings[j],
                         T.reshape(-1, 16)], axis=1)
    return mlp_forward(mlp_params, x)

def step(params, grad, lr=-1e-3):
    """Update a nested list or tuple of jnp arrays"""
    if isinstance(params, jnp.ndarray):
        return params + lr * grad
    elif isinstance(params, list):
        return [step(p, g, lr) for p, g in zip(params, grad)]
    elif isinstance(params, tuple):
        return tuple(step(p, g, lr) for p, g in zip(params, grad))

def train(params, n_iters, batch_fn, loss_fn, eval_fn=None):
    """Run the training loop"""
    loss_and_grad = jit(value_and_grad(loss_fn, argnums=0))
    losses, metrics = [], []
    for i in range(n_iters):
        loss, grad = loss_and_grad(params, batch_fn())
        params = step(params, grad)
        losses.append(loss)

        if eval_fn: metrics.append(eval_fn(params))
        print(f"Iteration {i}. Loss:\t{loss}")

    return (params, losses, metrics) if eval_fn else (params, losses)

def jnp_batch(*args, **kwargs):
    """Convert a batch of tensors to a batch of jnp arrays"""
    return [jnp.array(d.numpy()) for d in get_batch(*args, **kwargs)]

def supervise_gradients():
    n_objs = 10
    n_batch = 500
    n_eval = 100
    n_iters = 50
    alpha = 0.1 # Weighting between gradient loss and distance loss

    objs = random_objects(n_objs)
    batch_fn = lambda: jnp_batch(objs, n_batch, grad=True)
    params = collision_net_params([64, 64], n_objs, 30) # Init the network

    summed_net = lambda *args: collision_net(*args).sum()
    grad_net = jit(grad(summed_net, argnums=3)) # Returns d_pred/d_T

    def loss_fn(params, batch):
        inputs = batch[:3]
        labels, label_grads = batch[3:]
        preds = collision_net(params, *inputs)
        pred_grads = grad_net(params, *inputs)
        return (1. - alpha) * l2(preds, labels) + alpha * l2(pred_grads, label_grads)

    def eval_fn(params):
        batch = jnp_batch(objs, n_eval, grad=False)
        inputs = batch[:3]
        labels = batch[3]
        preds = collision_net(params, *inputs)
        acc = (jnp.sign(labels) == jnp.sign(preds)).mean()
        # label_grads = batch[4]
        # pred_grads = grad_net(params, *inputs)
        return acc.item()

    params, losses, metrics = train(params, n_iters, batch_fn, loss_fn, eval_fn)
    plt.plot(losses)
    plt.plot(metrics)
    plt.show()

if __name__ == "__main__":
    supervise_gradients()
