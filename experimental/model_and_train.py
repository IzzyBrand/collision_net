from jax import random, value_and_grad, jit, grad
import jax.numpy as jnp
from jax.nn import sigmoid, relu, silu, initializers
import optax

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

def train(params, n_iters, batch_fn, loss_fn, optimizer, eval_fn=None):
    """Run the training loop"""
    opt_state = optimizer.init(params)
    loss_and_grad_fn = jit(value_and_grad(loss_fn))
    losses, metrics = [], []

    for i in range(n_iters):
        loss, grads = loss_and_grad_fn(params, batch_fn())
        # params = step(params, grads)
        # losses.append(loss)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if eval_fn: metrics.append(eval_fn(params))
        print(f"Iteration {i}. Loss:\t{loss}")

    return (params, losses, metrics) if eval_fn else (params, losses)

def jnp_batch(*args, **kwargs):
    """Convert a batch of tensors to a batch of jnp arrays"""
    return [jnp.array(d.numpy()) for d in get_batch(*args, **kwargs)]
