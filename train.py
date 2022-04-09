import torch
from torch import nn
from torch.nn import functional as f
from tqdm import tqdm

from data import get_batch

class EMA:
    """Exponential moving average smoother"""
    def __init__(self, a, x0=None):
        self.a = a
        self.x = x0

    def update(self, x):
        self.x = (1 - self.a) * self.x + self.a * x if self.x else x
        return self.x

    def run(self, X):
        return [self.update(x) for x in X]

class DictEMA(EMA):
    """Exponential moving average smoother for dicts"""
    def update(self, x):
        if self.x:
            self.x = {k: self.e[k].update(v) for k, v in x.items()}
        else:
            self.x = x
            self.e = {k: EMA(self.a, v) for k, v in x.items()}

        return self.x


def gradient_metrics(model, batch):
    o1, o2, T, label, dlabel_dT = batch
    T = nn.Parameter(T)  # Make T a parameter so we can get the gradient
    model(o1, o2, T).sum().backward()
    dpred_dT = T.grad.detach()

    # Change in distance wrt relative position
    dpred_dpos = dpred_dT[:, :3, 3]
    dlabel_dpos = dlabel_dT[:, :3, 3]
    # Change in distance wrt relative orientation
    dpred_drot = dpred_dT[:, :3, :3].flatten(1)
    dlabel_drot = dlabel_dT[:, :3, :3].flatten(1)

    # Computes the angle between two vectors in radians
    def angle(v1, v2):
        return torch.rad2deg(torch.arccos(nn.CosineSimilarity()(v1, v2))).mean()

    return {
        "grad_pos_l1": f.l1_loss(dpred_dpos, dlabel_dpos).item(),
        "grad_pos_angle": angle(dpred_dpos, dlabel_dpos).item(),
        "grad_rot_l1": f.l1_loss(dpred_drot, dlabel_drot).item(),
        "grad_rot_angle": angle(dpred_drot, dlabel_drot).item(),
    }


def make_loss_fn(L):
    """Convert a typical L(pred, target) loss function into a more generic
    loss_fn(model, batch) format"""
    def loss_fn(model, batch):
        o1, o2, T, label = batch[:4]
        pred = model(o1, o2, T)
        return L(pred, label)

    return loss_fn


def evaluate(model, batch):
    o1, o2, T, label = batch[:4]

    with torch.no_grad():
        pred = model(o1, o2, T)
        metrics = {
            "acc": (pred.sign() == label.sign()).float().mean().item(),
            "rmse": torch.sqrt(torch.mean(torch.square(pred - label))).item(),
            "mae": f.l1_loss(pred, label).item(),
        }

    if len(batch) > 4:
        metrics.update(gradient_metrics(model, batch))

    return metrics


def training_step(model, batch_fn, loss_fn, optimizer):
    optimizer.zero_grad()
    batch = batch_fn()
    loss = loss_fn(model, batch)
    loss.backward()
    optimizer.step()

    return loss.item()


def train(model, n_iters, batch_fn, loss_fn, optimizer, eval_fn=None, verbose=True):
    losses, metrics = [], []

    pbar = tqdm(range(n_iters), disable=(not verbose))
    try:
        for n in pbar:
            loss = training_step(model, batch_fn, loss_fn, optimizer)
            losses.append(loss)

            if eval_fn:
                metric = eval_fn(model)
                metrics.append(metric)
                pbar.set_postfix({'Acc': metric['acc']})
            else:
                pbar.set_postfix({'Loss': loss})
    except KeyboardInterrupt:
        pass

    return (losses, metrics) if eval_fn else losses
