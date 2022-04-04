import fcl
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import approx_fprime
import torch


# How many arguments does each fcl object take
fcl_objects = [
    (fcl.Box, 3),
    (fcl.Sphere, 1),
    (fcl.Ellipsoid, 3),
    (fcl.Capsule, 2),
    (fcl.Cone, 2),
    (fcl.Cylinder, 2)
]

def random_object_params(n, bounds=[0.1, 0.2]):
    n_fcl_objs = len(fcl_objects)
    params = []
    for _ in range(n):
        i = np.random.randint(n_fcl_objs)
        _, n_args = fcl_objects[i]
        args = np.random.uniform(*bounds, size=n_args)
        params.append((i, args))
    return params

def objects_from_params(params):
    return [fcl_objects[i][0](*args) for i, args in params]

def random_objects(n, bounds=[0.1, 0.2]):
    params = random_object_params(n, bounds)
    return objects_from_params(params)

def sphere_uniform(n, d):
    # Sample uniform points on the surface of the sphere
    v = np.random.rand(n, d) * 2.0 - 1.0
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    # Sample radii inversely proportional to the surface area
    r = np.random.rand(n) ** (1.0 / d)
    # Scale surface points by radii
    return v * r[:, None]

def random_transforms(n):
    T = np.zeros([n, 4, 4])
    T[:, :3, :3] = Rotation.random(n).as_matrix()
    T[:, :3, 3] = sphere_uniform(n, 3) * 0.33
    T[:, 3, 3] = 1.0
    return T

def random_pairs(n_objs, n):
    obj_1 = np.random.randint(n_objs, size=n)
    o2 = np.random.randint(n_objs-1, size=n)
    o2 += (o2 >= obj_1)
    return obj_1, o2


_request = fcl.DistanceRequest()
_request.enable_signed_distance = True
_request.enable_nearest_points = True
_request.gjk_solver_type = 1 # NOTE(izzy): this fixed distance bug

def distance(o1, o2, T):
    # Set the tranform
    c1 = fcl.CollisionObject(o1, fcl.Transform(np.eye(3), np.zeros(3)))
    c2 = fcl.CollisionObject(o2, fcl.Transform(T[:3,:3], T[:3,3]))
    return  fcl.distance(c1, c2, _request)

def distance_grad(o1, o2, T):
    return approx_fprime(T.ravel(),
        lambda x: distance(o1, o2, x.reshape(4,4))).reshape(4,4)

def get_batch(objs, n, return_gradient=False):
    # Random transforms
    Ts = random_transforms(n)
    # Random pairs of objects
    pairs = random_pairs(len(objs), n)
    # Distance results
    ds = [distance(objs[i], objs[j], T) for i, j, T in zip(*pairs, Ts)]
    # Make it torch
    batch = [torch.LongTensor(pairs[0]),
             torch.LongTensor(pairs[1]),
             torch.Tensor(Ts),
             torch.Tensor(ds)]

    if return_gradient:
        dds = np.array([distance_grad(objs[i], objs[j], T)\
            for i, j, T in zip(*pairs, Ts)])
        batch.append(torch.Tensor(dds))

    return batch


if __name__ == "__main__":
    objs = random_objects(50)
    batch = get_batch(objs, 1000)

    import matplotlib.pyplot as plt
    plt.hist(batch[3].numpy(), bins=50)
    plt.show()
