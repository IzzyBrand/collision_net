import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, layer_dims, nonlin=nn.ReLU()):
        """Simple fully-connected feed forward network

        Args:
            layer_dims (list(int)): List of layer dims (including inputt and out)
            nonlin (nn.Module, optional): nonlin function
        """
        super(MLP, self).__init__()

        _layers = []
        n_layers = len(layer_dims) - 1
        for i in range(n_layers):
            _layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))

            if i < n_layers - 1: _layers.append(nonlin)

        self.layer_dims = layer_dims
        self.layers = nn.Sequential(*_layers)

    def forward(self, x):
        return self.layers(x)


class CollisionNet(nn.Module):
    def __init__(self,
                 n_objs,
                 mirror=True,
                 apply_transform=True,
                 x_dim=48,
                 h_dims=[96, 96],
                 nonlin=nn.ReLU()):
        """Predicts collisions between pairs of objects

        Args:
            n_objs (int): Number of objects
            mirror (bool, optional): Return mean dist(x1, x2), dist(x2, x1)
            apply_transform (bool, optional): Transform embedding before MLP
            x_dim (int, optional): Dimension of geometry embedding
            h_dims (list(int), optional): Witdths of hidden layers
            nonlin (nn.Module, optional): nonlinearity to use in mlp
        """
        super(CollisionNet, self).__init__()

        if apply_transform:
            assert x_dim % 3 == 0, "x_dim must be a multiple of three."

        self.x_dim = x_dim
        self.n_objs = n_objs
        self.mirror = mirror
        self.apply_transform = apply_transform
        # A tensor of geometry embeddings
        self.geoms = nn.Parameter(torch.zeros(self.n_objs, self.x_dim))
        # Create a neural network that implements the SDF
        first_layer_dim = 2 * x_dim + (not apply_transform) * 12
        last_layer_dim = 1
        self.mlp = MLP([first_layer_dim] + h_dims + [last_layer_dim], nonlin)

    def forward(self, o1, o2, T, mirrored=False):
        """Compute distance between object o1 and o2

        If self.apply_transform is true, x2 will be transformed into the frame
        x1 before passing both embeddings into the network. Otherwise, the
        embeddings are not rotated, and the transform is passed into the network.

        Args:
            o1 (torch.Tensor): [N] Index of first object
            o2 (torch.Tensor): [N] Index of second object
            T (torch.Tensor): [N x 4 x 4] Transform from first to second object
            mirrored (bool, optional): Indicates whether inputs were swapped

        Returns:
            torch.Tensor: Min distance between geoms
        """
        x1 = self.geoms[o1]
        x2 = self.geoms[o2]

        if self.apply_transform:
            x2 = apply_transforms_to_embeddings(x2, T)
            x = torch.cat([x1, x2], dim=1)
        else:
            # Convert [N x 4 x 4] to [N x 12] by dropping bottom row
            flat_T = torch.flatten(T[:, :3, :], start_dim=1)
            x = torch.cat([x1, x2, flat_T], dim=1)

        d_1_2 = self.mlp(x).squeeze()

        if self.mirror and not mirrored:
            T_inv = torch.inverse(T)
            # Set mirrored=True to avoid infinite recursion
            d_2_1 = self.forward(o2, o1, T_inv, mirrored=True)
            return 0.5 * (d_1_2 + d_2_1)
        else:
            return d_1_2


def apply_transforms_to_embeddings(x, T):
    """Applies 4x4 transform matrices to embedding vectors (Batched)

    The embedding vectors are treated as lists of 3-vectors, and each of those
    3-vectors are individually transformed.

    Args:
        x (torch.Tensor): [N x D] batch of vectors
        T (torch.Tensor): [N x 4 x 4] batch of transforms

    Returns:
        torch.Tensor: [N x D] batch of transformed vectors
    """
    N, D = x.shape
    # Convert x to a set of 3-vectors
    x = x.view(N, D // 3, 3)
    # Convert x to homogeneous coordinates
    ones = torch.ones(N, D // 3, 1)
    x = torch.cat([x, ones], dim=2)
    # Apply the tranform
    x = torch.einsum('Nij, Nxj -> Nxi', T, x)
    # Drop the homogenous coordinates
    x = x[..., :3]
    # And convert back to flat vectors
    return x.reshape(N, D)
