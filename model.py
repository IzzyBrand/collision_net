import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, layer_dims, nonlinearity=nn.ReLU()):
        """Simple fully-connected feed forward network

        Args:
            layer_dims (list(int)): List of layer dims (including inputt and out)
            nonlinearity (nn.Module, optional): nonlinearity function
        """
        super(MLP, self).__init__()

        _layers = []
        n_layers = len(layer_dims) - 1
        for i in range(n_layers):
            _layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))

            if i < n_layers - 1: _layers.append(nonlinearity)

        self.layer_dims = layer_dims
        self.layers = nn.Sequential(*_layers)

    def forward(self, x):
        return self.layers(x)


class SDF(nn.Module):
    def __init__(self,
                 x_dim=48,
                 apply_transform=True,
                 h_dims=[96, 96],
                 transform_shape=torch.Size([4,4])):
        """Computes the signed distance function between two objects
        represented as embedding vectors.

        Args:
            x_dim (int, optional): Embedding dimension
            apply_transform (bool, optional): Transform embedding before MLP
            h_dims (list, optional): Network hidden layer dimensions
            transform_shape (torch.Size, optional): Transform format
        """
        super(SDF, self).__init__()

        if apply_transform:
            assert x_dim % 3 == 0, "x_dim must be a multiple of three."
            assert transform_shape == torch.Size([4,4]), "transform_shape must be [4 x 4]."

        self.x_dim = x_dim
        self.apply_transform = apply_transform
        self.transform_shape = transform_shape

        # Create a neural network that implements the SDF
        first_layer_dim = 2 * x_dim + (not apply_transform) * transform_shape.numel()
        last_layer_dim = 1
        self.mlp = MLP([first_layer_dim] + h_dims + [last_layer_dim])


    def forward(self, x1, x2, T):
        """Compute the SDF between two objects given their relative transform.

        If self.apply_transform is true, x2 will be transformed into the frame
        x1 before passing both embeddings into the network. Otherwise, the
        embeddings are not rotated, and the transform is passed into the network.

        Args:
            x1 (torch.Tensor): [N x D] batch of vectors
            x2 (torch.Tensor): [N x D] batch of vectors
            T (torch.Tensor): [N x 4 x 4] batch of transforms (from 1 to 2)

        Returns:
            torch.Tensor: Distances
        """
        if self.apply_transform:
            x2 = apply_transforms_to_embeddings(x2, T)
            x = torch.cat([x1, x2], dim=1)
        else:
            x = torch.cat([x1, x2, torch.flatten(T, start_dim=1)], dim=1)

        return self.mlp(x).squeeze()


class CollisionNet(nn.Module):
    def __init__(self,
                 n_objs,
                 sdf=SDF(),
                 mirror=True):
        """Predicts collisions between pairs of objects

        Args:
            n_objs (int): Number of objects
            sdf (nn.Module, optional): Computes SDF between two geoms
            mirror (bool, optional): Return mean SDF(x1, x2) and SDF(x2, x1)
        """
        super(CollisionNet, self).__init__()

        self.n_objs = n_objs
        self.sdf = sdf
        self.mirror = mirror
        self.geoms = nn.Parameter(torch.zeros(self.n_objs, self.sdf.x_dim))


    def forward(self, o1, o2, T):
        """Compute distance between object o1 and o2

        Args:
            o1 (torch.Tensor): Index of first object
            o2 (torch.Tensor): Index of second object
            T (torch.Tensor): Transform from first to second object

        Returns:
            torch.Tensor: Min distance between geoms
        """
        x1 = self.geoms[o1]
        x2 = self.geoms[o2]

        if self.mirror:
            T_inv = torch.inverse(T)
            return 0.5 * (self.sdf(x1, x2, T) + self.sdf(x2, x1, T_inv))
        else:
            return self.sdf(x1, x2, T)


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
