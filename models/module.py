import torch
from torch import nn
from torch_sparse import SparseTensor
from .gnn import SGC, GNN
from .fp import APA


class Features(nn.Module):
    """
    Class that supports various types of node features.
    """

    def __init__(self, edge_index, num_nodes, version, num_iter, obs_nodes=None, missing_nodes=None, obs_features=None,fp_features=None,
                 dropout=0):
        """
        Class initializer.
        """
        super().__init__()
        self.version = version
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.propagation_model = APA(edge_index, fp_features, obs_nodes)
        self.indices = None
        self.values = None
        self.shape = None

        if version == 'diag':
            indices, values, shape = self.to_diag_features()
        elif version == 'degree':
            indices, values, shape = self.to_degree_features(edge_index)
        elif version == 'diag-degree':
            indices, values, shape = self.to_diag_degree_features(edge_index)
        elif version == 'obs-diag':
            indices, values, shape = self.to_obs_diag_features(obs_nodes, obs_features)
        elif version == 'fp':
            self.features = self.propagation_model.fp(fp_features, num_iter = num_iter).cuda()
        else:
            raise ValueError(version)
        if version=='fp':
            nonzero_indices = torch.nonzero(self.features)
            self.indices = nonzero_indices.t()
            self.values = self.features[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            self.shape = self.features.size()
            # sparse_coo = torch.sparse_coo_tensor(nonzero_indices.t(), values, dense_tensor.size())    
        else:
            self.indices = nn.Parameter(indices, requires_grad=False)
            self.values = nn.Parameter(values, requires_grad=False)
            self.shape = shape

    def forward(self):
        """
        Make a feature Tensor from the current information.
        """
        if self.version =='fp':
            return self.features
        else:
            return torch.sparse_coo_tensor(self.indices, self.values, size=self.shape,
                                device=self.indices.device).to_dense()
    def to_diag_features(self):
        """
        Make a diagonal feature matrix.
        """
        nodes = torch.arange(self.num_nodes)
        if self.training and self.dropout > 0:
            nodes = nodes[torch.rand(self.num_nodes) > self.dropout]
        indices = nodes.view(1, -1).expand(2, -1).contiguous()
        values = torch.ones(self.num_nodes)
        shape = self.num_nodes, self.num_nodes
        return indices, values, shape

    def to_degree_features(self, edge_index):
        """
        Make a degree-based feature matrix.
        """
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                             sparse_sizes=(self.num_nodes, self.num_nodes))
        degree = adj_t.sum(dim=0).long()
        degree_list = torch.unique(degree)
        degree_map = torch.zeros_like(degree)
        degree_map[degree_list] = torch.arange(len(degree_list))
        indices = torch.stack([torch.arange(self.num_nodes), degree_map[degree]], dim=0)
        values = torch.ones(indices.size(1))
        shape = self.num_nodes, indices[1, :].max() + 1
        return indices, values, shape

    def to_diag_degree_features(self, edge_index):
        """
        Combine the diagonal and degree-based feature matrices.
        """
        indices1, values1, shape1 = self.to_diag_features()
        indices2, values2, shape2 = self.to_degree_features(edge_index)
        indices = torch.cat([indices1, indices2], dim=1)
        values = torch.cat([values1, values2])
        shape = shape1[0], shape1[1] + shape2[1]
        return indices, values, shape

    def to_obs_diag_features(self, obs_nodes, obs_features):
        """
        Combine the observed features and diagonal ones.
        """
        num_features = obs_features.size(1) + self.num_nodes - len(obs_nodes)
        row, col = torch.nonzero(obs_features, as_tuple=True)
        indices1 = torch.stack([obs_nodes[row], col])
        values1 = obs_features[row, col]

        nodes2 = torch.arange(self.num_nodes)
        nodes2[obs_nodes] = False
        nodes2 = torch.nonzero(nodes2).view(-1)
        indices2 = torch.stack([nodes2, torch.arange(len(nodes2))])
        indices2[1, :] += obs_features.size(1)
        values2 = torch.ones(indices2.size(1))

        indices = torch.cat([indices1, indices2], dim=1)
        values = torch.cat([values1, values2], dim=0)
        shape = self.num_nodes, num_features
        return indices, values, shape


class Encoder(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, dropout, conv):
        super().__init__()
        if conv == 'sgc':
            self.model = SGC(num_features, hidden_size, num_layers)
        elif conv == 'lin':
            self.model = nn.Linear(num_features, hidden_size)
        elif conv in ['gcn', 'sage', 'gat']:
            self.model = GNN(num_features, hidden_size, num_layers, hidden_size, dropout, conv=conv)
        else:
            raise ValueError()

    def forward(self, features, edge_index):
        if isinstance(self.model, nn.Linear):
            return self.model(features)
        else:
            return self.model(features, edge_index)


class Decoder(nn.Module):
    """
    Encoder network in the proposed framework.
    """

    def __init__(self, input_size, output_size, hidden_size=16, num_layers=2, dropout=0.5):
        """
        Class initializer.
        """
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            out_size = output_size if i == num_layers - 1 else hidden_size
            if i > 0:
                layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            layers.append(nn.Linear(in_size, out_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Run forward propagation.
        """
        return self.layers(x)


class Identity(nn.Module):
    """
    PyTorch module that implements the identity function.
    """

    def __init__(self):
        """
        Class initializer.
        """
        super().__init__()

    def forward(self, x):
        """
        Run forward propagation.
        """
        return x


class UnitNorm(nn.Module):
    """
    Unit normalization of latent variables.
    """

    def __init__(self):
        """
        Class initializer.
        """
        super().__init__()

    def forward(self, vectors):
        """
        Run forward propagation.
        """
        valid_index = (vectors != 0).sum(1, keepdims=True) > 0
        vectors = torch.where(valid_index, vectors, torch.randn_like(vectors))
        return vectors / (vectors ** 2).sum(1, keepdims=True).sqrt()


class EmbNorm(nn.Module):
    """
    The normalization of node representations.
    """

    def __init__(self, hidden_size, function='unit', affine=True):
        """
        Class initializer.
        """
        super().__init__()
        if function == 'none':
            self.norm = Identity()
        elif function == 'unit':
            self.norm = UnitNorm()
        elif function == 'batchnorm':
            self.norm = nn.BatchNorm1d(hidden_size, affine=affine)
        elif function == 'layernorm':
            self.norm = nn.LayerNorm(hidden_size, elementwise_affine=affine)
        else:
            raise ValueError(function)

    def forward(self, vectors):
        """
        Run forward propagation.
        """
        return self.norm(vectors)