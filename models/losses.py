from torch import nn
import torch
from torch_geometric.utils import get_laplacian, subgraph, k_hop_subgraph, to_undirected
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops
import torch.nn.functional as F
import numpy as np


def to_x_loss(x_loss):
    """
    Make a loss term for estimating node features.
    """
    if x_loss in ['base', 'balanced']:
        return BernoulliLoss(x_loss)
    elif x_loss == 'gaussian':
        return MSELoss()
    else:
        raise ValueError(x_loss)


class BernoulliLoss(nn.Module):
    """
    Loss term for the binary features.
    """

    def __init__(self, version='base'):
        """
        Class initializer.
        """
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.version = version

    def forward(self, input, target):
        """
        Run forward propagation.
        """
        #assert (target < 0).sum() == 0
        if self.version == 'base':
            loss = self.loss(input, target)
        elif self.version == 'balanced':
            pos_ratio = (target > 0).float().mean()
            weight = torch.ones_like(target)
            weight[target > 0] = 1 / (2 * pos_ratio)
            weight[target == 0] = 1 / (2 * (1 - pos_ratio))
            loss = self.loss(input, target) * weight
        else:
            raise ValueError(self.version)
        return loss.mean()



class GraphLoss(nn.Module):
    def __init__(self, obs_nodes,  missing_nodes, edge_index, num_nodes, num_features, lambda_penalty=1.0, threshold=0.8):
        super().__init__()
        self.cached_adj = None
        self.lambda_penalty = lambda_penalty
        self.threshold = threshold
        self.loss = nn.MSELoss()
        #self.num_nodes = num_nodes
        #self.edge_index = self.get_k_hop_neighbors(edge_index, 2, self.num_nodes).cuda()
        #self.edge_index = self.sample_edges(edge_index, 0.2).cuda()
        self.f_k2u = self.compute_f_k2u(edge_index, obs_nodes,  num_nodes, num_features)
        self.f_u2k = self.count_unknown_neighbors(edge_index, missing_nodes, num_nodes)

    def compute_f_k2u(self, edge_index, know_mask, num_nodes, feat_dim):

        
        len_v_0tod_list = []
        f_k2u = torch.zeros(num_nodes, dtype = torch.int)
        v_0 = know_mask
        len_v_0tod_list.append(len(v_0))
        v_0_to_now = v_0
        f_k2u[v_0] = 0
        d = 1
        while True:
            v_d_hop_sub = k_hop_subgraph(v_0, d, edge_index, num_nodes=num_nodes)[0]
            v_d = torch.from_numpy(np.setdiff1d(v_d_hop_sub.cpu(), v_0_to_now.cpu())).to(v_0.device)
            if len(v_d) == 0:
                break
            f_k2u[v_d] = d
            v_0_to_now = torch.cat([v_0_to_now, v_d], dim=0)
            len_v_0tod_list.append(len(v_d))
            d += 1
        return f_k2u

    def count_unknown_neighbors(self, edge_index, unknown_mask, num_nodes):
        unknown_set = set(unknown_mask.tolist())
        count = torch.zeros(num_nodes, dtype=torch.long).cuda()
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        for src, tgt in zip(source_nodes, target_nodes):
            if tgt.item() in unknown_set:  
                count[src] += 1
        return count


    def get_k_hop_neighbors(self, edge_index, k, num_nodes):
        #edge_index = to_undirected(edge_index)
        for _ in range(k-1):  
            row, col = edge_index

            adj_matrix = torch.sparse_coo_tensor(edge_index, torch.ones_like(row), (num_nodes, num_nodes)).to_dense()
            power_adj_matrix = torch.mm(adj_matrix.float(), adj_matrix.float())
            power_adj_matrix[power_adj_matrix > 0] = 1  
            edge_index = power_adj_matrix.nonzero(as_tuple=False).t()
            edge_index, _ = remove_self_loops(edge_index)  
        return self.sample_edges(edge_index, 0.2).cuda()

    def sample_edges(self, edge_index, percentage=0.2):
        num_edges = edge_index.size(1)
        num_sampled_edges = int(num_edges * percentage)

        sampled_indices = torch.randperm(num_edges)[:num_sampled_edges]
        sampled_edge_index = edge_index[:, sampled_indices]

        return sampled_edge_index


    def homophily(self, z, edge_index, crash):
        #edge_index = self.get_k_hop_neighbors(edge_index, 2, z.shape[0])
        #edge_index = self.sample_edges(edge_index, 0.2)
        #print(edge_index.shape)
        i, j = edge_index
        similarity = torch.cosine_similarity(z[i], z[j], dim=1)
        homophily_scores = torch.zeros(z.size(0), device=z.device)
        homophily_scores.scatter_add_(0, i, similarity)
        degrees = torch.bincount(i)
        degrees[degrees == 0] = 1  
        if crash:
            homophily_scores /= degrees.float()
        else:
            homophily_scores /= degrees.float()
            homophily_scores = homophily_scores[self.missing_nodes]

        # target_distribution = torch.tensor([1.0], device=z.device)
        # homophily_prob = homophily_scores.mean().unsqueeze(0)
        # loss = torch.nn.functional.binary_cross_entropy(homophily_prob, target_distribution)

        return -homophily_scores.mean()

    def extra_loss(self, z, edge_index, crash):
        num_nodes = z.size(0)
        similarity_matrix = torch.mm(z, z.t())
        adjacency_matrix = torch.zeros((num_nodes, num_nodes), device=z.device)
        i, j = edge_index
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1  

        non_connected_similarity = similarity_matrix * (1 - adjacency_matrix)
        if crash:
            penalty = torch.where(non_connected_similarity > self.threshold, non_connected_similarity, torch.zeros_like(non_connected_similarity))
        else:
            penalty = torch.where(non_connected_similarity > self.threshold, non_connected_similarity, torch.zeros_like(non_connected_similarity))
            penalty = penalty[self.missing_nodes]
        return penalty.mean()


    def forward(self, z, edge_index, x_hat, epoch, alpha=0.9, eps=0.01):
        f_k2u = self.f_k2u.repeat(z.shape[1],1) 
        f_u2k = self.f_u2k.repeat(z.shape[1],1) 
        cor = torch.corrcoef(z.T).nan_to_num().fill_diagonal_(0)
        f_k2u = f_k2u.to(z.device)
        W = ( (alpha ** f_k2u.T) + (1-alpha**f_u2k.T) ) * (z - torch.mean(z, dim=0))
        B = torch.matmul(W, cor)
        z =  z + eps*B


        homophily_loss = self.homophily(z, edge_index, crash=True) #+ self.homophily(x_hat, edge_index, crash=False)
        penalty = self.extra_loss(z, edge_index, crash=True) #+ self.extra_loss(x_hat, edge_index, crash=False)
        total_loss =  homophily_loss + self.lambda_penalty *penalty #if epoch < 200 else homophily_loss # homophily_loss + self.lambda_penalty *
        return total_loss