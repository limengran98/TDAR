
import torch
from torch import nn
from torch.nn import MSELoss
from .losses import to_x_loss, GraphLoss
from .module import *



class TDAR(nn.Module):
    """
    Class of our proposed method.
    """

    def __init__(self, edge_index, num_nodes, num_features, num_classes, hidden_size=256, eps=0.01, alpha=0.9, num_iter=30, lamda=1,
                lambda_penalty=0.8, threshold=0.1, num_layers=2, conv='gcn', dropout=0.5, x_type='fp', x_loss='balanced',
                 emb_norm='unit', obs_nodes=None, missing_nodes=None, obs_features=None, fp_features=None, dec_bias=False):
        """
        Class initializer.
        """
        super().__init__()
        self.missing_nodes = missing_nodes
        self.lamda = lamda
        self.dropout = nn.Dropout(dropout)

        self.features = Features(edge_index, num_nodes, x_type, num_iter, obs_nodes, missing_nodes, obs_features, fp_features)
        self.encoder = Encoder(self.features.shape[1], hidden_size, num_layers, dropout, conv)
        self.emb_norm = EmbNorm(hidden_size, emb_norm)

        self.x_decoder = nn.Linear(hidden_size, num_features, bias=dec_bias)
        self.y_decoder = nn.Linear(hidden_size, num_classes, bias=dec_bias)

        self.x_loss = to_x_loss(x_loss)
        self.y_loss = nn.CrossEntropyLoss()
        self.a_loss = nn.MSELoss()
        self.graph_loss = GraphLoss(obs_nodes,  missing_nodes, edge_index,  num_nodes, num_features, lambda_penalty, threshold)

        self.alpha = alpha
        self.eps = eps


    def forward(self, edge_index, for_loss=False):
        """
        Run forward propagation.
        """
        z = self.emb_norm(self.encoder(self.features(), edge_index))
        z_dropped = self.dropout(z)
        x_hat = self.x_decoder(z_dropped)
        y_hat = self.y_decoder(z_dropped)
        if for_loss:
            return z, x_hat, y_hat
        return z, x_hat, y_hat

    def to_y_loss(self, y_hat, y_nodes, y_labels):
        """
        Make a loss term for observed labels.
        """
        if y_nodes is not None and y_labels is not None:
            return self.y_loss(y_hat[y_nodes], y_labels)
        else:
            return torch.zeros(1, device=y_hat.device)


    def to_losses(self, edge_index, x_nodes, x_features, y_nodes=None, y_labels=None, model=None, epoch=0):
        """
        Make three loss terms for the training.
        """
        
        z, x_hat, y_hat = self.forward(edge_index, for_loss=True)
        l1 = self.x_loss(x_hat[x_nodes], x_features) 
        l2 = torch.zeros(1, device=y_hat.device)#self.to_y_loss(y_hat, y_nodes, y_labels) #
        l3 = self.lamda * self.graph_loss(z, edge_index, x_hat, epoch, self.alpha, self.eps)
        return l1, l2, l3

