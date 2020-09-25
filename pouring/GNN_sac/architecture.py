import torch
from torch.nn import Sequential as Seq, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch_geometric.nn as geometric_nn
import numpy as np

def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi

class DPIGNNLayer(MessagePassing):
    def __init__(
        self,
        node_dim,
        edge_dim,
        effect_processor_hidden_layers,
        update_processor_hidden_layers,
    ):
        super(DPIGNNLayer, self).__init__(aggr='mean') #  "Max" aggregation.
        
        self.effect_linears = nn.ModuleList()
        self.effect_linear_in = nn.Linear(2 * node_dim, effect_processor_hidden_layers[0])
        for i in range(1, len(effect_processor_hidden_layers)):
            self.effect_linears.append(nn.Linear(effect_processor_hidden_layers[i-1], effect_processor_hidden_layers[i]))
        self.effect_linear_out = nn.Linear(effect_processor_hidden_layers[-1], edge_dim)

        self.update_linears = nn.ModuleList()
        self.update_linear_in = nn.Linear(node_dim + edge_dim, update_processor_hidden_layers[0])
        for i in range(1, len(update_processor_hidden_layers)):
            self.update_linears.append(nn.Linear(update_processor_hidden_layers[i-1], update_processor_hidden_layers[i]))
        self.update_linear_out = nn.Linear(update_processor_hidden_layers[-1], node_dim)
        
    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # this does the effect computation
        x = torch.cat([x_i, x_j], dim=-1)
        x = F.relu(self.effect_linear_in(x))
        for layer in self.effect_linears:
            x = F.relu(layer(x))
        x = F.relu(self.effect_linear_out(x) + edge_attr)
        # print("x.shape: ", x.shape)
        return x

    def update(self, aggr_out, x):
        # aggr_out has shape [N, edge_dim]
        
        # print("enter update, aggr_out shape is: ", aggr_out.shape)
        new_embedding = torch.cat([aggr_out, x], dim=1)
        new_embedding = F.relu(self.update_linear_in(new_embedding))
        for layer in self.update_linears:
            new_embedding = F.relu(layer(new_embedding))
        new_embedding = F.relu(self.update_linear_out(new_embedding) + x)
        
        return x

class GraphEncoder(nn.Module):
    def __init__(
        self,
        input_node_dim,
        input_edge_dim,
        feature_dim,
        gnn_layer_class=DPIGNNLayer,
        gnn_layer_kwargs=dict(),
        pooling_layer_class=geometric_nn.TopKPooling,
        pooling_layer_kwargs=dict(),
        global_aggregate_func=geometric_nn.global_mean_pool,
        num_gnn_layers=3,
    ):
        super().__init__()
        self.node_embedding_nn = nn.Sequential(
            nn.Linear(input_node_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        self.edge_embedding_nn = nn.Sequential(
            nn.Linear(input_edge_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        self.gnn_layers = nn.ModuleList()
        if pooling_layer_class is not None:
            self.pooling_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            self.gnn_layers.append(gnn_layer_class(**gnn_layer_kwargs))
            if pooling_layer_class is not None:
                self.pooling_layers.append(pooling_layer_class(**pooling_layer_kwargs))

        self.global_aggregate_func = global_aggregate_func
        self.pooling = pooling_layer_class is not None

    def forward(self, obs):
        x, edge_index, edge_attr, batch = obs.x, obs.edge_index, obs.edge_attr, obs.batch
        x = self.node_embedding_nn(x)
        edge_attr = self.edge_embedding_nn(edge_attr)    

        for gnn_layer, pooling_layer in zip(self.gnn_layers, self.pooling_layers):
            x = gnn_layer(x=x, edge_attr=edge_attr, edge_index=edge_index)
            if self.pooling:
                x, edge_index, edge_attr, batch, _, _ = pooling_layer(x, edge_index, edge_attr, batch)
        
        x = self.global_aggregate_func(x, batch)
        return x

class MLP_Actor(nn.Module):
    """MLP actor network."""

    def __init__(
        self,
        action_dim,
        feature_dim,
        log_std_min,
        log_std_max,
        encoder,
    ):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.encoder = encoder

        self.head_linear = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
        )

        self.head_mu = nn.Linear(feature_dim, action_dim)
        self.head_log_std = nn.Linear(feature_dim, action_dim)

    def forward(
      self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs)

        x = F.relu(self.head_linear(obs))
        mu = self.head_mu(x)
        log_std = self.head_log_std(x)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
          self.log_std_max - self.log_std_min
        ) * (log_std + 1)


        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class MLP_Q(nn.Module):
    """MLP Q network."""

    def __init__(
        self,
        action_dim,
        feature_dim,
    ):
        super().__init__()

        self.head_q = nn.Sequential(
            nn.Linear(feature_dim + action_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1)
        )

    def forward(
      self, obs, action
    ):
        x = torch.cat([obs, action], dim=1)

        q = self.head_q(x)

        return q

class MLP_critic(nn.Module):
    """Critic network, employes two q-functions."""

    def __init__(
        self, 
        encoder,
        q_kwargs=dict(),
    ):
        
        super().__init__()

        self.encoder = encoder

        self.Q1 = MLP_Q(
            **q_kwargs
        )
        self.Q2 = MLP_Q(
            **q_kwargs
        )

    def forward(self, obs, action, detach_encoder=False):
        obs = self.encoder(obs)
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        return q1, q2
