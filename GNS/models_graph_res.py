import numpy as np
import torch
import torch_geometric.nn
import torch_scatter

from torch_geometric.nn import MetaLayer

# ================== Material Encoder ================== #

# Currently Unused

class MaterialEncoder(torch.nn.Module):

    def __init__(self, input_size, hidden_size=16, output_size=16):

        super(MaterialEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, node_state):

        out = self.model(node_state)

        return out

# ================== Encoder ================== #

class NodeEncoder(torch.nn.Module):

    def __init__(self, input_size, hidden_size=128, output_size=128):

        super(NodeEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            #torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            #torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, node_state):

        out = self.model(node_state)

        return out

class EdgeEncoder(torch.nn.Module):

    def __init__(self, input_size, hidden_size=128, output_size=128):

        super(EdgeEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            #torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            #torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, edge_properties):

        out = self.model(edge_properties)

        return out

class Encoder(torch.nn.Module):

    def __init__(self, node_input_size, edge_input_size, hidden_size=128, output_size=128):

        super(Encoder, self).__init__()

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.node_encoder = NodeEncoder(self.node_input_size, self.hidden_size, self.output_size)
        self.edge_encoder = EdgeEncoder(self.edge_input_size, self.hidden_size, self.output_size)

    def forward(self, node_states, edge_properties):

        node_embedding = self.node_encoder(node_states)
        edge_embedding = self.edge_encoder(edge_properties)

        return node_embedding, edge_embedding

# ================== Processor ================== #

class EdgeModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size=128, output_size=128):

        super(EdgeModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, src, dest, edge_attr, u, batch):

        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.

        u_expanded = u.repeat([src.size()[0], 1])

        model_input = torch.cat([src, dest, edge_attr, u_expanded], 1)
        out = self.model(model_input)

        return out

class NodeModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size=128, output_size=128):

        super(NodeModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):

        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        _, edge_dst = edge_index
        edge_attr_aggregated = torch_scatter.scatter_add(edge_attr, edge_dst, dim=0, dim_size=x.size(0))
        u_expanded = u.repeat([x.size()[0], 1])

        model_input = torch.cat([x, edge_attr_aggregated, u_expanded], dim=1)
        out = self.model(model_input)

        return out

class GlobalModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size=128, output_size=128):

        super(GlobalModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):

        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        node_attr_sum = torch.sum(x, dim=0, keepdim=True)
        edge_attr_sum = torch.sum(edge_attr, dim=0, keepdim=True)

        model_input = torch.cat([u, node_attr_sum, edge_attr_sum], dim=1)
        out = self.model(model_input)

        return out

class GNBlock(torch.nn.Module):

    def __init__(self, input_size, hidden_size=128, output_size=128, use_global=True):

        super(GNBlock, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if use_global:
            self.model = MetaLayer(EdgeModel(self.input_size[0], self.hidden_size, self.output_size),
                                   NodeModel(self.input_size[1], self.hidden_size, self.output_size),
                                   GlobalModel(self.input_size[2], self.hidden_size, 1))
        else:
            self.model = MetaLayer(EdgeModel(self.input_size[0], self.hidden_size, self.output_size),
                                   NodeModel(self.input_size[1], self.hidden_size, self.output_size),
                                   None)

    def forward(self, x, edge_index, edge_attr, u, batch):

        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        x, edge_attr, u = self.model(x, edge_index, edge_attr, u, batch)

        return x, edge_attr, u

class Processor(torch.nn.Module):

    def __init__(self, input_size, hidden_size=128, output_size=128, use_global=True):

        super(Processor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_global = use_global

        self.gn1 = GNBlock(self.input_size, self.hidden_size, self.output_size, self.use_global)
        self.gn2 = GNBlock(2*self.input_size, self.hidden_size, self.output_size, self.use_global)
        self.gn3 = GNBlock(2*self.input_size, self.hidden_size, self.output_size, self.use_global)
        self.gn4 = GNBlock(2*self.input_size, self.hidden_size, self.output_size, self.use_global)
        self.gn5 = GNBlock(self.input_size, self.hidden_size, self.output_size, self.use_global)

    def forward(self, x, edge_index, edge_attr, u, batch):

        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        x1, edge_attr1, u1 = self.gn1(x, edge_index, edge_attr, u, batch)
        x_res = x1 + x
        edge_attr_res = edge_attr1 + edge_attr
        u_res = u1 + u

        x2, edge_attr2, u2 = self.gn2(x_res, edge_index, edge_attr_res, u_res, batch)
        x_res = x2 + x_res
        edge_attr_res = edge_attr2 + edge_attr_res
        u_res = u2 + u_res

        x3, edge_attr3, u3 = self.gn3(x_res, edge_index, edge_attr_res, u_res, batch)
        x_res = x3 + x_res
        edge_attr_res = edge_attr3 + edge_attr_res
        u_res = u3 + u_res

        x4, edge_attr4, u4 = self.gn4(x_res, edge_index, edge_attr_res, u_res, batch)
        x_res = x4 + x_res
        edge_attr_res = edge_attr4 + edge_attr_res
        u_res = u4 + u_res

        x_out, edge_attr_out, u_out = self.gn5(x_res, edge_index, edge_attr_res, u_res, batch)

        return x_out, edge_attr_out, u_out

# ================== Decoder ================== #

class Decoder(torch.nn.Module):

    def __init__(self, input_size=128, hidden_size=128, output_size=3):

        super(Decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, node_feat):

        out = self.model(node_feat)

        return out