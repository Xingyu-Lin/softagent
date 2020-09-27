import numpy as np
import torch
import torch.sparse


class Edge(object):
    def __init__(self, edge_dim):
        self.edge_dim = edge_dim
        self.edge_attrs = np.empty((0, edge_dim), dtype=np.float32)
        self.edge_r = []
        self.edge_s = []
        self.global_r_idx = None
        self.global_s_idx = None

    def add_edge(self, edge_r, edge_s, edge_attrs, global_r_idx=None, global_s_idx=None):
        # edge_r and edge_s are in local index.
        self.edge_r = edge_r
        self.edge_s = edge_s
        self.edge_attrs = edge_attrs
        if global_r_idx is not None:
            self.global_r_idx = global_r_idx
            self.global_s_idx = global_s_idx

    def get_tensor(self):
        # Return sparse matrices of shape [num_node, num_edge]
        iR = torch.cat([self.edge_r[None], torch.arange(len(self.edge_r))[None]], dim=0)
        iS = torch.cat([self.edge_s[None], torch.arange(len(self.edge_s))[None]], dim=0)
        values = torch.ones(len(self.edge_r))
        return torch.sparse.FloatTensor(iR, values, torch.Size([len(self.global_r_idx), len(self.edge_r)])), \
               torch.sparse.FloatTensor(iS, values, torch.Size([len(self.global_s_idx), len(self.edge_s)])), \
               torch.FloatTensor(self.edge_attrs), \
               torch.LongTensor(self.global_r_idx), torch.LongTensor(self.global_s_idx)


class GraphExt(object):
    def __init__(self,
                 state_dim,
                 attr_dim,
                 edge_dim):
        self.state_dim = state_dim
        self.attr_dim = attr_dim
        self.edge_dim = edge_dim

        self.states = np.empty((0, state_dim), dtype=np.float32)
        self.attrs = np.empty((0, attr_dim), dtype=np.float32)
        self.p_edges = []  # Edges have multiple stages, i.e. 4 stages for hierarchical modeling in DPI-Net
        self.info = {}

    @property
    def num_node(self):
        return len(self.states)

    @property
    def num_stage(self):
        return len(self.p_edges)

    def add_node(self, states, attrs):
        self.states = np.vstack([self.states, states])
        self.attrs = np.vstack([self.attrs, attrs])

    def add_edge(self, stage, *args, **kwargs):
        while stage >= len(self.p_edges):
            edges = Edge(self.edge_dim)
            self.p_edges.append(edges)
        self.p_edges[stage].add_edge(*args, **kwargs)

    def get_edge_tensor(self, stage, cuda=True):
        edge_tensors = self.p_edges[stage].get_tensor()
        if cuda:
            edge_tensors = [tensor.cuda() for tensor in edge_tensors]
        return edge_tensors

    def get_node_tensor(self, cuda=True):
        node_tensors = [torch.FloatTensor(self.states), torch.FloatTensor(self.attrs)]
        if cuda:
            node_tensors = [tensor.cuda() for tensor in node_tensors]
        return node_tensors


def convert_dpi_to_graph(attr, state, relations, n_particles, n_shapes, instance_idx):
    Rr_idxs, Rs_idxs, values, Ras, node_r_idxs, node_s_idxs, psteps = relations
    graph = GraphExt(state.shape[1], attr.shape[1], Ras[0].shape[1])
    graph.add_node(state, attr)
    for stage, (Rr_idx, Rs_idx, Ra, node_r_idx, node_s_idx) in enumerate(zip(Rr_idxs, Rs_idxs, Ras, node_r_idxs, node_s_idxs)):
        iR = Rr_idx[0, :]
        iS = Rs_idx[0, :]
        # Here iR, iS, Ra are already tensor
        graph.add_edge(stage, iR, iS, Ra, node_r_idx, node_s_idx)

    graph.info['psteps'] = psteps
    graph.info['n_particles'] = n_particles
    graph.info['n_shapes'] = n_shapes
    graph.info['instance_idx'] = instance_idx
    return graph
