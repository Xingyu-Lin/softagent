import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F


### Dynamic Particle Interaction Networks

class RelationEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        '''
        f_R^enc in the paper
        '''
        super(RelationEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        Args:
            x: [n_relations, input_size]
        Returns:
            [n_relations, output_size]
        '''
        return self.model(x)


class ParticleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        '''
        f_O^enc in the paper
        '''
        super(ParticleEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        '''
        # print(x.size())
        return self.model(x)


class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual=False):
        '''
        relation and object propagator in the paper (f_R nad f_O).
        '''
        super(Propagator, self).__init__()

        self.residual = residual

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, res=None):
        '''
        Args:
            x: [n_relations/n_particles, input_size]
        Returns:
            [n_relations/n_particles, output_size]
        '''
        if self.residual:
            x = self.relu(self.linear(x) + res)
        else:
            x = self.relu(self.linear(x))

        return x


class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        '''
        f_O^ouput in the paper.
        '''
        super(ParticlePredictor, self).__init__()

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        '''
        x = self.relu(self.linear_0(x))
        x = self.relu(self.linear_1(x))

        return self.linear_2(x)


class DPINet(nn.Module):
    def __init__(self, args, stat, phases_dict, residual=False, use_gpu=False):

        super(DPINet, self).__init__()

        self.args = args

        state_dim = args.state_dim
        attr_dim = args.attr_dim  # NOTE: object = (state, obj_attr)
        relation_dim = args.relation_dim  # NOTE: relation = (obj_receive, obj_send, relation_attri)
        nf_particle = args.nf_particle  # NOTE: hidden layers for particle encoder network
        nf_relation = args.nf_relation  # NOTE: hidden layers for relation encoder network
        nf_effect = args.nf_effect  # NOTE: hidden layers for propagation network

        self.nf_effect = args.nf_effect

        self.stat = stat
        self.use_gpu = use_gpu
        self.residual = residual

        if use_gpu:
            self.pi = Variable(torch.FloatTensor([np.pi])).cuda()
            self.dt = Variable(torch.FloatTensor([args.dt])).cuda()
            self.mean_v = Variable(torch.FloatTensor(stat[1][:, 0])).cuda()
            self.std_v = Variable(torch.FloatTensor(stat[1][:, 1])).cuda()
            self.mean_p = Variable(torch.FloatTensor(stat[0][:3, 0])).cuda()
            self.std_p = Variable(torch.FloatTensor(stat[0][:3, 1])).cuda()
        else:
            self.pi = Variable(torch.FloatTensor([np.pi]))
            self.dt = Variable(torch.FloatTensor(args.dt))
            self.mean_v = Variable(torch.FloatTensor(stat[1][:, 0]))
            self.std_v = Variable(torch.FloatTensor(stat[1][:, 1]))
            self.mean_p = Variable(torch.FloatTensor(stat[0][:3, 0]))
            self.std_p = Variable(torch.FloatTensor(stat[0][:3, 1]))

        # (1) particle attr (2) state
        self.particle_encoder_list = nn.ModuleList()
        for i in range(args.n_stages):
            print("particle encoder input size: ", attr_dim + state_dim * 2)
            self.particle_encoder_list.append(
                ParticleEncoder(attr_dim + state_dim * 2, nf_particle, nf_effect))
            # NOTE: reason for state_dim*2: 1 is the original state, another is offset to the center of mass
            # (which is non-zero for rigid and zero for fluid)

        # (1) sender attr (2) receiver attr (3) state receiver (4) state sender (5) relation attr
        self.relation_encoder_list = nn.ModuleList()
        for i in range(args.n_stages):
            self.relation_encoder_list.append(RelationEncoder(
                2 * attr_dim + 4 * state_dim + relation_dim,
                nf_relation, nf_relation))

        # (1) relation encode (2) sender effect (3) receiver effect
        self.relation_propagator_list = nn.ModuleList()
        for i in range(args.n_stages):
            self.relation_propagator_list.append(Propagator(nf_relation + 2 * nf_effect, nf_effect))

        # (1) particle encode (2) particle effect
        # NOTE: unlike in the paper equation (3), the privious h (particle effect) is not used as input for the 
        # MLP, but just used as a residual connection.
        self.particle_propagator_list = nn.ModuleList()
        for i in range(args.n_stages):
            self.particle_propagator_list.append(Propagator(2 * nf_effect, nf_effect, self.residual))

        # (1) set particle effect
        self.rigid_particle_predictor = ParticlePredictor(nf_effect, nf_effect, 7)  # predict rigid motion
        self.fluid_particle_predictor = ParticlePredictor(nf_effect, nf_effect, args.position_dim)  # NOTE: I thought label is velocity??
        # but anyway, in all the cases position_dim = velocity_dim (check for RiceGrip)

    def rotation_matrix_from_quaternion(self, params):
        # params dim - 4: w, x, y, z

        if self.use_gpu:
            one = Variable(torch.ones(1, 1)).cuda()
            zero = Variable(torch.zeros(1, 1)).cuda()
        else:
            one = Variable(torch.ones(1, 1))
            zero = Variable(torch.zeros(1, 1))

        # multiply the rotation matrix from the right-hand side
        # the matrix should be the transpose of the conventional one

        # Reference
        # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm

        params = params / torch.norm(params)
        w, x, y, z = params[0].view(1, 1), params[1].view(1, 1), params[2].view(1, 1), params[3].view(1, 1)

        rot = torch.cat((
            torch.cat((one - y * y * 2 - z * z * 2, x * y * 2 + z * w * 2, x * z * 2 - y * w * 2), 1),
            torch.cat((x * y * 2 - z * w * 2, one - x * x * 2 - z * z * 2, y * z * 2 + x * w * 2), 1),
            torch.cat((x * z * 2 + y * w * 2, y * z * 2 - x * w * 2, one - x * x * 2 - y * y * 2), 1)), 0)

        return rot

    def forward(self, graph, phases_dict, verbose=0):
        state, attr = graph.get_node_tensor()
        instance_idx, psteps = graph.info['instance_idx'], graph.info['psteps']

        # calculate particle encoding
        particle_effect = torch.zeros((attr.size(0), self.nf_effect)).cuda()
        # add offset to center-of-mass for rigids to attr
        offset = torch.zeros((attr.size(0), state.size(1))).cuda()

        for i in range(len(instance_idx) - 1):
            st, ed = instance_idx[i], instance_idx[i + 1]
            if phases_dict['material'][i] == 'rigid':
                c = torch.mean(state[st:ed], dim=0)
                offset[st:ed] = state[st:ed] - c
        attr = torch.cat([attr, offset], 1)

        n_stage = graph.num_stage
        # NOTE: the particle_effect is inherited from the last stage to the next stage. so the hierarchical instance order matters.
        for s in range(n_stage):
            if verbose:
                print("=== Stage", s)
            Rr, Rs, Ra, node_r_idx, node_s_idx = graph.get_edge_tensor(s)
            Rrp = Rr.t()  # transpose, then has shape (n_rel, n_receiver)
            Rsp = Rs.t()

            # receiver_attr, sender_attr
            attr_r = attr[node_r_idx]
            attr_s = attr[node_s_idx]
            attr_r_rel = Rrp.mm(attr_r)  # NOTE: use matrix multiplication to extract the receiver's attribute
            attr_s_rel = Rsp.mm(attr_s)

            # receiver_state, sender_state
            state_r = state[node_r_idx]
            state_s = state[node_s_idx]
            state_r_rel = Rrp.mm(state_r)  # NOTE: again use matrix multiplication to extract the receiver's state
            state_s_rel = Rsp.mm(state_s)
            # state_diff = state_r_rel - state_s_rel  # NOTE: state_diff is actually never used

            # particle encode
            if verbose:
                print('attr_r', attr_r.shape, 'state_r', state_r.shape)  # NOTE: the attr has concated the offset to the mass center
            particle_encode = self.particle_encoder_list[s](torch.cat([attr_r, state_r], 1))

            # calculate relation encoding
            relation_encode = self.relation_encoder_list[s](
                torch.cat([attr_r_rel, attr_s_rel, state_r_rel, state_s_rel, Ra], 1))
            if verbose:
                print("relation encode:", relation_encode.size())

            for i in range(psteps[s]):
                if verbose:
                    print("pstep", i)
                    print("Receiver index range", np.min(node_r_idx), np.max(node_r_idx))
                    print("Sender index range", np.min(node_s_idx), np.max(node_s_idx))

                effect_p_r = particle_effect[node_r_idx]
                effect_p_s = particle_effect[node_s_idx]

                receiver_effect = Rrp.mm(effect_p_r)  # Rrp: (n_rel, n_receiver), where each row is (0, 0, ..., 1, ..., 0), 1 at receiver idx
                sender_effect = Rsp.mm(effect_p_s)

                # calculate relation effect
                effect_rel = self.relation_propagator_list[s](
                    torch.cat([relation_encode, receiver_effect, sender_effect], 1))
                if verbose:
                    print("relation effect:", effect_rel.size())

                # calculate particle effect by aggregating relation effect
                effect_p_r_agg = Rr.mm(effect_rel)  # (n_receiver, n_rel) x (n_rel, nf_effect) -> (n_receiver, nf_effect)
                # NOTE: this is correct. The first row of Rr[s] encodes in which rels the first particle is receiver.

                # calculate particle effect
                effect_p = self.particle_propagator_list[s](
                    torch.cat([particle_encode, effect_p_r_agg], 1),
                    res=effect_p_r)
                if verbose:
                    print("particle effect:", effect_p.size())

                # NOTE: update the residual
                particle_effect[node_r_idx] = effect_p

        pred = []
        for i in range(len(instance_idx) - 1):
            st, ed = instance_idx[i], instance_idx[i + 1]

            if phases_dict['material'][i] == 'rigid':
                t = self.rigid_particle_predictor(torch.mean(particle_effect[st:ed], 0)).view(-1)

                R = self.rotation_matrix_from_quaternion(t[:4])
                b = t[4:] * self.std_p

                p_0 = state[st:ed, :3] * self.std_p + self.mean_p
                c = torch.mean(p_0, dim=0)
                p_1 = torch.mm(p_0 - c, R) + b + c
                v = (p_1 - p_0) / self.dt
                pred.append((v - self.mean_v) / self.std_v)

            elif phases_dict['material'][i] == 'fluid':
                pred.append(self.fluid_particle_predictor(particle_effect[st:ed]))

        pred = torch.cat(pred, 0)

        if verbose:
            print("pred:", pred.size())

        return pred
