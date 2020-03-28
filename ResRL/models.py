import torch.nn as nn
from envs.env import Env
import torch
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class VisualEncoder(nn.Module):
    __constants__ = ['embedding_size', 'image_dim']

    def __init__(self, embedding_size, image_dim, image_c, activation_function='relu'):
        super().__init__()
        self.image_dim = image_dim
        self.image_c = image_c
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        if image_dim == 128:
            self.conv0 = nn.Conv2d(image_c, 16, 4, stride=2)
            self.conv1 = nn.Conv2d(16, 32, 4, stride=2)
        elif image_dim == 64:
            self.conv1 = nn.Conv2d(image_c, 32, 4, stride=2)
        else:
            raise NotImplementedError
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)

    def forward(self, observation):
        observation = observation.view(-1, self.image_c, self.image_dim, self.image_dim)
        if self.image_dim == 128:
            hidden = self.act_fn(self.conv0(observation))
            hidden = self.act_fn(self.conv1(hidden))
        elif self.image_dim == 64:
            hidden = self.act_fn(self.conv1(observation))
        else:
            raise NotImplementedError
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        return hidden


class VisualEncoderConv1d(nn.Module):
    __constants__ = ['embedding_size', 'image_dim']

    def __init__(self, embedding_size, image_dim, image_c, activation_function='relu'):
        super().__init__()
        self.image_dim = image_dim
        self.image_c = image_c
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        if image_dim == 128:
            self.conv0 = nn.Conv1d(image_c, 16, 4, stride=2)
            self.conv1 = nn.Conv1d(16, 32, 4, stride=2)
        elif image_dim == 64:
            self.conv1 = nn.Conv1d(image_c, 32, 4, stride=2)
        else:
            raise NotImplementedError
        self.conv2 = nn.Conv1d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv1d(128, 256, 4, stride=2)
        self.fc = nn.Identity() if embedding_size == 512 else nn.Linear(512, embedding_size)

    def forward(self, observation):
        observation = observation.view(-1, self.image_c, self.image_dim, self.image_dim)
        observation = observation[:, :, :, self.image_dim // 2]  # Only take the one row in the middle
        if self.image_dim == 128:
            hidden = self.act_fn(self.conv0(observation))
            hidden = self.act_fn(self.conv1(hidden))
        elif self.image_dim == 64:
            hidden = self.act_fn(self.conv1(observation))
        else:
            raise NotImplementedError
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 512)
        hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        return hidden


class VisualEncoderFc1d(nn.Module):
    __constants__ = ['embedding_size', 'image_dim']

    def __init__(self, embedding_size, image_dim, image_c, activation_function='relu'):
        super().__init__()
        self.image_dim = image_dim
        self.image_c = image_c
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc = nn.Linear(image_dim * image_c, embedding_size)

    def forward(self, observation):
        observation = observation.view(-1, self.image_c, self.image_dim, self.image_dim)
        observation = observation[:, :, :, self.image_dim // 2].reshape(-1, self.image_c * self.image_dim)  # Only take the one row in the middle
        hidden = self.act_fn(self.fc(observation))
        return hidden


class ActionEncoder(nn.Module):
    def __init__(self, embedding_size, action_dim, activation_function='relu'):
        super().__init__()
        self.l1 = nn.Linear(action_dim, 64)
        self.l2 = nn.Linear(64, embedding_size)
        self.act_fn = getattr(F, activation_function)

    def forward(self, action):
        fa = self.act_fn(self.l1(action))
        fa = self.act_fn(self.l2(fa))
        return fa


class ConvActor(nn.Module):
    def __init__(self, obs_embed_dim, action_dim, image_dim, image_c, max_action, visual_encoder_name):
        super(ConvActor, self).__init__()
        visual_encoder_class = visual_encoder_dict[visual_encoder_name]
        self.visual_encoder = visual_encoder_class(obs_embed_dim, image_dim, image_c)
        self.l1 = nn.Linear(obs_embed_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, obs):
        state = self.visual_encoder(obs)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class ConvCritic(nn.Module):
    def __init__(self, obs_embed_dim, action_dim, image_dim, image_c, action_embed_dim, visual_encoder_name):
        super(ConvCritic, self).__init__()
        visual_encoder_class = visual_encoder_dict[visual_encoder_name]
        self.visual_encoder1 = visual_encoder_class(obs_embed_dim, image_dim, image_c)
        self.action_encoder1 = ActionEncoder(action_embed_dim, action_dim)

        self.visual_encoder2 = visual_encoder_class(obs_embed_dim, image_dim, image_c)
        self.action_encoder2 = ActionEncoder(action_embed_dim, action_dim)

        # Q1 architecture
        self.l1 = nn.Linear(obs_embed_dim + action_embed_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(obs_embed_dim + action_embed_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, obs, action):
        state = self.visual_encoder1(obs)
        action_embed = self.action_encoder1(action)
        sa = torch.cat([state, action_embed], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        state = self.visual_encoder2(obs)
        action_embed = self.action_encoder2(action)
        sa = torch.cat([state, action_embed], 1)
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, obs, action):
        state = self.visual_encoder1(obs)
        action_embed = self.action_encoder1(action)
        sa = torch.cat([state, action_embed], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


visual_encoder_dict = {
    'VisualEncoder': VisualEncoder,  # Conv2d encoder
    'VisualEncoderConv1d': VisualEncoderConv1d,
    'VisualEncoderFc1d': VisualEncoderFc1d
}
