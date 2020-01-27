import torch.nn as nn
import torch
from torch.nn import functional as F
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.torch.core import eval_np


class VisualEncoder(nn.Module):
    def __init__(self, embedding_size, image_dim, activation_function='relu'):
        super().__init__()
        self.image_dim = image_dim
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        if image_dim == 128:
            self.conv0 = nn.Conv2d(3, 16, 4, stride=2)
            self.conv1 = nn.Conv2d(16, 32, 4, stride=2)
        elif image_dim == 64:
            self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        else:
            raise NotImplementedError
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)

    def forward(self, observation):
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


class ConvQ(nn.Module):
    def __init__(self, embedding_dim, image_dim, hidden_sizes, action_dim):
        super().__init__()
        self.image_dim = image_dim
        self.encoder = VisualEncoder(embedding_dim, image_dim)
        self.mlp_Q = FlattenMlp(input_size=embedding_dim + action_dim, output_size=1, hidden_sizes=hidden_sizes)

    def forward(self, obs, action):
        obs = obs.view(-1, 3, self.image_dim, self.image_dim)
        embed_repr = torch.cat([self.encoder(obs), action], dim=1)
        return self.mlp_Q(embed_repr)


class ConvPolicy(nn.Module):
    def __init__(self, embedding_dim, image_dim, hidden_sizes, action_dim, policy_class, activation_function='relu', **kwargs):
        super().__init__()
        self.image_dim = image_dim
        self.encoder = VisualEncoder(embedding_dim, image_dim)
        if isinstance(policy_class, TanhGaussianPolicy):
            self.mlp_policy = policy_class(obs_dim=embedding_dim, action_dim=action_dim, hidden_sizes=hidden_sizes, **kwargs)
        elif isinstance(policy_class, TanhMlpPolicy):
            self.mlp_policy = policy_class(input_size=embedding_dim, output_size=action_dim, hidden_sizes=hidden_sizes, **kwargs)
        self.act_fn = getattr(F, activation_function)

    def forward(self, obs, **kwargs):
        obs = obs.view(-1, 3, self.image_dim, self.image_dim)
        embed_repr = self.encoder(obs)
        return self.mlp_policy(self.act_fn(embed_repr), **kwargs)

    def get_action(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0][0], {}

        # actions = self.get_actions(obs_np[None], deterministic=deterministic)
        # return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def reset(self):
        pass
