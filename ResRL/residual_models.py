import torch.nn as nn
from envs.env import Env
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import torch
from ResRL.models import *


class UNet(nn.Module):
    def __init__(self, image_dim, image_c, activation_function='relu'):
        super().__init__()
        self.image_dim = image_dim
        self.image_c = image_c
        self.act_fn = getattr(F, activation_function)
        if image_dim == 128:
            self.conv0 = nn.Conv1d(image_c, 4, 4, stride=2, padding=1)
            self.conv1 = nn.Conv1d(4, 8, 4, stride=2, padding=1)
        else:
            raise NotImplementedError
        self.conv2 = nn.Conv1d(8, 16, 4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(16, 32, 4, stride=2, padding=1)
        self.conv4 = nn.Conv1d(32, 64, 4, stride=2, padding=1)
        self.bottleneck = nn.Conv1d(64, 64, 3, stride=1, padding=1)

        self.upconv4 = nn.ConvTranspose1d(128, 32, 4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose1d(64, 16, 4, stride=2, padding=1)
        if self.image_dim == 128:
            self.upconv2 = nn.ConvTranspose1d(32, 8, 4, stride=2, padding=1)
            self.upconv1 = nn.ConvTranspose1d(16, 4, 4, stride=2, padding=1)
            self.upconv0 = nn.ConvTranspose1d(4, 2, 4, stride=2, padding=1)
        elif self.image_dim == 64:
            raise NotImplementedError

    def forward(self, observation):
        observation = observation.view(-1, self.image_c, self.image_dim, self.image_dim)
        observation = observation[:, :, :, self.image_dim // 2]  # Only take the one row in the middle
        if self.image_dim == 128:
            conv_feature0 = self.act_fn(self.conv0(observation))
            conv_feature1 = self.act_fn(self.conv1(conv_feature0))
        else:
            raise NotImplementedError
        conv_feature2 = self.act_fn(self.conv2(conv_feature1))
        conv_feature3 = self.act_fn(self.conv3(conv_feature2))
        conv_feature4 = self.act_fn(self.conv4(conv_feature3))
        bottleneck_feature = self.act_fn(self.bottleneck(conv_feature4))
        upconv_feature4 = self.upconv4(torch.cat([bottleneck_feature, conv_feature4], 1))
        upconv_feature3 = self.upconv3(torch.cat([upconv_feature4, conv_feature3], 1))
        upconv_feature2 = self.upconv2(torch.cat([upconv_feature3, conv_feature2], 1))
        upconv_feature1 = self.upconv1(torch.cat([upconv_feature2, conv_feature1], 1))
        upconv_feature0 = self.upconv0(upconv_feature1)
        return bottleneck_feature, upconv_feature0


class ResidualActor1D(nn.Module):
    def __init__(self, image_dim, image_c, action_dim, max_action, activation_function='relu',
                 attn_embed_dim=64, attn_num_heads=8):
        super().__init__()
        self.image_dim = image_dim
        self.image_c = image_c
        self.attn_embed_dim = attn_embed_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.act_fn = getattr(F, activation_function)
        self.unet = UNet(image_dim, image_c, activation_function)
        self.multihead_attn = MultiheadAttention(attn_embed_dim, attn_num_heads, kdim=2, vdim=2)
        self.action_encoder = ActionEncoder(attn_embed_dim, action_dim, activation_function)
        self.fc_bottleneck = nn.Linear(256, 256)
        self.fc1 = nn.Linear(attn_embed_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, observation):
        bottleneck_feature, upconv_feature0 = self.unet(observation)
        action_bottleneck = self.get_fc_action(bottleneck_feature.reshape(-1, 256))
        action_residual, attn_output_weights = self.get_MHA_action(upconv_feature0, action_bottleneck)
        action = F.tanh(action_bottleneck + action_residual) * self.max_action
        self.info = {'action_bottleneck': action_bottleneck,
                     'action_residual': action_residual,
                     'attn_action_output_weights': attn_output_weights}

        return action

    def get_fc_action(self, bottleneck_feature):
        hidden = self.act_fn(self.fc_bottleneck(bottleneck_feature))
        hidden = self.act_fn(self.fc2(hidden))
        action = self.fc3(hidden)
        return action

    def get_MHA_action(self, spatial_feature, action):
        N, H, C = spatial_feature.shape
        key = value = spatial_feature.contiguous().permute(2, 0, 1)
        action_embed = self.action_encoder(action)
        query = action_embed.view(1, -1, self.attn_embed_dim)
        # print(query.shape, key.shape)
        # print(self.multihead_attn.kdim, self.multihead_attn.vdim, self.multihead_attn.embed_dim)
        # print(self.multihead_attn._qkv_same_embed_dim)
        attn_output, attn_output_weights = self.multihead_attn(query, key, value, need_weights=True)
        attn_output, attn_output_weights = attn_output[0], attn_output_weights[0]
        hidden = self.act_fn(self.fc1(attn_output))
        hidden = self.act_fn(self.fc2(hidden))
        action = self.fc3(hidden)
        return action, attn_output_weights

    def get_info(self):
        return self.info


# class ActionEncoder(nn.Module):
#     def __init__(self, embedding_size, action_dim, activation_function='relu'):
#         super().__init__()
#         self.l1 = nn.Linear(action_dim, 64)
#         self.l2 = nn.Linear(64, embedding_size)
#         self.act_fn = getattr(F, activation_function)
#
#     def forward(self, action):
#         fa = self.act_fn(self.l1(action))
#         fa = self.act_fn(self.l2(fa))
#         return fa


class ResidualCriticSingle1D(nn.Module):
    def __init__(self, image_dim, image_c, action_dim, activation_function='relu',
                 attn_embed_dim=64, attn_num_heads=8):
        super().__init__()
        self.image_dim = image_dim
        self.image_c = image_c
        self.action_dim = action_dim
        self.attn_embed_dim = attn_embed_dim
        self.act_fn = getattr(F, activation_function)
        self.unet = UNet(image_dim, image_c, activation_function)
        self.multihead_attn = MultiheadAttention(attn_embed_dim, attn_num_heads, kdim=2, vdim=2)

        action_embed_dim = attn_embed_dim
        self.action_encoder = ActionEncoder(action_embed_dim, action_dim, activation_function)
        self.action_attn_encoder = ActionEncoder(attn_embed_dim, action_dim, activation_function)
        self.fc_bottleneck = nn.Linear(256 + action_embed_dim, 256)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, observation, action):
        bottleneck_feature, upconv_feature0 = self.unet(observation)
        value_bottleneck = self.get_fc_value(bottleneck_feature, action)
        value_residual, attn_output_weights = self.get_MHA_value(upconv_feature0, action)
        action = value_bottleneck + value_residual
        info = {'value_bottleneck': value_bottleneck,
                'value_residual': value_residual,
                'attn_value_output_weights': attn_output_weights}
        return action, info

    def get_fc_value(self, bottleneck_feature, action):
        action_embed = self.action_encoder(action)
        hidden = self.act_fn(self.fc_bottleneck(torch.cat([bottleneck_feature.reshape(-1, 256), action_embed], 1)))
        hidden = self.act_fn(self.fc2(hidden))
        action = self.fc3(hidden)
        return action

    def get_MHA_value(self, spatial_feature, action):
        N, H, C = spatial_feature.shape
        key = value = spatial_feature.contiguous().permute(2, 0, 1)
        action_embed = self.action_attn_encoder(action)
        query = action_embed.view(1, -1, self.attn_embed_dim)
        attn_output, attn_output_weights = self.multihead_attn(query, key, value, need_weights=True)
        hidden = self.act_fn(self.fc1(attn_output))
        hidden = self.act_fn(self.fc2(hidden))
        value = self.fc3(hidden)
        return value, attn_output_weights


class ResidualCritic1D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.critic1 = ResidualCriticSingle1D(*args, **kwargs)
        self.critic2 = ResidualCriticSingle1D(*args, **kwargs)

    def forward(self, observation, action):
        q1, info1 = self.critic1(observation, action)
        q2, info2 = self.critic2(observation, action)
        info1 = {k + '_q1': v for k, v in info1.items()}
        info2 = {k + '_q2': v for k, v in info2.items()}
        self.info = info1
        self.info.update(info2)
        return q1, q2

    def Q1(self, observation, action):
        q1, info1 = self.critic1(observation, action)
        self.info = {k + '_q1': v for k, v in info1.items()}
        return q1

    def get_info(self):
        return self.info
