from ResRL.residual_models import *


class UNetActor1D(nn.Module):
    def __init__(self, image_dim, image_c, action_dim, max_action, activation_function='relu',
                 attn_embed_dim=64):
        super().__init__()
        self.image_dim = image_dim
        self.image_c = image_c
        self.attn_embed_dim = attn_embed_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.act_fn = getattr(F, activation_function)
        self.unet = UNet(image_dim, image_c, activation_function)
        self.spatial_pool = nn.AvgPool1d(image_dim)
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, observation):
        _, upconv_feature0 = self.unet(observation)

        hidden = self.spatial_pool(upconv_feature0).squeeze(-1)
        hidden = self.act_fn(self.fc1(hidden))
        hidden = self.act_fn(self.fc2(hidden))
        action = F.tanh(self.fc3(hidden)) * self.max_action
        return action

    # def get_info(self):
    #     return self.info


class UNetCriticSingle1D(nn.Module):
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
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, observation, action):
        _, upconv_feature0 = self.unet(observation)
        value_residual, attn_output_weights = self.get_MHA_value(upconv_feature0, action)
        return value_residual

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


class UNetCritic1D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.critic1 = UNetCriticSingle1D(*args, **kwargs)
        self.critic2 = UNetCriticSingle1D(*args, **kwargs)

    def forward(self, observation, action):
        q1 = self.critic1(observation, action)
        q2 = self.critic2(observation, action)
        # info1 = {k + '_q1': v for k, v in info1.items()}
        # info2 = {k + '_q2': v for k, v in info2.items()}
        # self.info = info1
        # self.info.update(info2)
        return q1, q2

    def Q1(self, observation, action):
        q1 = self.critic1(observation, action)
        # self.info = {k + '_q1': v for k, v in info1.items()}
        return q1

    # def get_info(self):
    #     return self.info
