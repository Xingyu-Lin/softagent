from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F
from planet.models import RewardModel, Encoder, ObservationModel, bottle, bottle3


class ValueModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    @jit.script_method
    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief.detach(), state.detach()], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        value = self.fc3(hidden).squeeze(dim=1)
        return value


class ActionModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, action_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3_mean = nn.Linear(hidden_size, action_size)
        self.fc3_std = nn.Linear(hidden_size, action_size)

    @jit.script_method
    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief.detach(), state.detach()], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        action_mean = F.tanh(self.fc3_mean(hidden).squeeze(dim=1)) * 5
        action_std = F.softplus(self.fc3_std(hidden).squeeze(dim=1))
        action = action_mean + action_std * torch.randn_like(action_std)
        return F.tanh(action), F.tanh(action_mean)

    def act(self, belief, state, deterministic=False):
        action, action_mean = self.forward(belief, state)
        return action_mean if deterministic else action


class TransitionModel(jit.ScriptModule):
    __constants__ = ['min_std_dev']

    def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation_function='relu',
                 min_std_dev=0.1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)

    # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # a : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    @jit.script_method
    def forward(self, prev_state: torch.Tensor, actions: torch.Tensor, prev_belief: torch.Tensor,
                observations: Optional[torch.Tensor] = None, nonterminals: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """ :returns Poster-states s_{t+1} (a total of chunk_size -1), corresponding to action a_t"""
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, \
                                                                                                                    [torch.empty(0)] * T, \
                                                                                                                    [torch.empty(0)] * T, \
                                                                                                                    [torch.empty(0)] * T, \
                                                                                                                    [torch.empty(0)] * T, \
                                                                                                                    [torch.empty(0)] * T, \
                                                                                                                    [torch.empty(0)] * T
        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
        # Loop over time sequence
        for t in range(T - 1):
            _state = prior_states[t] if observations is None else posterior_states[t]  # Select appropriate previous state
            _state = _state if nonterminals is None else _state * nonterminals[t]  # Mask if previous transition was terminal
            # Compute belief (deterministic hidden state)
            hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])
            # Compute state prior by applying transition dynamics
            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                hidden = self.act_fn(
                    self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(
                    posterior_means[t + 1])
        # Return new hidden states
        hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0),
                  torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        if observations is not None:
            hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0),
                       torch.stack(posterior_std_devs[1:], dim=0)]
        return hidden

    # @jit.script_method
    def imagine(self, prev_state: torch.Tensor, prev_belief: torch.Tensor, action_model, horizon: int,
                nonterminals: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """ Imagine H time steps into the future with the provided action model.
            :return The imagined states, belief and actions"""
        # TODO use nonterminals to predict when the model will end
        beliefs, prior_states, prior_means, prior_std_devs, actions = [torch.empty(0)] * horizon, [torch.empty(0)] * horizon, \
                                                                      [torch.empty(0)] * horizon, [torch.empty(0)] * horizon, \
                                                                      [torch.empty(0)] * horizon
        beliefs[0], prior_states[0] = prev_belief, prev_state
        # Loop over time sequence
        for t in range(horizon - 1):
            _state = prior_states[t] if nonterminals is None else prior_states[t] * nonterminals[t]  # Mask if previous transition was terminal
            actions[t], _ = action_model(_state, beliefs[t])  # Only using the sampled actions
            # Compute belief (deterministic hidden state)
            hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])
            # Compute state prior by applying transition dynamics
            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
        hidden = [torch.stack(beliefs, dim=0), torch.stack(prior_states, dim=0), torch.stack(actions[:-1], dim=0)]
        return hidden
