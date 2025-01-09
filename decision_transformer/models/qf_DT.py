import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            rtg_scale=1,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_ctx=1024,
            **kwargs
        )
        self.config = config
        self.rtg_scale = rtg_scale

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        # self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        # self.predict_rewards = torch.nn.Linear(hidden_size, 1)
        # self.predict_returns = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards=None, targets=None, returns_to_go=None, timesteps=None, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (state_embeddings, returns_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer( inputs_embeds=stacked_inputs,
                                                attention_mask=stacked_attention_mask,)
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)
        action_preds = self.predict_action(x[:, 1])

        return action_preds

    def get_noise_action(self, critic, states, actions, rewards=None, returns_to_go=None, timesteps=None, repeat_num=10, **kwargs):
        states, actions, rewards, returns_to_go, timesteps, attention_mask = self.change_status(states, actions, rewards, returns_to_go, timesteps)
        with torch.no_grad():
            # rtg_preds = rewardToGo(states[:, -1], timesteps[:, -1])
            return_max_preds, return_mean_preds = critic(states, actions, rewards, None,
                                                         returns_to_go=returns_to_go,
                                                         timesteps=timesteps,
                                                         attention_mask=attention_mask,
                                                         **kwargs)
            index_end = returns_to_go.shape[1]
            rtg_temp_ = torch.zeros_like(returns_to_go, device=returns_to_go.device, dtype=returns_to_go.dtype)  # 1 * context_len * 1
            rtg_temp_[:, -1] = return_max_preds[:, -1]
            rtg_temp_ = rtg_temp_.repeat_interleave(repeats=repeat_num, dim=0)  # repeat_num * context_len * 1
            noise = torch.cat([torch.zeros(1), torch.randn(repeat_num - 1) * 0.05], dim=0).to(rtg_temp_.device)
            rtg_temp_[:, -1, 0] = rtg_temp_[:, -1, 0] + noise
            for i in range(repeat_num):
                for t in range(index_end - 2, -1, -1):
                    rtg_temp_[i, t, 0] = rtg_temp_[i, t + 1, 0] + rewards[0, t, 0] * self.rtg_scale
            states_ = states.repeat_interleave(repeats=repeat_num, dim=0)
            actions_ = actions.repeat_interleave(repeats=repeat_num, dim=0)
            rewards_ = rewards.repeat_interleave(repeats=repeat_num, dim=0)
            timesteps_ = timesteps.repeat_interleave(repeats=repeat_num, dim=0)
            attention_mask_ = attention_mask.repeat_interleave(repeats=repeat_num, dim=0)
            # _, actions_preds, _, _ = self.forward(states_, actions_, rewards_, None,
            actions_preds_ = self.forward(states_, actions_, rewards_, None,
                                         returns_to_go=rtg_temp_,
                                         timesteps=timesteps_,
                                         attention_mask=attention_mask_,
                                         **kwargs)
            # batch * K * dim
            actions_preds_[:, :-1] = actions_[:, :-1]
            return_max_preds_, return_mean_preds_ = critic(states_, actions_preds_, rewards_, None,
                                        returns_to_go=rtg_temp_,
                                        timesteps=timesteps_,
                                        attention_mask=attention_mask_,
                                        **kwargs)
            # batch * K * dim -> batch  * dim
            return_mean_preds_ = return_mean_preds_[:, -1].reshape(-1)
            idx = torch.multinomial(F.softmax(return_mean_preds_, dim=-1), 1).item()
        return actions_preds_[idx, -1], return_mean_preds_[idx].item()

    def get_rtg_action(self, critic, rewardToGo, states, actions, rewards=None, returns_to_go=None, timesteps=None, **kwargs):
        states, actions, rewards, returns_to_go, timesteps, attention_mask = self.change_status(states, actions, rewards, returns_to_go, timesteps)
        with torch.no_grad():
            return_max_preds, return_mean_preds = critic(states, actions, rewards, None,
                                                         returns_to_go=returns_to_go,
                                                         timesteps=timesteps,
                                                         attention_mask=attention_mask,
                                                         **kwargs)
            index_end = returns_to_go.shape[1]
            for t in range(index_end - 2, -1, -1):
                return_max_preds[0, t, 0] = return_max_preds[0, t + 1, 0] + rewards[0, t, 0] * self.rtg_scale
            actions_preds = self.forward(states, actions, rewards, None,
                                          returns_to_go=return_max_preds,
                                          timesteps=timesteps,
                                          attention_mask=attention_mask,
                                          **kwargs)
        return actions_preds[0, -1, :], return_max_preds[0, -1, 0].item()

    def change_status(self, states, actions, rewards=None, returns_to_go=None, timesteps=None):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        rewards = rewards.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        # if self.max_length is not None:
        states = states[:, -self.max_length:]
        actions = actions[:, -self.max_length:]
        rewards = rewards[:, -self.max_length:]
        timesteps = timesteps[:, -self.max_length:]
        returns_to_go = returns_to_go[:, -self.max_length:]
        # padding
        attention_mask = torch.cat([torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])])
        attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
        states = torch.cat(
            [torch.zeros(
                (
                    states.shape[0],
                    self.max_length - states.shape[1],
                    self.state_dim
                ),
                device=states.device),
                states
            ], dim=1).to(dtype=torch.float32)
        returns_to_go = torch.cat(
            [torch.zeros(
                (
                    returns_to_go.shape[0],
                    self.max_length - returns_to_go.shape[1],
                    1
                ),
                device=returns_to_go.device
            ),
                returns_to_go
            ], dim=1).to(dtype=torch.float32)
        timesteps = torch.cat(
            [torch.zeros(
                (
                    timesteps.shape[0],
                    self.max_length - timesteps.shape[1]
                ),
                device=timesteps.device
            ),
                timesteps
            ], dim=1).to(dtype=torch.long)
        rewards = torch.cat(
            [torch.zeros(
                (
                    rewards.shape[0],
                    self.max_length - rewards.shape[1],
                    1
                ),
                device=rewards.device),
                rewards
            ], dim=1).to(dtype=torch.float32)

        actions = torch.cat(
            [torch.zeros(
                (
                    actions.shape[0],
                    self.max_length - actions.shape[1],
                    self.act_dim),
                device=actions.device),
                actions
            ], dim=1).to(dtype=torch.float32)
        # else:
        #     attention_mask = None
        return states, actions, rewards, returns_to_go, timesteps, attention_mask


class ReturnTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            rtg_scale=1,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_ctx=1024,
            **kwargs
        )
        self.config = config
        self.rtg_scale = rtg_scale

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_mean_returns = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )
        self.predict_max_returns = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

    def forward(self, states, actions, rewards=None, targets=None, returns_to_go=None, timesteps=None, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer( inputs_embeds=stacked_inputs,
                                                attention_mask=stacked_attention_mask,)
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

        returns_max_preds = self.predict_max_returns(x[:, 0])
        returns_mean_preds = self.predict_mean_returns(x[:, 1])

        return returns_max_preds, returns_mean_preds

