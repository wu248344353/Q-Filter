import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import copy
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import CosineAnnealingLR


class EMA():
    '''
        empirical moving average
    '''

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer:

    def __init__(self,
                 model,
                 critic,
                 rewardToGo,
                 batch_size,
                 tau,
                 discount,
                 get_batch,
                 loss_fn,
                 eval_fns=None,
                 max_q_backup=False,
                 eta=1.0,
                 eta2=1.0,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 weight_decay=1e-4,
                 lr_decay=False,
                 lr_maxt=100000,
                 lr_min=0.,
                 grad_norm=1.0,
                 rtg_scale=1.0,
                 u_percent=0.99,
                 q_percent=0.05,
                 reward_scale=1.0,
                 ):

        self.actor = model
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=weight_decay)
        self.rewardToGo = rewardToGo
        self.rtg_optimizer = torch.optim.Adam(self.rewardToGo.parameters(), lr=lr)

        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=lr_min)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=lr_min)
            self.rewardToGo_lr_scheduler = CosineAnnealingLR(self.rtg_optimizer, T_max=lr_maxt, eta_min=lr_min)

        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.tau = tau
        self.max_q_backup = max_q_backup
        self.discount = discount
        self.grad_norm = grad_norm
        self.eta = eta
        self.eta2 = eta2
        self.lr_decay = lr_decay
        self.rtg_scale = rtg_scale
        self.percent = u_percent
        self.q_percent = q_percent
        self.reward_scale = reward_scale

        self.start_time = time.time()
        self.step = 0
        # self.train_a = True
        self.max_length = self.actor.max_length

    def step_ema(self):
        if self.step > self.step_start_ema and self.step % self.update_ema_every == 0:
            self.ema.update_model_average(self.ema_model, self.actor)

    def train_iteration(self, num_steps, logger, iter_num=0, log_writer=None):

        logs = dict()

        train_start = time.time()

        self.actor.train()
        self.critic.train()
        self.rewardToGo.train()
        loss_metric = {
            'bc_loss': [],
            'ql_loss': [],
            'rtg_loss': [],
            'actor_loss': [],
            'critic_loss': [],
            'target_q_mean': [],
        }
        for _ in trange(num_steps):
            loss_metric = self.train_step(log_writer, loss_metric)

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.rewardToGo_lr_scheduler.step()

        logger.record_tabular('BC Loss', np.mean(loss_metric['bc_loss']))
        logger.record_tabular('QL Loss', np.mean(loss_metric['ql_loss']))
        logger.record_tabular('Rtg Loss', np.mean(loss_metric['rtg_loss']))
        logger.record_tabular('Actor Loss', np.mean(loss_metric['actor_loss']))
        logger.record_tabular('Critic Loss', np.mean(loss_metric['critic_loss']))
        logger.record_tabular('Target Q Mean', np.mean(loss_metric['target_q_mean']))
        logger.dump_tabular()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.actor.eval()
        self.critic.eval()
        self.rewardToGo.eval()
        for eval_fn in self.eval_fns:
            # outputs = eval_fn(self.actor, self.critic_target, self.rewardToGo)
            outputs = eval_fn(self.actor, self.critic, self.rewardToGo)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        logger.log('=' * 80)
        logger.log(f'Iteration {iter_num}')
        best_ret = -10000
        best_nor_ret = -10000
        for k, v in logs.items():
            if 'return_mean' in k:
                best_ret = max(best_ret, float(v))
            if 'normalized_score' in k:
                best_nor_ret = max(best_nor_ret, float(v))
            logger.record_tabular(k, float(v))
        logger.record_tabular('Current actor learning rate', self.actor_optimizer.param_groups[0]['lr'])
        logger.record_tabular('Current critic learning rate', self.critic_optimizer.param_groups[0]['lr'])
        logger.dump_tabular()

        logs['Best_return_mean'] = best_ret
        logs['Best_normalized_score'] = best_nor_ret
        return logs

    def scale_up_eta(self, lambda_):
        self.eta2 = self.eta2 / lambda_

    def train_step(self, log_writer=None, loss_metric={}):
        '''
            Train the model for one step
            states: (batch_size, max_len, state_dim)
        '''
        states, actions, rewards, action_target, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        # action_target = torch.clone(actions)
        batch_size = states.shape[0]
        state_dim = states.shape[-1]
        action_dim = actions.shape[-1]
        device = states.device

        '''RTG Training'''
        rtg_predict = self.rewardToGo(states, timesteps)
        rtg_preds = rtg_predict.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        rtg_target = rtg[:, :-1].reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        # norm = rtg_target.abs().mean()
        # u = (rtg_target - rtg_preds) / norm
        u = rtg_target - rtg_preds
        rtg_loss = torch.mean(torch.abs(self.percent - (u < 0).float()) * u ** 2)
        self.rtg_optimizer.zero_grad()
        rtg_loss.backward()
        if self.grad_norm > 0:
            rtg_grad_norms = nn.utils.clip_grad_norm_(self.rewardToGo.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.rtg_optimizer.step()

        '''Q Training'''
        current_q1, current_q2 = self.critic.forward(states, actions)

        T = current_q1.shape[1]

        # rtg_preds, action_preds, state_preds, reward_preds
        with torch.no_grad():
            next_rtg = self.rewardToGo(states, timesteps)
            index_end = next_rtg.shape[1]
            for t in range(index_end - 2, -1, -1):
                # next_rtg[:, t, :] = next_rtg[:, t + 1, :] + rewards[:, t, :] / self.scale
                next_rtg[:, t, :] = next_rtg[:, t + 1, :] + rewards[:, t, :] * self.rtg_scale
            _, next_action, _, _ = self.ema_model(
                states, actions, rewards, action_target, next_rtg, timesteps, attention_mask=attention_mask,
            )

            critic_next_states = states[:, -1]
            critic_next_action = next_action[:, -1]
            target_q1, target_q2 = self.critic_target(critic_next_states, critic_next_action)
            target_q = torch.min(target_q1, target_q2)  # [B, 1]
            q_target = torch.zeros_like(rewards) # [B, T, 1]
            not_done = (1 - dones[:, -1])  # [B, 1]
            q_target[:, -1] = not_done * target_q
            for t in range(T-2, -1, -1):
                q_target[:, t] = rewards[:, t] / self.reward_scale + self.discount * q_target[:, t+1]
        critic_loss = F.mse_loss(current_q1[:, :-1][attention_mask[:, :-1] > 0],
                                 q_target[:, :-1][attention_mask[:, :-1] > 0].detach()) + F.mse_loss(
            current_q2[:, :-1][attention_mask[:, :-1] > 0], q_target[:, :-1][attention_mask[:, :-1] > 0].detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        '''Policy Training'''
        # state_preds, action_preds, reward_preds = self.actor.forward(
        rtg_preds, action_preds, state_preds, reward_preds = self.actor.forward(
            states, actions, rewards, action_target, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        )

        action_mask = (attention_mask.reshape(-1) > 0)
        if self.q_percent > 0.01:
            # 每个数据都不学还是某些动作不学
            q_target = self.critic.q_min(states, action_target).detach()  # batch * 1 * 1
            # q_target = self.critic.q_min(states[:, -1], action_target[:, -1]).detach()  # batch * 1 * 1
            q_percent = torch.quantile(q_target.reshape(-1), self.q_percent)
            action_mask = (attention_mask.reshape(-1) > 0) & (q_target.reshape(-1) > q_percent)
            # action_mask = (attention_mask.reshape(-1) > 0) & (q_target.repeat(1, states.shape[1], 1).reshape(-1) > q_percent)
        action_preds_ = action_preds.reshape(-1, action_dim)[action_mask.reshape(-1)]
        action_target_ = action_target.reshape(-1, action_dim)[action_mask.reshape(-1)]
        bc_loss = F.mse_loss(action_preds_, action_target_)
        # q_action_loss
        actor_states = states.reshape(-1, state_dim)[action_mask]
        q1_new_action, q2_new_action = self.critic(actor_states, action_preds.reshape(-1, action_dim)[action_mask])
        q_targets = self.critic.q_min(actor_states, action_target.reshape(-1, action_dim)[action_mask]).detach().abs().mean()
        # 用最小q更新还是两个q都用于更新
        q_loss = -(q1_new_action.mean() / q_targets + q2_new_action.mean() / q_targets)
        actor_loss = self.eta2 * bc_loss + self.eta * q_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.actor_optimizer.step()

        """ Step Target network """
        self.step_ema()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.step += 1

        with torch.no_grad():
            self.diagnostics['training/action_error'] = bc_loss.item()

        if log_writer is not None:
            if self.grad_norm > 0:
                log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
            log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
            log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
            log_writer.add_scalar('Rtg Loss', rtg_loss.item(), self.step)
            log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
            log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

        loss_metric['bc_loss'].append(bc_loss.item())
        loss_metric['ql_loss'].append(q_loss.item())
        loss_metric['rtg_loss'].append(rtg_loss.item())
        loss_metric['critic_loss'].append(critic_loss.item())
        loss_metric['actor_loss'].append(actor_loss.item())
        loss_metric['target_q_mean'].append(target_q.mean().item())

        return loss_metric

