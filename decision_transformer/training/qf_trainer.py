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
                 batch_size,
                 tau,
                 discount,
                 get_batch,
                 loss_fn,
                 eval_fns=None,
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

        # self.step_start_ema = step_start_ema
        # self.ema = EMA(ema_decay)
        # self.ema_model = copy.deepcopy(self.actor)
        # self.update_ema_every = update_ema_every

        self.critic = critic
        # self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, weight_decay=weight_decay)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=lr_min)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=lr_min)

        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.tau = tau
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

    # def step_ema(self):
    #     if self.step > self.step_start_ema and self.step % self.update_ema_every == 0:
    #         self.ema.update_model_average(self.ema_model, self.actor)
    #         self.ema.update_model_average(self.critic_target, self.critic)

    def train_iteration(self, num_steps, logger, iter_num=0, log_writer=None):

        logs = dict()

        train_start = time.time()

        self.actor.train()
        self.critic.train()
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
        for eval_fn in self.eval_fns:
            # outputs = eval_fn(self.actor, self.critic_target, self.rewardToGo)
            outputs = eval_fn(self.actor, self.critic)
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

    # def scale_up_eta(self, lambda_):
    #     self.eta2 = self.eta2 / lambda_

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
        train_mask = (attention_mask.reshape(-1) > 0)

        returns_max_preds, returns_mean_preds = self.critic.forward(
            states, actions, rewards, action_target, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        )
        '''Max RTG loss'''
        rtg_max_preds = returns_max_preds.reshape(-1, 1)[train_mask]
        rtg_target = rtg[:, :-1].reshape(-1, 1)[train_mask]
        u = rtg_target - rtg_max_preds
        rtg_loss = torch.mean(torch.abs(self.percent - (u < 0).float()) * u ** 2)

        '''Mean RTG loss'''
        rtg_mean_preds = returns_mean_preds.reshape(-1, 1)[train_mask]
        critic_loss = F.mse_loss(rtg_mean_preds, rtg_target)

        '''RTG Function Backward'''
        return_loss = rtg_loss + critic_loss

        self.critic_optimizer.zero_grad()
        return_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        '''Policy Training'''
        # state_preds, action_preds, reward_preds = self.actor.forward(
        action_preds = self.actor.forward(
            states, actions, rewards, action_target, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        )

        action_preds_ = action_preds.reshape(-1, action_dim)[train_mask]
        action_target_ = action_target.reshape(-1, action_dim)[train_mask]
        bc_loss = F.mse_loss(action_preds_, action_target_)
        # q_action_loss
        returns_max_preds_, returns_mean_preds_ = self.critic.forward(
            states, action_preds, rewards, action_target, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        )
        rtg_preds_ = returns_mean_preds_.reshape(-1, 1)[train_mask]
        q_loss = - rtg_preds_.mean()

        actor_loss = self.eta2 * bc_loss + self.eta * q_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.actor_optimizer.step()

        # """ Step Target network """
        # self.step_ema()

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
            # log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)
            log_writer.add_scalar('Target_Q Mean', returns_mean_preds_.mean().item(), self.step)

        loss_metric['bc_loss'].append(bc_loss.item())
        loss_metric['ql_loss'].append(q_loss.item())
        loss_metric['rtg_loss'].append(rtg_loss.item())
        loss_metric['critic_loss'].append(critic_loss.item())
        loss_metric['actor_loss'].append(actor_loss.item())
        # loss_metric['target_q_mean'].append(target_q.mean().item())
        loss_metric['target_q_mean'].append(returns_mean_preds_.mean().item())

        return loss_metric

