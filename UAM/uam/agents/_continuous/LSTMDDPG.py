import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import uam.common.buffer as buffer
import uam.common.nets as nets
from uam import logger
from uam.agents.base import BaseAgent
from uam.common.configparser import ConfigFile
from uam.common.exploration import Gaussian_Noise


class LSTMDDPGAgent(BaseAgent):
    def __init__(self, c: ConfigFile, agent_name, init_critic=True):
        super().__init__(c, agent_name)

        # 属性和超参数
        self.lr_actor         = c.lr_actor
        self.lr_critic        = c.lr_critic
        self.tau              = c.tau
        self.actor_weights    = c.actor_weights
        self.critic_weights   = c.critic_weights
        self.net_struc_actor  = c.net_struc_actor
        self.net_struc_critic = c.net_struc_critic
        self.needs_history    = True
        self.history_length   = getattr(c.Agent, agent_name)["history_length"]
        self.use_past_actions = getattr(c.Agent, agent_name)["use_past_actions"]

        # 检查
        assert not (self.mode == "test" and (self.actor_weights is None or self.critic_weights is None)), "Need prior weights in test mode."

        if self.state_type == "image":
            raise NotImplementedError("Currently, image input is not supported for continuous action spaces.")

        if self.net_struc_actor is not None or self.net_struc_critic is not None:
            logger.warning("The net structure cannot be controlled via the config-spec for LSTM-based agents.")

        # 噪声
        self.noise = Gaussian_Noise(action_dim = self.num_actions)

        # 经验缓冲区
        if self.mode == "train":
            self.replay_buffer = buffer.UniformReplayBuffer_LSTM(state_type     = self.state_type, 
                                                                 state_shape    = self.state_shape, 
                                                                 buffer_length  = self.buffer_length,
                                                                 batch_size     = self.batch_size,
                                                                 device         = self.device,
                                                                 disc_actions   = False,
                                                                 action_dim     = self.num_actions,
                                                                 history_length = self.history_length)
        # 初始化演员和评论家
        if self.state_type == "feature":
            self.actor = nets.LSTM_Actor(state_shape      = self.state_shape,
                                         action_dim       = self.num_actions,
                                         use_past_actions = self.use_past_actions).to(self.device)
            
            if init_critic:
                self.critic = nets.LSTM_Critic(state_shape      = self.state_shape,
                                               action_dim       = self.num_actions,
                                               use_past_actions = self.use_past_actions).to(self.device)

        # 演员和评论家的参数个数
        if init_critic:
            self.n_params = self._count_params(self.actor), self._count_params(self.critic)

        # 加载先前的权重（如果可用）
        if self.actor_weights is not None and self.critic_weights is not None:
            self.actor.load_state_dict(torch.load(self.actor_weights, map_location=self.device))            
            
            if init_critic:
                self.critic.load_state_dict(torch.load(self.critic_weights, map_location=self.device))

        # 初始化目标网络
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        
        if init_critic:
            self.target_critic = copy.deepcopy(self.critic).to(self.device)
        
        # 相对于优化器冻结目标网络以避免不必要的计算
        for p in self.target_actor.parameters():
            p.requires_grad = False
        
        if init_critic:
            for p in self.target_critic.parameters():
                p.requires_grad = False

        # 定义优化器
        if self.optimizer == "Adam":
            self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
            if init_critic:
                self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        else:
            self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=self.lr_actor, alpha=0.95, centered=True, eps=0.01)
            if init_critic:
                self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=self.lr_critic, alpha=0.95, centered=True, eps=0.01)

    @torch.no_grad()
    def select_action(self, s, s_hist, a_hist, hist_len):
        """Selects action via actor network for a given state. Adds exploration bonus from noise and clips to action scale.
        s:        np.array with shape (state_shape,)
        s_hist:   np.array with shape (history_length, state_shape)
        a_hist:   np.array with shape (history_length, action_dim)
        hist_len: int
        
        returns: np.array with shape (action_dim,)
        """
        # reshape arguments and convert to tensors
        s = torch.tensor(s, dtype=torch.float32).view(1, self.state_shape).to(self.device)
        s_hist = torch.tensor(s_hist, dtype=torch.float32).view(1, self.history_length, self.state_shape).to(self.device)
        if a_hist is not None:
            a_hist = torch.tensor(a_hist, dtype=torch.float32).view(1, self.history_length, self.num_actions).to(self.device)
        hist_len = torch.tensor(hist_len).to(self.device)

        # 前向传播
        a, _ = self.actor(s, s_hist, a_hist, hist_len)
        
        # 添加噪声
        if self.mode == "train":
            a += torch.tensor(self.noise.sample()).to(self.device)
        
        # 在[-1,1]中剪辑动作
        return torch.clamp(a, -1, 1).cpu().numpy().reshape(self.num_actions)

    def memorize(self, s, a, r, s2, d):
        """将当前转换存储在重放缓冲区中"""
        self.replay_buffer.add(s, a, r, s2, d)

    def _compute_target(self, s2_hist, a2_hist, hist_len2, r, s2, d):
        with torch.no_grad():
            target_a, _ = self.target_actor(s=s2, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2)
                        
            # 下一个Q估计
            Q_next = self.target_critic(s=s2, a=target_a, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2, log_info=False)

            # 目标
            y = r + self.gamma * Q_next * (1 - d)
        return y

    def _compute_loss(self, Q, y, reduction="mean"):
        if self.loss == "MSELoss":
            return F.mse_loss(Q, y, reduction=reduction)

        elif self.loss == "SmoothL1Loss":
            return F.smooth_l1_loss(Q, y, reduction=reduction)

    def train(self):
        """来自经验缓冲区的样本，更新critic,actor."""
        # 样本批次
        batch = self.replay_buffer.sample()

        # 解包批次
        s_hist, a_hist, hist_len, s2_hist, a2_hist, hist_len2, s, a, r, s2, d = batch

        #-------- 训练 critic --------
        # 清空梯度
        self.critic_optimizer.zero_grad()
        
        # Q估计
        Q, critic_net_info = self.critic(s=s, a=a, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len, log_info=True)
 
        # 计算目标
        y = self._compute_target(s2_hist, a2_hist, hist_len2, r, s2, d)

        # 计算损失
        critic_loss = self._compute_loss(Q, y)

        # 计算梯度
        critic_loss.backward()

        # 梯度缩放和裁剪
        if self.grad_rescale:
            for p in self.critic.parameters():
                p.grad *= 1 / math.sqrt(2)
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10)
        
        # 执行优化步骤
        self.critic_optimizer.step()
        
        # 批评家培训日志
        self.logger.store(Critic_loss=critic_loss.detach().cpu().numpy().item(), **critic_net_info)
        self.logger.store(Q_val=Q.detach().mean().cpu().numpy().item())
        
        #-------- 训练actor --------
        # 冻结评论家，因此在训练演员时不会浪费梯度计算
        for param in self.critic.parameters():
            param.requires_grad = False
        
        # 清除梯度
        self.actor_optimizer.zero_grad()
        
        # 通过actor获取当前动作
        curr_a, act_net_info = self.actor(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
        
        # 计算损失，这是来自批评家的负Q值
        actor_loss = -self.critic(s=s, a=curr_a, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len, log_info=False).mean()

        # 计算梯度
        actor_loss.backward()
        
        # 梯度缩放和裁剪
        if self.grad_rescale:
            for p in self.actor.parameters():
                p.grad *= 1 / math.sqrt(2)
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
        
        # 使用优化器执行步骤
        self.actor_optimizer.step()

        # 解冻批评家，以便在下一次迭代中对其进行训练
        for param in self.critic.parameters():
            param.requires_grad = True
        
        # actor训练日志
        self.logger.store(Actor_loss=actor_loss.detach().cpu().numpy().item(), **act_net_info)
        
        #------- 更新目标网络 -------
        self.polyak_update()

    @torch.no_grad()
    def polyak_update(self):
        """目标网络权重的软更新."""
        for target_p, main_p in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_p.data.copy_(self.tau * main_p.data + (1-self.tau) * target_p.data)
        
        for target_p, main_p in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_p.data.copy_(self.tau * main_p.data + (1-self.tau) * target_p.data)
