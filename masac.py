import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import math
from typing import Dict, Tuple, List
from copy import deepcopy
from contextlib import nullcontext


class Actor(nn.Module):
    def __init__(self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, deterministic: bool = False):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = dist.rsample()

        log_prob = dist.log_prob(pre_tanh).sum(dim=1, keepdim=True)
        log_2 = math.log(2.0)
        log_prob -= (2 * (log_2 - pre_tanh - F.softplus(-2 * pre_tanh))).sum(dim=1, keepdim=True)

        action = torch.tanh(pre_tanh)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self,
        total_obs_act_dim: int,
        hidden_dim: int = 256
    ):
        super(Critic, self).__init__()
        # Q1
        self.q1_fc1 = nn.Linear(total_obs_act_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        # Q2
        self.q2_fc1 = nn.Linear(total_obs_act_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

    def forward(self, obs_list: List[torch.Tensor], action_list: List[torch.Tensor]):
        if len(obs_list) != len(action_list):
            raise ValueError(f"obs_list and action_list must have same length, got {len(obs_list)} and {len(action_list)}")
        
        x = torch.cat(obs_list + action_list, dim=1)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)
        return q1, q2


class CentralReplayBuffer:
    def __init__(self,
        capacity: int,
        total_obs_dim: int,
        total_action_dim: int,
        device
    ):
        self.capacity = int(capacity)
        self.device = device
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((self.capacity, total_obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, total_action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, total_obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def add(self, obs_joint, actions_joint, reward, next_obs_joint, done):
        self.obs[self.ptr] = obs_joint
        self.actions[self.ptr] = actions_joint
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs_joint
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.actions[idx], device=self.device),
            torch.as_tensor(self.rewards[idx], device=self.device),
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.dones[idx], device=self.device),
        )

    def __len__(self):
        return self.size


class AdaptiveAlpha:
    def __init__(self,
        action_dim: int,
        initial_alpha: float = 0.2,
        lr: float = 3e-4,
        device: torch.device = torch.device('cpu')
    ):
        self.device = device
        # Store action_dim and scheduling fields for dynamic entropy/alpha control
        self.action_dim = float(action_dim)
        self.entropy_scale = 1.0
        self.alpha_min = 0.0
        self.target_entropy = -float(action_dim)
        self.log_alpha = nn.Parameter(torch.tensor(np.log(initial_alpha), dtype=torch.float32, device=self.device))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def update(self, log_pi: torch.Tensor):
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return alpha_loss.item()

    @property
    def alpha(self):
        a = self.log_alpha.exp()
        if self.alpha_min > 0.0:
            a = torch.clamp(a, min=self.alpha_min)
        return a

    # Scheduling helpers
    def set_entropy_scale(self, scale: float):
        try:
            s = float(scale)
        except Exception:
            s = 1.0
        self.entropy_scale = s
        self.target_entropy = -self.entropy_scale * self.action_dim

    def set_alpha_min(self, alpha_min: float):
        try:
            m = float(alpha_min)
        except Exception:
            m = 0.0
        self.alpha_min = m


class MASAC:
    def __init__(self,
        dim_info: Dict[str, Tuple[int, int]],
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        buffer_size: int = int(1e6),
        device: str = 'cpu',
        reward_type: str = 'mean',
        hidden_dim: int = 256,
        initial_alpha: float = 0.2,
        alpha_lr: float = 3e-4,
        max_grad_norm: float = 1.0
    ):
        self.device = torch.device(device)
        self.agent_ids = list(dim_info.keys())
        self.n_agents = len(self.agent_ids)
        self.reward_type = reward_type
        self.max_grad_norm = max_grad_norm
        # dim_info: dict with agent_id as key and (obs_dim, action_dim) as value
        self.obs_dims = [dim_info[aid][0] for aid in self.agent_ids]
        self.act_dims = [dim_info[aid][1] for aid in self.agent_ids]
        self.total_obs_dim = sum(self.obs_dims)
        self.total_act_dim = sum(self.act_dims)
        self.total_obs_act_dim = self.total_obs_dim + self.total_act_dim

        self.shared_critic = Critic(self.total_obs_act_dim, hidden_dim=hidden_dim).to(self.device)
        self.shared_critic_target = deepcopy(self.shared_critic)
        self.critic_optimizer = torch.optim.Adam(self.shared_critic.parameters(), lr=critic_lr)

        self.actors: Dict[str, Actor] = {}
        self.actor_optimizers: Dict[str, torch.optim.Adam] = {}
        self.alphas: Dict[str, AdaptiveAlpha] = {}

        for agent_id, (obs_dim, action_dim) in dim_info.items():
            self.actors[agent_id] = Actor(obs_dim, action_dim, hidden_dim=hidden_dim).to(self.device)
            self.actor_optimizers[agent_id] = torch.optim.Adam(
                self.actors[agent_id].parameters(), lr=actor_lr
            )
            self.alphas[agent_id] = AdaptiveAlpha(action_dim, initial_alpha=initial_alpha,
                                                    lr=alpha_lr, device=self.device)

        self.central_buffer = CentralReplayBuffer(buffer_size, self.total_obs_dim, self.total_act_dim, self.device)

        # For AMP (robust GradScaler init across PyTorch versions)
        self.scaler = None
        if self.device.type == 'cuda':
            try:
                self.scaler = torch.amp.GradScaler(device_type='cuda')
            except Exception:
                self.scaler = None

    # ============ Helpers to adjust entropy/alpha across all agents ============
    def set_entropy_scale_all(self, scale: float):
        for aa in self.alphas.values():
            aa.set_entropy_scale(scale)

    def set_alpha_min_all(self, alpha_min: float):
        for aa in self.alphas.values():
            aa.set_alpha_min(alpha_min)

    def _autocast(self, enabled: bool):
        """AMP autocast compatibility wrapper.
        Uses torch.amp.autocast for CUDA if available, otherwise returns nullcontext."""
        if not enabled or self.device.type != 'cuda':
            return nullcontext()
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            try:
                return torch.amp.autocast('cuda')
            except Exception:
                return nullcontext()
        return nullcontext()

    def _split_joint_obs(self, joint_obs: torch.Tensor) -> List[torch.Tensor]:
        splits = []
        idx = 0
        for d in self.obs_dims:
            splits.append(joint_obs[:, idx: idx + d])
            idx += d
        return splits

    def _split_joint_actions(self, joint_actions: torch.Tensor) -> List[torch.Tensor]:
        splits = []
        idx = 0
        for d in self.act_dims:
            splits.append(joint_actions[:, idx: idx + d])
            idx += d
        return splits

    def _check_input_consistency(self, obs_list: List[torch.Tensor], action_list: List[torch.Tensor]):
        if len(obs_list) != self.n_agents or len(action_list) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} obs and action tensors, got {len(obs_list)} and {len(action_list)}")
        
        for i, (obs, act) in enumerate(zip(obs_list, action_list)):
            if obs.shape[1] != self.obs_dims[i]:
                raise ValueError(f"Agent {self.agent_ids[i]} obs dim mismatch: expected {self.obs_dims[i]}, got {obs.shape[1]}")
            if act.shape[1] != self.act_dims[i]:
                raise ValueError(f"Agent {self.agent_ids[i]} action dim mismatch: expected {self.act_dims[i]}, got {act.shape[1]}")

    def select_action(self, obs_dict: Dict[str, np.ndarray], deterministic: bool = False):
        actions = {}
        for agent_id, obs in obs_dict.items():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _ = self.actors[agent_id](obs_tensor, deterministic=deterministic)
            actions[agent_id] = action.squeeze(0).cpu().numpy()
        return actions

    def store_transition(self, obs_dict, action_dict, reward_dict, next_obs_dict, done_dict, info_dict=None):
        obs_joint = np.concatenate([obs_dict[aid] for aid in self.agent_ids], axis=0).astype(np.float32)
        actions_joint = np.concatenate([action_dict[aid] for aid in self.agent_ids], axis=0).astype(np.float32)
        next_obs_joint = np.concatenate([next_obs_dict[aid] for aid in self.agent_ids], axis=0).astype(np.float32)

        if self.reward_type == 'sum':
            team_reward = np.sum([reward_dict[aid] for aid in self.agent_ids])
        elif self.reward_type == 'mean':
            team_reward = np.mean([reward_dict[aid] for aid in self.agent_ids])
        elif self.reward_type == 'global':
            team_reward = reward_dict.get('global', np.mean([reward_dict[aid] for aid in self.agent_ids]))
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")
            
        team_reward = np.array([team_reward], dtype=np.float32).reshape(1)
        # Episode terminal for bootstrapping: true only if collision termination (not time-limit)
        all_done = all(done_dict[aid] for aid in self.agent_ids)
        time_limit = False
        if info_dict is not None:
            try:
                time_limit = all(bool(info_dict[aid].get('time_limit', False)) for aid in self.agent_ids)
            except Exception:
                time_limit = False
        terminal_flag = float(all_done and not time_limit)
        done_flag = np.array([terminal_flag], dtype=np.float32).reshape(1)
        self.central_buffer.add(obs_joint, actions_joint, team_reward, next_obs_joint, done_flag)

    def update(self,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005
    ):
        if len(self.central_buffer) < batch_size:
            return {}

        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = self.central_buffer.sample(batch_size)

        all_obs = self._split_joint_obs(obs_batch)
        all_actions = self._split_joint_actions(actions_batch)
        all_next_obs = self._split_joint_obs(next_obs_batch)

        self._check_input_consistency(all_obs, all_actions)
        self._check_input_consistency(all_next_obs, all_actions)

        losses = {'critic': 0, 'actors': {}, 'alphas': {}}
        use_amp = self.device.type == 'cuda' and self.scaler is not None

        # Critic Update
        with self._autocast(use_amp):
            with torch.no_grad():
                next_actions = []
                next_log_probs = []
                for i, aid in enumerate(self.agent_ids):
                    next_a, next_logp = self.actors[aid](all_next_obs[i])
                    next_actions.append(next_a)
                    next_log_probs.append(next_logp)

                q1_next, q2_next = self.shared_critic_target(all_next_obs, next_actions)
                q_next = torch.min(q1_next, q2_next)

                entropy_term = torch.zeros_like(rewards_batch)
                for i, aid in enumerate(self.agent_ids):
                    entropy_term += self.alphas[aid].alpha.detach() * next_log_probs[i]
                # Average entropy across agents to match reward aggregation
                entropy_bonus = entropy_term / len(self.agent_ids)

                q_target = rewards_batch + gamma * (1 - dones_batch) * (q_next - entropy_bonus)

            q1, q2 = self.shared_critic(all_obs, all_actions)
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        if use_amp:
            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(self.critic_optimizer)
            critic_grad = torch.nn.utils.clip_grad_norm_(self.shared_critic.parameters(), self.max_grad_norm)
            self.scaler.step(self.critic_optimizer)
            self.scaler.update()
        else:
            critic_loss.backward()
            critic_grad = torch.nn.utils.clip_grad_norm_(self.shared_critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
        losses['critic'] = critic_loss.item()
        losses['q_target_mean'] = q_target.mean().item()
        losses['q1_mean'] = q1.mean().item()
        losses['q2_mean'] = q2.mean().item()
        losses['alpha_vals'] = {aid: float(self.alphas[aid].alpha.item()) for aid in self.agent_ids}
        # Additional diagnostics
        with torch.no_grad():
            td_error = (q_target - q1).abs()
            losses['td_error_mean'] = float(td_error.mean().item())
            losses['q1_min'] = float(q1.min().item())
            losses['q1_max'] = float(q1.max().item())
            losses['q2_min'] = float(q2.min().item())
            losses['q2_max'] = float(q2.max().item())
            losses['q_target_min'] = float(q_target.min().item())
            losses['q_target_max'] = float(q_target.max().item())
            # Mean log prob across agents and batch (from next_log_probs)
            try:
                lp_stack = torch.stack(next_log_probs)  # (n_agents, batch, 1)
                losses['logp_mean'] = float(lp_stack.mean().item())
            except Exception:
                losses['logp_mean'] = 0.0
        try:
            losses['grad_norm_critic'] = float(critic_grad if isinstance(critic_grad, (int, float)) else critic_grad.item())
        except Exception:
            try:
                # If tensor
                losses['grad_norm_critic'] = float(critic_grad.detach().cpu().item())
            except Exception:
                losses['grad_norm_critic'] = 0.0

        # Actor Update
        # To properly calculate the actor loss for each agent, we need to use the actions
        # from the *current* policies of all agents. Using actions from the replay buffer
        # (which are from old policies) would introduce significant off-policy error.
        # Detach observations to prevent gradient leak from critic update
        all_obs_detached = [obs.detach() for obs in all_obs]

        with torch.no_grad():
            current_policy_actions_detached = [self.actors[aid](all_obs_detached[i])[0] for i, aid in enumerate(self.agent_ids)]

        grad_actor_norms = {}
        for agent_idx, agent_id in enumerate(self.agent_ids):
            with self._autocast(use_amp):
                # Recompute the action for the current agent to get gradients
                current_actions, current_logp = self.actors[agent_id](all_obs_detached[agent_idx])

                # Create the list of actions for the Q-function input.
                # All actions are from current policies, but only the current agent's has a gradient.
                all_actions_for_q = list(current_policy_actions_detached)
                all_actions_for_q[agent_idx] = current_actions

                # The actor's loss is based on the Q-value from the critic.
                # Gradients should only flow through the actor, not the critic.
                # This is naturally handled as the actor_optimizer only contains actor params.
                q1_pi, q2_pi = self.shared_critic(all_obs_detached, all_actions_for_q)
                q_pi = torch.min(q1_pi, q2_pi)

                alpha_detached = self.alphas[agent_id].alpha.detach()
                actor_loss = (alpha_detached * current_logp - q_pi).mean()

            self.actor_optimizers[agent_id].zero_grad()
            if use_amp:
                self.scaler.scale(actor_loss).backward()
                self.scaler.unscale_(self.actor_optimizers[agent_id])
                a_grad = torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
                self.scaler.step(self.actor_optimizers[agent_id])
                self.scaler.update()
            else:
                actor_loss.backward()
                a_grad = torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
                self.actor_optimizers[agent_id].step()
            losses['actors'][agent_id] = actor_loss.item()

            alpha_loss = self.alphas[agent_id].update(current_logp.detach())
            losses['alphas'][agent_id] = alpha_loss
            try:
                grad_actor_norms[agent_id] = float(a_grad if isinstance(a_grad, (int, float)) else a_grad.item())
            except Exception:
                try:
                    grad_actor_norms[agent_id] = float(a_grad.detach().cpu().item())
                except Exception:
                    grad_actor_norms[agent_id] = 0.0

        # Soft Update (target network)
        self._soft_update(self.shared_critic, self.shared_critic_target, tau)
        losses['grad_norm_actor'] = grad_actor_norms
        return losses

    def _soft_update(self, source, target, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, path: str):
        hparams = {
            'hidden_dim': self.actors[self.agent_ids[0]].fc1.out_features
        }
        checkpoint = {
            'hparams': hparams,
            'critic': self.shared_critic.state_dict(),
            'actors': {aid: self.actors[aid].state_dict() for aid in self.agent_ids},
            # Save alphas on CPU for device-agnostic loading
            'alphas': {aid: self.alphas[aid].log_alpha.detach().cpu() for aid in self.agent_ids},
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_optimizers': {aid: self.actor_optimizers[aid].state_dict() for aid in self.agent_ids},
            'alpha_optimizers': {aid: self.alphas[aid].alpha_optimizer.state_dict() for aid in self.agent_ids},
            'scaler': self.scaler.state_dict() if self.scaler else None
        }
        torch.save(checkpoint, path)

    def load(self, checkpoint: dict):
        self.shared_critic.load_state_dict(checkpoint['critic'])
        self.shared_critic_target = deepcopy(self.shared_critic)
        for aid in self.agent_ids:
            self.actors[aid].load_state_dict(checkpoint['actors'][aid])
            # Load alpha onto current device safely
            self.alphas[aid].log_alpha.data.copy_(checkpoint['alphas'][aid].to(self.device))
            self.actor_optimizers[aid].load_state_dict(checkpoint['actor_optimizers'][aid])
            self.alphas[aid].alpha_optimizer.load_state_dict(checkpoint['alpha_optimizers'][aid])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if self.scaler and checkpoint.get('scaler'):
            self.scaler.load_state_dict(checkpoint['scaler'])


 
if __name__ == "__main__":
    # Test script for MASAC algorithm
    print("=" * 50)
    print("Testing MASAC Algorithm...")
    print("=" * 50)
    
    dim_info = {
        'agent_0': (8, 2),
        'agent_1': (8, 2),
        'agent_2': (8, 2)
    }
    masac = MASAC(dim_info, device='cuda' if torch.cuda.is_available() else 'cpu', buffer_size=10000)
    
    batch_size = 256
    buffer_population_steps = 1000
    update_steps = 10

    print(f"Populating buffer with {buffer_population_steps} transitions...")
    
    for _ in range(buffer_population_steps):
        obs_dict = {
            'agent_0': np.random.randn(8),
            'agent_1': np.random.randn(8),
            'agent_2': np.random.randn(8)
        }
        actions = masac.select_action(obs_dict)
        next_obs_dict = {k: np.random.randn(8) for k in obs_dict.keys()}
        reward_dict = {k: np.random.randn() for k in obs_dict.keys()}
        done_dict = {k: False for k in obs_dict.keys()}
        masac.store_transition(obs_dict, actions, reward_dict, next_obs_dict, done_dict)

    print(f"Buffer size: {len(masac.central_buffer)}")
    
    if len(masac.central_buffer) >= batch_size:
        print(f"\nRunning {update_steps} training updates...")
        for i in range(update_steps):
            losses = masac.update(batch_size=batch_size)
            if (i + 1) % 2 == 0:
                print(f"Update step {i+1}/{update_steps}, Losses: {{'critic': {losses.get('critic', 0):.4f}, 'actor_loss': {np.mean(list(losses.get('actors', {}).values())):.4f}}}")
    else:
        print("Buffer not full enough to update.")
    print("\nMASAC test completed!")
