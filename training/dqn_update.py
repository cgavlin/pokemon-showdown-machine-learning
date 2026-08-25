"""
Shared Double-DQN parameter-update step.

Used by training/trainer.py (Trainer), training/self_play_trainer.py
(SelfPlayTrainer), and training/pooled_self_play_trainer.py
(PooledSelfPlayTrainer) so all three trainers run the exact same
update math against a sampled replay batch. Before this existed, the
same ~20 lines were copy-pasted across trainers; fixing a bug here now
fixes it everywhere instead of needing synchronized edits in each file.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def double_dqn_update(
    q_network: nn.Module,
    target_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    device: torch.device,
    gamma: float,
    max_grad_norm: float = 10.0,
) -> float:
    """One Double-DQN gradient step on a sampled replay batch. Returns
    the scalar loss value for logging."""
    obs = torch.as_tensor(batch["obs"], device=device)
    next_obs = torch.as_tensor(batch["next_obs"], device=device)
    actions = torch.as_tensor(batch["actions"], device=device)
    rewards = torch.as_tensor(batch["rewards"], device=device)
    dones = torch.as_tensor(batch["dones"], device=device)
    next_masks = torch.as_tensor(batch["next_action_masks"], device=device)

    with torch.no_grad():
        next_q_online = q_network(next_obs)
        next_q_online_masked = next_q_online.masked_fill(next_masks == 0, -1e9)
        best_next_actions = next_q_online_masked.argmax(dim=-1, keepdim=True)

        next_q_target = target_network(next_obs)
        next_q_value = next_q_target.gather(1, best_next_actions).squeeze(-1)
        td_target = rewards + gamma * (1.0 - dones) * next_q_value

    q_values = q_network(obs)
    q_taken = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    loss = F.smooth_l1_loss(q_taken, td_target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=max_grad_norm)
    optimizer.step()

    return float(loss.item())
