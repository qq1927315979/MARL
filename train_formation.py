import argparse
import os
import time
import json
from typing import Dict, Any

import numpy as np
import torch

from formation_env import MultiAgentRacecarFormationEnv
from masac import MASAC


def parse_args():
    p = argparse.ArgumentParser(description="Train/Eval MASAC on PyBullet multi-agent racecar formation env")

    # Env
    p.add_argument('--n-agents', type=int, default=3)
    p.add_argument('--num-rays', type=int, default=64)
    p.add_argument('--neighbor-max', type=int, default=3)
    p.add_argument('--lidar-max-dist', type=float, default=20.0)
    p.add_argument('--enable-obstacles', action='store_true', help='enable obstacles in the environment')
    p.add_argument('--obstacle-count', type=int, default=12)
    p.add_argument('--map-half-size', type=float, default=15.0)
    p.add_argument('--max-steps', type=int, default=1000, help='episode step limit (also passed to env)')
    p.add_argument('--gui', action='store_true', help='connect with PyBullet GUI')
    p.add_argument('--render', type=str, default=None, choices=['on', 'off', 'auto'],
                   help='render toggle when GUI is on: on/off, default auto=follow gui')
    p.add_argument('--no-randomize-formation', action='store_true', help='disable random formation on reset')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-boundaries', action='store_true', help='disable boundary walls around map')
    p.add_argument('--no-leader-traj', action='store_true', help='disable drawing leader trajectory')
    p.add_argument('--leader-traj-len', type=int, default=300, help='max number of segments to keep for leader trajectory')
    p.add_argument('--no-fixed-obstacles', action='store_true', help='disable fixed obstacles across episodes')

    # Algo
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--buffer-size', type=int, default=int(1e6))
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--actor-lr', type=float, default=3e-4)
    p.add_argument('--critic-lr', type=float, default=3e-4)
    p.add_argument('--alpha-lr', type=float, default=3e-4)
    p.add_argument('--initial-alpha', type=float, default=0.2)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--reward-type', type=str, default='mean', choices=['sum', 'mean', 'global'])
    # Entropy/alpha scheduling
    p.add_argument('--entropy-scale-start', type=float, default=1.5,
                   help='initial entropy scale: target_entropy = -scale * action_dim')
    p.add_argument('--entropy-scale-end', type=float, default=0.3,
                   help='final entropy scale after annealing')
    p.add_argument('--entropy-anneal-steps', type=int, default=500000,
                   help='steps over which to linearly anneal entropy scale')
    p.add_argument('--alpha-min-start', type=float, default=0.02,
                   help='initial minimum alpha (exploration floor)')
    p.add_argument('--alpha-min-end', type=float, default=0.0,
                   help='final minimum alpha after annealing')

    # Train/Eval
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--updates-per-step', type=int, default=2)
    p.add_argument('--warmup-steps', type=int, default=20000)  # Reduced from 80k for faster convergence
    p.add_argument('--deterministic-eval', action='store_true')
    p.add_argument('--eval', action='store_true', help='run evaluation only (no updates)')

    # IO
    p.add_argument('--save-every', type=int, default=200)
    p.add_argument('--save-dir', type=str, default='checkpoints')
    p.add_argument('--load', type=str, default=None, help='path to a checkpoint to load')

    return p.parse_args()


def make_env(args) -> MultiAgentRacecarFormationEnv:
    render_flag = None
    if args.render == 'on':
        render_flag = True
    elif args.render == 'off':
        render_flag = False
    # auto -> None (follow gui)

    env = MultiAgentRacecarFormationEnv(
        n_agents=args.n_agents,
        num_rays=args.num_rays,
        neighbor_max=args.neighbor_max,
        lidar_max_dist=args.lidar_max_dist,
        enable_obstacles=args.enable_obstacles,
        obstacle_count=args.obstacle_count,
        map_half_size=args.map_half_size,
        gui=args.gui,
        render=render_flag,
        seed=args.seed,
        frame_skip=4,
        randomize_formation_on_reset=(not args.no_randomize_formation),
        max_steps=args.max_steps,
        build_boundaries=(not args.no_boundaries),
        draw_leader_traj=(not args.no_leader_traj),
        leader_traj_max_len=args.leader_traj_len,
        fixed_obstacles=(not args.no_fixed_obstacles),
    )
    return env


def aggregate_reward(r: Dict[str, float], mode: str) -> float:
    if mode == 'sum':
        return float(np.sum(list(r.values())))
    if mode == 'mean':
        return float(np.mean(list(r.values())))
    # 'global' not provided by env; fallback to mean for logging
    return float(np.mean(list(r.values())))


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    log_dir = os.path.join(args.save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    detailed_log_path = os.path.join(log_dir, f'training_log_{timestamp}.jsonl')
    step_log_path = os.path.join(log_dir, f'step_log_{timestamp}.jsonl')

    try:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    env = make_env(args)
    dim_info = env.get_dim_info()

    masac = MASAC(
        dim_info,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        buffer_size=args.buffer_size,
        device=args.device,
        reward_type=args.reward_type,
        hidden_dim=args.hidden_dim,
        initial_alpha=args.initial_alpha,
        alpha_lr=args.alpha_lr,
    )

    try:
        masac.set_entropy_scale_all(args.entropy_scale_start)
        masac.set_alpha_min_all(args.alpha_min_start)
    except Exception:
        pass

    if args.load:
        ckpt = torch.load(args.load, map_location=args.device)
        masac.load(ckpt)
        print(f"[LOAD] {args.load}")

    global_steps = 0
    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = {aid: False for aid in env.agent_ids}
        ep_return = 0.0
        ep_len = 0
        t0 = time.time()

        slot_accum = 0.0
        critic_loss_sum = 0.0
        actor_loss_sum = 0.0
        update_count = 0
        q1_mean_sum = 0.0
        q2_mean_sum = 0.0
        q_target_mean_sum = 0.0
        alpha_sum_per_agent: Dict[str, float] = {aid: 0.0 for aid in env.agent_ids}

        per_agent_rewards = {aid: [] for aid in env.agent_ids}
        per_agent_slot_dists = {aid: [] for aid in env.agent_ids}
        collision_count = 0
        step_in_episode = 0

        # Initialize diagnostic variables - fix for locals() anti-pattern
        td_error_sum = 0.0
        logp_mean_sum = 0.0
        q1_min_sum = 0.0
        q1_max_sum = 0.0
        q2_min_sum = 0.0
        q2_max_sum = 0.0
        qt_min_sum = 0.0
        qt_max_sum = 0.0
        grad_critic_sum = 0.0
        grad_actor_sum_dict = {aid: 0.0 for aid in env.agent_ids}

        while not any(done.values()):
            step_in_episode += 1
            try:
                if args.entropy_anneal_steps > 0 and global_steps >= args.warmup_steps:
                    training_steps = global_steps - args.warmup_steps
                    frac = min(1.0, float(training_steps) / float(args.entropy_anneal_steps))
                else:
                    frac = 0.0
                cur_scale = args.entropy_scale_start + (args.entropy_scale_end - args.entropy_scale_start) * frac
                cur_alpha_min = args.alpha_min_start + (args.alpha_min_end - args.alpha_min_start) * frac
                masac.set_entropy_scale_all(cur_scale)
                masac.set_alpha_min_all(cur_alpha_min)
            except Exception:
                pass

            deterministic = args.eval and args.deterministic_eval
            actions = masac.select_action(obs, deterministic=deterministic)
            next_obs, reward, done, info = env.step(actions)

            step_log = {
                "episode": ep,
                "step": step_in_episode,
                "global_step": global_steps,
                "agents": {}
            }

            for aid in env.agent_ids:
                per_agent_rewards[aid].append(float(reward.get(aid, 0.0)))
                per_agent_slot_dists[aid].append(float(info[aid].get('slot_distance', 0.0)))
                if info[aid].get('collision', False):
                    collision_count += 1

                step_log["agents"][aid] = {
                    "action": actions[aid].tolist() if hasattr(actions[aid], 'tolist') else list(actions[aid]),
                    "reward": float(reward.get(aid, 0.0)),
                    "slot_distance": float(info[aid].get('slot_distance', 0.0)),
                    "collision": bool(info[aid].get('collision', False)),
                    "vel_mismatch": float(info[aid].get('vel_mismatch', 0.0)),
                    # Extended diagnostics for non-convergence analysis
                    "reward_breakdown": info[aid].get('reward_breakdown', {}),
                    "min_obstacle_dist": float(info[aid].get('min_obstacle_dist', 0.0)),
                    "nearest_neighbor_dist": float(info[aid].get('nearest_neighbor_dist', 0.0)),
                    "action_change": float(info[aid].get('action_change', 0.0)),
                    "norm_jerk": float(info[aid].get('norm_jerk', 0.0)),
                    "dist_delta": float(info[aid].get('dist_delta', 0.0)),
                    "time_limit": bool(info[aid].get('time_limit', False)),
                }

            with open(step_log_path, 'a') as f:
                f.write(json.dumps(step_log) + '\n')

            if not args.eval:
                masac.store_transition(obs, actions, reward, next_obs, done, info)
                if global_steps >= args.warmup_steps:
                    for _ in range(args.updates_per_step):
                        losses = masac.update(batch_size=args.batch_size, gamma=args.gamma, tau=args.tau)
                        if losses:
                            critic_loss_sum += float(losses.get('critic', 0.0))
                            actor_losses = list(losses.get('actors', {}).values())
                            if len(actor_losses) > 0:
                                actor_loss_sum += float(np.mean(actor_losses))
                            if 'q1_mean' in losses:
                                q1_mean_sum += float(losses['q1_mean'])
                            if 'q2_mean' in losses:
                                q2_mean_sum += float(losses['q2_mean'])
                            if 'q_target_mean' in losses:
                                q_target_mean_sum += float(losses['q_target_mean'])
                            if 'alpha_vals' in losses and isinstance(losses['alpha_vals'], dict):
                                for aid, aval in losses['alpha_vals'].items():
                                    if aid in alpha_sum_per_agent:
                                        alpha_sum_per_agent[aid] += float(aval)
                            # Additional diagnostic aggregations (fixed: removed locals() anti-pattern)
                            td_error_sum += float(losses.get('td_error_mean', 0.0))
                            logp_mean_sum += float(losses.get('logp_mean', 0.0))
                            q1_min_sum += float(losses.get('q1_min', 0.0))
                            q1_max_sum += float(losses.get('q1_max', 0.0))
                            q2_min_sum += float(losses.get('q2_min', 0.0))
                            q2_max_sum += float(losses.get('q2_max', 0.0))
                            qt_min_sum += float(losses.get('q_target_min', 0.0))
                            qt_max_sum += float(losses.get('q_target_max', 0.0))
                            grad_critic_sum += float(losses.get('grad_norm_critic', 0.0))
                            # Per-agent actor grad norms
                            if 'grad_norm_actor' in losses and isinstance(losses['grad_norm_actor'], dict):
                                for aid in env.agent_ids:
                                    grad_actor_sum_dict[aid] += float(losses['grad_norm_actor'].get(aid, 0.0))
                            update_count += 1
                global_steps += 1

            obs = next_obs
            ep_return += aggregate_reward(reward, args.reward_type)
            ep_len += 1
            slot_accum += float(np.mean([info[a]['slot_distance'] for a in env.agent_ids]))

        dt = time.time() - t0
        avg_slot_dist = (slot_accum / max(1, ep_len)) if env.agent_ids else 0.0

        if update_count > 0:
            critic_loss_avg = critic_loss_sum / update_count
            actor_loss_avg = actor_loss_sum / update_count if actor_loss_sum != 0 else 0.0
            q1_mean_avg = q1_mean_sum / update_count if q1_mean_sum != 0 else 0.0
            q2_mean_avg = q2_mean_sum / update_count if q2_mean_sum != 0 else 0.0
            q_target_mean_avg = q_target_mean_sum / update_count if q_target_mean_sum != 0 else 0.0

            alpha_dict = {aid: alpha_sum_per_agent[aid] / update_count if update_count > 0 else 0.0
                         for aid in env.agent_ids}
            # Compute averages for diagnostics (fixed: variables now always initialized)
            td_error_avg = td_error_sum / update_count
            logp_mean_avg = logp_mean_sum / update_count
            q1_min_avg = q1_min_sum / update_count
            q1_max_avg = q1_max_sum / update_count
            q2_min_avg = q2_min_sum / update_count
            q2_max_avg = q2_max_sum / update_count
            qt_min_avg = qt_min_sum / update_count
            qt_max_avg = qt_max_sum / update_count
            grad_critic_avg = grad_critic_sum / update_count
            grad_actor_avg = {aid: grad_actor_sum_dict[aid] / update_count for aid in env.agent_ids}
        else:
            critic_loss_avg = 0.0
            actor_loss_avg = 0.0
            q1_mean_avg = 0.0
            q2_mean_avg = 0.0
            q_target_mean_avg = 0.0
            alpha_dict = {aid: 0.0 for aid in env.agent_ids}
            td_error_avg = 0.0
            logp_mean_avg = 0.0
            q1_min_avg = 0.0
            q1_max_avg = 0.0
            q2_min_avg = 0.0
            q2_max_avg = 0.0
            qt_min_avg = 0.0
            qt_max_avg = 0.0
            grad_critic_avg = 0.0
            grad_actor_avg = {aid: 0.0 for aid in env.agent_ids}

        # Print with Q-value monitoring for convergence diagnosis
        alpha_avg = sum(alpha_dict.values()) / len(alpha_dict) if alpha_dict else 0.0
        print(f"[EP {ep:4d}/{args.episodes}] Steps={ep_len:3d} | R={ep_return:7.2f} | Slot={avg_slot_dist:.2f} | "
              f"C={critic_loss_avg:.4f} A={actor_loss_avg:.4f} | Q={q1_mean_avg:.2f} Qt={q_target_mean_avg:.2f} | "
              f"α={alpha_avg:.3f} | {dt:.1f}s")

        detailed_log = {
            "episode": ep,
            "global_steps": global_steps,
            "steps": ep_len,
            "return": float(ep_return),
            "avg_slot_distance": float(avg_slot_dist),
            "time_seconds": float(dt),
            "losses": {
                "critic": float(critic_loss_avg),
                "actor": float(actor_loss_avg),
            },
            "q_values": {
                "q1_mean": float(q1_mean_avg),
                "q2_mean": float(q2_mean_avg),
                "q_target_mean": float(q_target_mean_avg),
                "q1_min": float(q1_min_avg),
                "q1_max": float(q1_max_avg),
                "q2_min": float(q2_min_avg),
                "q2_max": float(q2_max_avg),
                "q_target_min": float(qt_min_avg),
                "q_target_max": float(qt_max_avg),
            },
            "alpha": alpha_dict,
            "diagnostics": {
                "td_error_mean": float(td_error_avg),
                "logp_mean": float(logp_mean_avg),
                "grad_norm_critic": float(grad_critic_avg),
                "grad_norm_actor": {aid: float(grad_actor_avg.get(aid, 0.0)) for aid in env.agent_ids},
            },
            "per_agent_episode_stats": {
                aid: {
                    "mean_reward": float(np.mean(per_agent_rewards[aid])),
                    "std_reward": float(np.std(per_agent_rewards[aid])),
                    "min_reward": float(np.min(per_agent_rewards[aid])),
                    "max_reward": float(np.max(per_agent_rewards[aid])),
                    "mean_slot_dist": float(np.mean(per_agent_slot_dists[aid])),
                    "std_slot_dist": float(np.std(per_agent_slot_dists[aid])),
                    "min_slot_dist": float(np.min(per_agent_slot_dists[aid])),
                    "max_slot_dist": float(np.max(per_agent_slot_dists[aid])),
                } for aid in env.agent_ids
            },
            "collision_count": collision_count,
            "warmup": update_count == 0,
            "update_count": update_count,
        }

        with open(detailed_log_path, 'a') as f:
            f.write(json.dumps(detailed_log) + '\n')

        if not args.eval and (ep % args.save_every == 0):
            path = os.path.join(args.save_dir, f"masac_ep{ep}.pt")
            masac.save(path)
            print(f"[SAVE] Episode {ep}")

    if not args.eval:
        final_path = os.path.join(args.save_dir, "masac_final.pt")
        masac.save(final_path)
        print(f"[SAVE] Final model")

    env.close()


if __name__ == '__main__':
    main()
