#!/usr/bin/env python3
"""
测试脚本：加载训练好的模型并在GUI模式下可视化

用法:
    python test.py --checkpoint checkpoints/masac_ep100.pt --episodes 5

    如果没有checkpoint文件，会使用随机策略进行测试
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import time

from formation_env import MultiAgentRacecarFormationEnv
from masac import MASAC


def parse_args():
    parser = argparse.ArgumentParser(description='测试MASAC模型')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型checkpoint路径 (如果不指定，使用随机策略)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='测试episode数量')
    parser.add_argument('--n-agents', type=int, default=3,
                        help='智能体数量')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='每个episode最大步数')
    parser.add_argument('--gui', action='store_true', default=True,
                        help='启用GUI显示')
    parser.add_argument('--no-gui', dest='gui', action='store_false',
                        help='禁用GUI显示')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='使用确定性策略（不加噪声）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--slow-motion', type=float, default=0.0,
                        help='慢动作模式：每步后暂停的秒数（例如0.01）')
    return parser.parse_args()


def load_model(checkpoint_path, env, device='cpu'):
    """加载训练好的模型"""
    print(f"正在加载模型: {checkpoint_path}")

    # 创建MASAC实例
    agent_ids = [f'agent_{i}' for i in range(env.n_agents)]
    obs_dims = [env.observation_space[aid].shape[0] for aid in agent_ids]
    act_dims = [env.action_space[aid].shape[0] for aid in agent_ids]

    masac = MASAC(
        agent_ids=agent_ids,
        obs_dims=obs_dims,
        act_dims=act_dims,
        device=torch.device(device)
    )

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型参数
    masac.shared_critic.load_state_dict(checkpoint['critic_state_dict'])
    masac.shared_critic_target.load_state_dict(checkpoint['critic_target_state_dict'])

    for i, aid in enumerate(agent_ids):
        masac.actors[aid].load_state_dict(checkpoint[f'actor_{i}_state_dict'])
        masac.alphas[aid].alpha.data = checkpoint[f'alpha_{i}']

    print(f"✓ 模型加载成功！")
    print(f"  - Episode: {checkpoint.get('episode', 'unknown')}")
    print(f"  - Global steps: {checkpoint.get('global_steps', 'unknown')}")

    return masac


def test_episode(env, masac, max_steps=1000, deterministic=True, slow_motion=0.0):
    """运行单个测试episode"""
    obs_dict, info = env.reset()
    episode_rewards = {aid: 0.0 for aid in env.agent_ids}
    done = False
    step = 0

    slot_distances = []

    while not done and step < max_steps:
        # 选择动作
        if masac is not None:
            action_dict = masac.select_action(obs_dict, deterministic=deterministic)
        else:
            # 随机策略
            action_dict = {aid: env.action_space[aid].sample() for aid in env.agent_ids}

        # 执行动作
        next_obs_dict, reward_dict, done_dict, truncated_dict, info = env.step(action_dict)

        # 累积奖励
        for aid in env.agent_ids:
            episode_rewards[aid] += reward_dict[aid]

        # 记录槽位距离
        if 'slot_distances' in info:
            slot_distances.append(np.mean(info['slot_distances']))

        obs_dict = next_obs_dict
        done = all(done_dict.values()) or all(truncated_dict.values())
        step += 1

        # 慢动作模式
        if slow_motion > 0:
            time.sleep(slow_motion)

    # 统计信息
    total_reward = sum(episode_rewards.values())
    avg_slot_dist = np.mean(slot_distances) if slot_distances else 0.0
    final_slot_dist = slot_distances[-1] if slot_distances else 0.0
    min_slot_dist = np.min(slot_distances) if slot_distances else 0.0

    return {
        'total_reward': total_reward,
        'episode_rewards': episode_rewards,
        'steps': step,
        'avg_slot_distance': avg_slot_dist,
        'final_slot_distance': final_slot_dist,
        'min_slot_distance': min_slot_dist,
        'success': final_slot_dist < 1.0  # 成功阈值
    }


def main():
    args = parse_args()

    print("=" * 80)
    print("MASAC 模型测试")
    print("=" * 80)
    print(f"配置:")
    print(f"  - Checkpoint: {args.checkpoint if args.checkpoint else '无 (使用随机策略)'}")
    print(f"  - Episodes: {args.episodes}")
    print(f"  - GUI: {'是' if args.gui else '否'}")
    print(f"  - 确定性策略: {'是' if args.deterministic else '否'}")
    print(f"  - 慢动作: {args.slow_motion}s/step" if args.slow_motion > 0 else "")
    print("=" * 80)

    # 创建环境
    env = MultiAgentRacecarFormationEnv(
        n_agents=args.n_agents,
        gui=args.gui,
        seed=args.seed
    )

    # 加载模型
    masac = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"⚠️  Checkpoint文件不存在: {args.checkpoint}")
            print(f"⚠️  将使用随机策略进行测试")
        else:
            masac = load_model(args.checkpoint, env)
    else:
        print("未指定checkpoint，使用随机策略")

    print("\n开始测试...")
    print("-" * 80)

    # 运行测试episodes
    all_results = []
    for ep in range(args.episodes):
        print(f"\n[Episode {ep + 1}/{args.episodes}]")

        result = test_episode(
            env, masac,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
            slow_motion=args.slow_motion
        )
        all_results.append(result)

        # 打印结果
        print(f"  步数: {result['steps']}")
        print(f"  总奖励: {result['total_reward']:.2f}")
        print(f"  平均槽位距离: {result['avg_slot_distance']:.2f}m")
        print(f"  最终槽位距离: {result['final_slot_distance']:.2f}m")
        print(f"  最小槽位距离: {result['min_slot_distance']:.2f}m")
        print(f"  是否成功: {'✓' if result['success'] else '✗'}")

        # 每个智能体的奖励
        for aid, reward in result['episode_rewards'].items():
            print(f"    {aid}: {reward:.2f}")

    # 总结统计
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    avg_reward = np.mean([r['total_reward'] for r in all_results])
    avg_steps = np.mean([r['steps'] for r in all_results])
    avg_slot_dist = np.mean([r['avg_slot_distance'] for r in all_results])
    avg_final_dist = np.mean([r['final_slot_distance'] for r in all_results])
    avg_min_dist = np.mean([r['min_slot_distance'] for r in all_results])
    success_rate = np.mean([r['success'] for r in all_results]) * 100

    print(f"平均总奖励: {avg_reward:.2f} (±{np.std([r['total_reward'] for r in all_results]):.2f})")
    print(f"平均步数: {avg_steps:.1f}")
    print(f"平均槽位距离: {avg_slot_dist:.2f}m")
    print(f"平均最终槽位距离: {avg_final_dist:.2f}m")
    print(f"平均最小槽位距离: {avg_min_dist:.2f}m")
    print(f"成功率: {success_rate:.1f}%")

    # 最佳episode
    best_idx = np.argmax([r['total_reward'] for r in all_results])
    print(f"\n最佳Episode: #{best_idx + 1}")
    print(f"  - 奖励: {all_results[best_idx]['total_reward']:.2f}")
    print(f"  - 最终槽位距离: {all_results[best_idx]['final_slot_distance']:.2f}m")

    env.close()
    print("\n测试完成！")


if __name__ == "__main__":
    main()
