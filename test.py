#!/usr/bin/env python3
"""
测试脚本：加载训练好的模型并在GUI模式下可视化

用法:
    python test.py --checkpoint checkpoints/masac_ep600.pt --episodes 5

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

    # 环境配置参数（用于匹配训练时的配置）
    parser.add_argument('--num-rays', type=int, default=16,
                        help='激光雷达射线数量（需要和训练时一致）')
    parser.add_argument('--neighbor-max', type=int, default=8,
                        help='最大邻居数量（需要和训练时一致）')

    return parser.parse_args()


def infer_env_config_from_checkpoint(checkpoint, n_agents=3):
    """从checkpoint推断环境配置"""
    # 从critic网络的第一层权重推断总输入维度
    critic_input_dim = checkpoint['critic']['q1_fc1.weight'].shape[1]

    # Critic输入 = n_agents * (obs_dim + action_dim)
    # action_dim = 2 (固定)
    obs_plus_act = critic_input_dim // n_agents
    obs_dim = obs_plus_act - 2  # action_dim = 2

    # obs_dim = num_rays + 2 + 1 + 1 + 3*neighbor_max + 2 + 2 + 1 + 1
    # obs_dim = num_rays + 3*neighbor_max + 10
    # 需要推断num_rays和neighbor_max的组合

    # 尝试常见的配置（扩大搜索范围）
    for num_rays in [8, 12, 16, 20, 24, 28, 32, 40, 48, 49, 52, 64, 73]:
        for neighbor_max in range(0, 25):  # 0到24
            expected_obs_dim = num_rays + 3 * neighbor_max + 10
            if expected_obs_dim == obs_dim:
                return num_rays, neighbor_max

    # 如果找不到，使用默认值并警告
    print(f"⚠️  无法从checkpoint推断环境配置！")
    print(f"   期望obs_dim={obs_dim}, 但找不到匹配的num_rays和neighbor_max组合")
    print(f"   使用默认值: num_rays=16, neighbor_max=8")
    return 16, 8


def load_model(checkpoint_path, env, device='cpu'):
    """加载训练好的模型"""
    print(f"正在加载模型: {checkpoint_path}")

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 创建MASAC实例 - 使用dim_info字典
    dim_info = env.get_dim_info()  # 返回 {agent_id: (obs_dim, action_dim)}

    masac = MASAC(
        dim_info=dim_info,
        device=device
    )

    # 使用MASAC自带的load方法
    try:
        masac.load(checkpoint)
        print(f"✓ 模型加载成功！")
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"❌ 模型维度不匹配！")
            print(f"   错误详情: {e}")
            print(f"\n💡 这通常是因为训练时和测试时的环境配置不同")
            print(f"   当前环境obs_dim={env.obs_dim}")

            # 尝试推断正确的配置
            critic_input_dim = checkpoint['critic']['q1_fc1.weight'].shape[1]
            print(f"   checkpoint期望的critic输入维度={critic_input_dim}")

            num_rays, neighbor_max = infer_env_config_from_checkpoint(checkpoint, env.n_agents)
            print(f"\n推荐的解决方案：")
            print(f"   python test.py --checkpoint {checkpoint_path} \\")
            print(f"                  --num-rays {num_rays} \\")
            print(f"                  --neighbor-max {neighbor_max}")
            raise
        else:
            raise

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
            action_dict = {aid: np.random.uniform(-1, 1, size=env.action_dim).astype(np.float32)
                          for aid in env.agent_ids}

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
    if args.slow_motion > 0:
        print(f"  - 慢动作: {args.slow_motion}s/step")
    print(f"  - 环境配置: num_rays={args.num_rays}, neighbor_max={args.neighbor_max}")
    print("=" * 80)

    # 创建环境 - 使用命令行参数指定的配置
    env = MultiAgentRacecarFormationEnv(
        n_agents=args.n_agents,
        num_rays=args.num_rays,
        neighbor_max=args.neighbor_max,
        gui=args.gui,
        seed=args.seed
    )

    print(f"环境信息: obs_dim={env.obs_dim}, action_dim={env.action_dim}")

    # 加载模型
    masac = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"⚠️  Checkpoint文件不存在: {args.checkpoint}")
            print(f"⚠️  将使用随机策略进行测试")
        else:
            try:
                masac = load_model(args.checkpoint, env)
            except RuntimeError:
                print(f"\n❌ 加载失败！请根据上面的提示调整命令行参数后重试。")
                env.close()
                return
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
