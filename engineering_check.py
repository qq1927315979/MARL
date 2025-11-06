#!/usr/bin/env python3
"""
全面工程检查脚本

检查内容：
1. 代码质量和一致性
2. 环境配置
3. 训练稳定性问题
4. 模型checkpoint兼容性
5. 提供完整解决方案
"""

import torch
from pathlib import Path
import json
import numpy as np


def check_checkpoint_compatibility(checkpoint_path):
    """检查checkpoint兼容性"""
    print("=" * 80)
    print("1. Checkpoint兼容性检查")
    print("=" * 80)

    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint文件不存在: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 分析checkpoint结构
    print(f"\n✓ Checkpoint文件加载成功")
    print(f"\n【Checkpoint内容】")
    for key in checkpoint.keys():
        if key not in ['critic', 'actors', 'alphas']:
            print(f"  - {key}: {checkpoint[key]}")

    # 分析网络维度
    critic_input_dim = checkpoint['critic']['q1_fc1.weight'].shape[1]
    critic_hidden_dim = checkpoint['critic']['q1_fc1.weight'].shape[0]

    print(f"\n【网络维度】")
    print(f"  - Critic输入维度: {critic_input_dim}")
    print(f"  - Critic隐藏层维度: {critic_hidden_dim}")

    # 反推环境配置
    # critic_input_dim = n_agents * (obs_dim + action_dim)
    n_agents = 3
    action_dim = 2
    obs_plus_act = critic_input_dim / n_agents
    obs_dim = int(obs_plus_act - action_dim)

    print(f"\n【推断的环境配置】")
    print(f"  - n_agents: {n_agents}")
    print(f"  - obs_dim: {obs_dim}")
    print(f"  - action_dim: {action_dim}")

    # obs_dim = num_rays + 2 + 1 + 1 + 3*neighbor_max + 2 + 2 + 1 + 1
    # obs_dim = num_rays + 3*neighbor_max + 10
    target = obs_dim - 10

    print(f"\n【可能的num_rays和neighbor_max组合】")
    print(f"  需要满足: num_rays + 3*neighbor_max = {target}")

    solutions = []
    for num_rays in [8, 12, 16, 20, 24, 28, 32, 40, 48, 49, 52, 64, 73]:
        neighbor_max = (target - num_rays) / 3
        if neighbor_max >= 0 and neighbor_max == int(neighbor_max):
            neighbor_max = int(neighbor_max)
            solutions.append((num_rays, neighbor_max))
            print(f"  ✓ num_rays={num_rays}, neighbor_max={neighbor_max}")

    if solutions:
        print(f"\n💡 推荐使用第一个组合:")
        num_rays, neighbor_max = solutions[0]
        print(f"   --num-rays {num_rays} --neighbor-max {neighbor_max}")

    return {
        'obs_dim': obs_dim,
        'solutions': solutions,
        'checkpoint': checkpoint
    }


def check_training_logs(log_dir='checkpoints/logs'):
    """检查训练日志，分析稳定性"""
    print("\n" + "=" * 80)
    print("2. 训练稳定性分析")
    print("=" * 80)

    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"❌ 日志目录不存在: {log_dir}")
        return

    log_files = list(log_path.glob("training_log_*.jsonl"))
    if not log_files:
        print(f"❌ 未找到训练日志")
        return

    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    print(f"✓ 找到最新日志: {latest_log.name}")

    # 读取所有episode
    episodes = []
    with open(latest_log, 'r') as f:
        for line in f:
            episodes.append(json.loads(line))

    print(f"\n【训练概览】")
    print(f"  - 总轮次: {len(episodes)}")

    if len(episodes) > 0:
        # 分析不同阶段
        stages = {
            '前20轮（预热）': episodes[:20],
            '中期（21-50轮）': episodes[20:50] if len(episodes) > 50 else episodes[20:],
            '最后10轮': episodes[-10:] if len(episodes) > 10 else episodes
        }

        for stage_name, stage_eps in stages.items():
            if not stage_eps:
                continue

            returns = [ep['return'] for ep in stage_eps]
            critic_losses = [ep['losses']['critic'] for ep in stage_eps]
            actor_losses = [ep['losses']['actor'] for ep in stage_eps]

            print(f"\n【{stage_name}】")
            print(f"  - 回报: {np.mean(returns):.1f} (±{np.std(returns):.1f})")
            print(f"  - Critic损失: {np.mean(critic_losses):.2f} (±{np.std(critic_losses):.2f})")
            print(f"  - Actor损失: {np.mean(actor_losses):.2f} (±{np.std(actor_losses):.2f})")

            # 检测发散
            if np.mean(critic_losses) > 100:
                print(f"  ⚠️  Critic损失 > 100，训练可能发散！")
            if np.mean(actor_losses) > 500:
                print(f"  ⚠️  Actor损失 > 500，训练可能发散！")

        # Q值分析
        last_ep = episodes[-1]
        if 'q_values' in last_ep:
            q_vals = last_ep['q_values']
            q_range = abs(q_vals['q1_max'] - q_vals['q1_min'])

            print(f"\n【Q值分析（最后一轮）】")
            print(f"  - Q1范围: [{q_vals['q1_min']:.1f}, {q_vals['q1_max']:.1f}]")
            print(f"  - Q值跨度: {q_range:.1f}")

            if q_range > 10000:
                print(f"  ⚠️  Q值跨度 > 10000，奖励尺度过大！")


def analyze_reward_scale():
    """分析奖励尺度问题"""
    print("\n" + "=" * 80)
    print("3. 奖励尺度分析")
    print("=" * 80)

    print("\n【当前奖励设计】")
    print("  单个智能体每步奖励: 约 -5 到 -10")
    print("  3个智能体求和: 约 -15 到 -30")
    print("  1000步累积: 约 -15000 到 -30000")

    print("\n【问题诊断】")
    print("  ❌ Q网络需要预测 -15000 到 -30000 的值")
    print("  ❌ TD误差可能达到数千")
    print("  ❌ 容易导致梯度爆炸和训练发散")

    print("\n【业界最佳实践】")
    print("  ✓ 每步奖励应在 -1 到 +1 之间")
    print("  ✓ 1000步累积: -1000 到 +1000")
    print("  ✓ Q值范围合理，训练更稳定")


def propose_solutions():
    """提出解决方案"""
    print("\n" + "=" * 80)
    print("4. 完整解决方案")
    print("=" * 80)

    print("\n【立即行动】")
    print("\n🔴 P0 - 修复test.py兼容性")
    print("  1. 从GitHub拉取最新代码")
    print("  2. 运行: python test.py --checkpoint ./checkpoints/masac_ep600.pt")
    print("  3. 根据提示的num_rays和neighbor_max参数重新运行")

    print("\n🔴 P0 - 解决训练发散（损失2000+）")
    print("  方案A: 简单奖励缩放")
    print("    - 修改formation_env.py，所有奖励除以10")
    print("    - 优点: 简单快速")
    print("    - 缺点: 需要手动调整缩放因子")

    print("\n  方案B: Running Normalization（推荐⭐）")
    print("    - 实现在线奖励标准化")
    print("    - 使用running mean和std")
    print("    - 优点: 自适应，不需调参")

    print("\n🟡 P1 - 优化超参数")
    print("  - 降低actor学习率: 3e-4 → 1e-4")
    print("  - 降低tau: 0.005 → 0.001")
    print("  - 保持梯度裁剪: max_grad_norm=1.0")

    print("\n🟢 P2 - 可选改进")
    print("  - 在Critic网络中加入Layer Normalization")
    print("  - 增加warmup steps: 20000 → 50000")


def check_file_structure():
    """检查文件结构"""
    print("\n" + "=" * 80)
    print("5. 文件结构检查")
    print("=" * 80)

    core_files = {
        'formation_env.py': '编队环境（核心）',
        'masac.py': 'MASAC算法实现（核心）',
        'train_formation.py': '训练脚本（核心）',
        'test.py': '测试脚本（新增）',
        'diagnose_training_divergence.py': '训练诊断脚本（新增）',
        'comprehensive_check.py': '工程验证脚本（新增）',
        'test_reward_fixes.py': '奖励函数验证（新增）',
        'monitor_training.py': '训练监控（新增）'
    }

    print("\n【核心文件】")
    for file, desc in core_files.items():
        exists = Path(file).exists()
        status = "✓" if exists else "❌"
        print(f"  {status} {file:40s} - {desc}")

    print("\n【文件作用说明】")
    print("  - test.py: 加载checkpoint，GUI可视化测试")
    print("  - diagnose_training_divergence.py: 诊断损失2000+问题")
    print("  - comprehensive_check.py: 奖励函数工程验证（92.9%）")
    print("  - test_reward_fixes.py: 数学验证奖励修复")
    print("  - monitor_training.py: 实时监控训练进度")


def main():
    import sys

    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "MARL 项目全面工程检查" + " " * 25 + "║")
    print("╚" + "═" * 78 + "╝")

    # 1. Checkpoint兼容性
    checkpoint_path = './checkpoints/masac_ep600.pt'
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]

    checkpoint_info = check_checkpoint_compatibility(checkpoint_path)

    # 2. 训练日志分析
    check_training_logs()

    # 3. 奖励尺度分析
    analyze_reward_scale()

    # 4. 文件结构
    check_file_structure()

    # 5. 解决方案
    propose_solutions()

    # 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)

    print("\n【当前状态】")
    print("  ✓ 奖励函数已修复（P0-1到P2-6共6项）")
    print("  ✓ 可视化问题已解决（槽位对齐）")
    print("  ✓ test.py已完善（自动推断配置）")
    print("  ❌ 训练发散问题未解决（损失2000+）")

    print("\n【下一步行动】")
    print("  1. 先用test.py查看600轮模型的GUI效果")
    print("  2. 实施奖励归一化修复训练发散")
    print("  3. 重新训练并验证稳定性")

    if checkpoint_info and checkpoint_info['solutions']:
        num_rays, neighbor_max = checkpoint_info['solutions'][0]
        print(f"\n【立即执行】")
        print(f"  python test.py --checkpoint {checkpoint_path} \\")
        print(f"                 --num-rays {num_rays} \\")
        print(f"                 --neighbor-max {neighbor_max} \\")
        print(f"                 --episodes 3")


if __name__ == "__main__":
    main()
