#!/usr/bin/env python3
"""
诊断训练发散问题

分析600轮训练后损失达到2000+的根本原因
"""

import numpy as np
import json
from pathlib import Path

def analyze_reward_scale():
    """分析奖励尺度问题"""
    print("=" * 80)
    print("1. 奖励尺度分析")
    print("=" * 80)

    # 理论分析
    print("\n【理论奖励范围】")
    print("单个智能体每步奖励范围：约 -5 到 -10")
    print("3个智能体求和：约 -15 到 -30")
    print("1000步累积：约 -15000 到 -30000")
    print("\n问题：这个尺度太大！")
    print("  - Q网络需要预测-15000到-30000的值")
    print("  - TD误差也会很大")
    print("  - 容易导致梯度爆炸")

    # 建议的奖励范围
    print("\n【建议的奖励范围】")
    print("最佳实践：每步奖励在 -1 到 +1 之间")
    print("1000步累积：-1000 到 +1000")
    print("这样Q值范围合理，更容易稳定训练")

def analyze_gradient_issue():
    """分析梯度问题"""
    print("\n" + "=" * 80)
    print("2. 梯度裁剪分析")
    print("=" * 80)

    print("\n【当前设置】")
    print("max_grad_norm = 1.0")
    print("\n【问题】")
    print("当Q值范围在-15000到-30000时：")
    print("  - TD误差可能达到数千")
    print("  - 梯度会非常大")
    print("  - 裁剪到1.0可能太激进，导致学习太慢")
    print("  - 但不裁剪又会爆炸")
    print("\n【根本问题】：奖励尺度太大，导致梯度裁剪两难")

def analyze_learning_rate():
    """分析学习率"""
    print("\n" + "=" * 80)
    print("3. 学习率分析")
    print("=" * 80)

    print("\n【当前设置】")
    print("actor_lr = 3e-4")
    print("critic_lr = 3e-4")
    print("\n【问题】")
    print("对于大尺度Q值（-15000到-30000）：")
    print("  - 3e-4的学习率可能太高")
    print("  - 每次更新变化太大")
    print("  - 容易震荡和发散")

def analyze_network_capacity():
    """分析网络容量"""
    print("\n" + "=" * 80)
    print("4. 网络容量分析")
    print("=" * 80)

    print("\n【Q值预测范围】")
    print("需要预测：-15000 到 -30000")
    print("跨度：15000")
    print("\n【问题】")
    print("MLP网络预测这么大的值范围：")
    print("  - 需要很大的权重")
    print("  - 容易数值不稳定")
    print("  - 建议使用 Layer Normalization")

def propose_solutions():
    """提出解决方案"""
    print("\n" + "=" * 80)
    print("5. 解决方案（优先级排序）")
    print("=" * 80)

    print("\n🔴 【P0 - 必须修复】奖励归一化")
    print("方案1：固定缩放")
    print("  - 将所有奖励除以10或100")
    print("  - 简单有效，但需要手动调整")
    print("\n方案2：在线标准化（推荐）")
    print("  - 使用running mean和std标准化奖励")
    print("  - 自适应，不需要手动调整")
    print("  - 标准化后奖励均值0，方差1")

    print("\n🟡 【P1 - 强烈建议】降低学习率")
    print("  - actor_lr: 3e-4 → 1e-4")
    print("  - critic_lr: 3e-4 → 3e-4 (保持不变或降到1e-4)")

    print("\n🟡 【P1 - 强烈建议】调整梯度裁剪")
    print("  - 如果做了奖励归一化，max_grad_norm=1.0就够了")
    print("  - 如果不做归一化，建议增加到10.0")

    print("\n🟢 【P2 - 可选】网络改进")
    print("  - 在Critic网络中加入Layer Normalization")
    print("  - 有助于处理大范围的Q值")

    print("\n🟢 【P2 - 可选】调整tau")
    print("  - 当前tau=0.005可能太大")
    print("  - 建议降到0.001，让目标网络更稳定")

def check_training_logs():
    """检查现有的训练日志"""
    print("\n" + "=" * 80)
    print("6. 训练日志检查")
    print("=" * 80)

    # 尝试找最新的训练日志
    log_dir = Path("checkpoints/logs")
    if not log_dir.exists():
        print("\n未找到训练日志目录")
        return

    log_files = list(log_dir.glob("training_log_*.jsonl"))
    if not log_files:
        print("\n未找到训练日志文件")
        return

    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    print(f"\n最新日志：{latest_log}")

    # 读取并分析
    try:
        episodes_data = []
        with open(latest_log, 'r') as f:
            for line in f:
                episodes_data.append(json.loads(line))

        if not episodes_data:
            print("日志文件为空")
            return

        print(f"\n总轮次：{len(episodes_data)}")

        # 分析前几轮和最后几轮
        if len(episodes_data) >= 5:
            print("\n【前5轮】")
            for ep in episodes_data[:5]:
                print(f"  Ep {ep['episode']:3d}: Return={ep['return']:8.1f}, "
                      f"CriticLoss={ep['losses']['critic']:7.2f}, "
                      f"ActorLoss={ep['losses']['actor']:7.2f}")

            print("\n【最后5轮】")
            for ep in episodes_data[-5:]:
                print(f"  Ep {ep['episode']:3d}: Return={ep['return']:8.1f}, "
                      f"CriticLoss={ep['losses']['critic']:7.2f}, "
                      f"ActorLoss={ep['losses']['actor']:7.2f}")

            # 检测发散
            last_critic_loss = episodes_data[-1]['losses']['critic']
            last_actor_loss = episodes_data[-1]['losses']['actor']

            print(f"\n【发散检测】")
            if last_critic_loss > 100:
                print(f"  ⚠️  Critic损失={last_critic_loss:.1f} >> 100，已严重发散！")
            if last_actor_loss > 500:
                print(f"  ⚠️  Actor损失={last_actor_loss:.1f} >> 500，已严重发散！")

            # Q值分析
            if 'q_values' in episodes_data[-1]:
                q_vals = episodes_data[-1]['q_values']
                print(f"\n【Q值范围】")
                print(f"  Q1: [{q_vals['q1_min']:.1f}, {q_vals['q1_max']:.1f}]")
                print(f"  Q2: [{q_vals['q2_min']:.1f}, {q_vals['q2_max']:.1f}]")
                print(f"  Qt: [{q_vals['q_target_min']:.1f}, {q_vals['q_target_max']:.1f}]")

                q_range = abs(q_vals['q1_max'] - q_vals['q1_min'])
                print(f"  Q1跨度: {q_range:.1f}")
                if q_range > 10000:
                    print(f"  ⚠️  Q值跨度>10000，奖励尺度太大！")

    except Exception as e:
        print(f"分析日志时出错：{e}")

def main():
    print("\n" + "🔍" * 40)
    print("训练发散诊断报告")
    print("问题：600轮后，损失达到2000+")
    print("🔍" * 40)

    analyze_reward_scale()
    analyze_gradient_issue()
    analyze_learning_rate()
    analyze_network_capacity()
    check_training_logs()
    propose_solutions()

    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("""
根本原因：奖励尺度太大（每步-15到-30，1000步累积-15000到-30000）

立即行动：
1. 【必须】实现奖励归一化（除以10或使用running normalization）
2. 【强烈建议】降低actor学习率到1e-4
3. 【建议】如果不做归一化，增加梯度裁剪阈值到10.0

这三项修改应该能显著改善训练稳定性。
""")

if __name__ == "__main__":
    main()
