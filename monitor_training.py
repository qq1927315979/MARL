#!/usr/bin/env python3
"""Real-time training monitor for MASAC formation control"""

import json
import sys
import time
import os
from pathlib import Path

def get_latest_log():
    """Find the most recent training log"""
    log_dir = Path("checkpoints/logs")
    if not log_dir.exists():
        return None

    training_logs = list(log_dir.glob("training_log_*.jsonl"))
    if not training_logs:
        return None

    return max(training_logs, key=lambda p: p.stat().st_mtime)

def compute_stats(episodes, window=10):
    """Compute rolling statistics"""
    if len(episodes) < window:
        return None

    recent = episodes[-window:]
    returns = [ep['return'] for ep in recent]
    slots = [ep['avg_slot_distance'] for ep in recent]

    return {
        'mean_return': sum(returns) / len(returns),
        'mean_slot': sum(slots) / len(slots),
        'best_return': max(returns),
        'best_slot': min(slots)
    }

def main():
    log_path = get_latest_log()
    if not log_path:
        print("No training log found!")
        sys.exit(1)

    print(f"Monitoring: {log_path}")
    print("\n" + "="*90)
    print("TRAINING PROGRESS MONITOR")
    print("="*90)

    episodes = []
    last_size = 0

    try:
        while True:
            # Check if file has new data
            current_size = log_path.stat().st_size
            if current_size > last_size:
                with open(log_path, 'r') as f:
                    # Read all episodes
                    episodes = []
                    for line in f:
                        try:
                            episodes.append(json.loads(line))
                        except:
                            pass

                last_size = current_size

                # Print latest episode
                if episodes:
                    latest = episodes[-1]
                    ep = latest['episode']
                    ret = latest['return']
                    slot = latest['avg_slot_distance']
                    c_loss = latest['losses']['critic']
                    a_loss = latest['losses']['actor']
                    q1 = latest['q_values']['q1_mean']

                    alpha_vals = list(latest['alpha'].values())
                    alpha_avg = sum(alpha_vals) / len(alpha_vals) if alpha_vals else 0

                    warmup = latest.get('warmup', False)
                    marker = " [WARMUP]" if warmup else ""

                    print(f"\n[EP {ep:4d}] R={ret:8.1f} | Slot={slot:5.2f}m | "
                          f"C={c_loss:7.2f} A={a_loss:7.2f} | Q={q1:7.1f} | "
                          f"α={alpha_avg:.3f}{marker}")

                    # Show rolling statistics (post-warmup only)
                    if not warmup and len(episodes) >= 21:
                        training_episodes = [e for e in episodes if not e.get('warmup', False)]
                        if len(training_episodes) >= 10:
                            stats = compute_stats(training_episodes, window=10)
                            if stats:
                                print(f"  └─ Last 10: Avg Return={stats['mean_return']:.1f} | "
                                      f"Avg Slot={stats['mean_slot']:.2f}m | "
                                      f"Best Return={stats['best_return']:.1f} | "
                                      f"Best Slot={stats['best_slot']:.2f}m")

            time.sleep(5)  # Check every 5 seconds

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

        # Final summary
        if len(episodes) > 20:
            training_episodes = [e for e in episodes if not e.get('warmup', False)]
            print(f"\n{'='*90}")
            print("TRAINING SUMMARY")
            print(f"{'='*90}")
            print(f"Total Episodes: {len(episodes)}")
            print(f"Training Episodes: {len(training_episodes)}")

            if training_episodes:
                returns = [e['return'] for e in training_episodes]
                slots = [e['avg_slot_distance'] for e in training_episodes]

                print(f"\nBest Return: {max(returns):.1f} (Episode {[e['episode'] for e in training_episodes if e['return'] == max(returns)][0]})")
                print(f"Best Slot Distance: {min(slots):.2f}m (Episode {[e['episode'] for e in training_episodes if e['avg_slot_distance'] == min(slots)][0]})")
                print(f"Latest Return: {returns[-1]:.1f}")
                print(f"Latest Slot Distance: {slots[-1]:.2f}m")

                # Compute improvement
                if len(training_episodes) >= 20:
                    early = sum(returns[:10]) / 10
                    recent = sum(returns[-10:]) / 10
                    improvement = recent - early
                    print(f"\nImprovement (first 10 vs last 10 training episodes): {improvement:+.1f}")

if __name__ == '__main__':
    main()
