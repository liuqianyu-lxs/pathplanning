import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


    
def plot_multi_seed_metric(result_root, model_code, metric_name, output_path):
    seeds = [0, 100, 499, 999, 5000]
    all_curves = []

    for seed in seeds:
        log_path = os.path.join(result_root, f"results_{model_code}_seed{seed}/logs/episode_data.csv")
        if not os.path.exists(log_path):
            print(f"❌ Missing file: {log_path}")
            continue

        df = pd.read_csv(log_path)
        if metric_name not in df.columns:
            print(f"⚠️ Metric '{metric_name}' not found in seed {seed}")
            continue

        values = df[metric_name].values
        all_curves.append(values)

    if len(all_curves) < 2:
        print("⚠️ Not enough valid runs to plot.")
        return

    all_curves = np.array(all_curves)
    avg_curve = np.mean(all_curves, axis=0)
    std_curve = np.std(all_curves, axis=0)
    episodes = np.arange(len(avg_curve))

    # 绘图
    plt.figure(figsize=(10, 6))
    for curve in all_curves:
        plt.plot(episodes, curve, alpha=0.3, color='gray')
    plt.plot(episodes, avg_curve, label='Average', color='blue', linewidth=2)
    #plt.fill_between(episodes, avg_curve - std_curve, avg_curve + std_curve, color='blue', alpha=0.2)

    plt.title(f"{metric_name.replace('_', ' ').title()} - {model_code}")
    plt.xlabel("Episode")
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{model_code}_{metric_name}.png"))
    plt.close()