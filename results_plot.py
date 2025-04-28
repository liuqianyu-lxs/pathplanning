import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 路径设置
result_root = "./"  # 根目录
model_types = ["A1", "A2", "B1", "B2"]
seeds = [0, 100, 499, 999, 5000]
metrics = ["total_reward", "success", "shortest_path_ratio", "blind_step_ratio", "path_len", "loss"]

output_dir = "aggregated_metrics"
os.makedirs(output_dir, exist_ok=True)

for metric_name in metrics:
    all_model_data = {}

    for model in model_types:
        metric_values = []

        for seed in seeds:
            log_path = os.path.join(result_root, f"results_{model}_seed{seed}/logs/episode_data.csv")
            if not os.path.exists(log_path):
                print(f"\u26a0\ufe0f Missing: {log_path}")
                continue

            df = pd.read_csv(log_path)
            if metric_name not in df.columns:
                print(f"\u26a0\ufe0f Metric '{metric_name}' not found in {log_path}")
                continue

            metric_values.append(df[metric_name].values)

        if metric_values:
            metric_values = np.array(metric_values)
            avg = np.mean(metric_values, axis=0)
            std = np.std(metric_values, axis=0)
            all_model_data[f"{model}-mean"] = avg
            all_model_data[f"{model}-std"] = std

    # 生成合并表
    if all_model_data:
        df_out = pd.DataFrame(all_model_data)
        save_path = os.path.join(output_dir, f"{metric_name}.csv")
        df_out.to_csv(save_path, index=False)
        print(f"✅ Saved: {save_path}")
    else:
        print(f"\u26a0\ufe0f No data available for metric: {metric_name}")

# 路径设置
aggregated_dir = "aggregated_metrics"
plot_output_dir = "aggregated_plots"
os.makedirs(plot_output_dir, exist_ok=True)

# 要绘制的指标
metrics = ["total_reward", "success", "shortest_path_ratio", "blind_step_ratio", "path_len", "loss"]
model_types = ["A1", "A2", "B1", "B2"]

for metric_name in metrics:
    csv_path = os.path.join(aggregated_dir, f"{metric_name}.csv")
    if not os.path.exists(csv_path):
        print(f"\u26a0\ufe0f Missing aggregated file for: {metric_name}")
        continue

    df = pd.read_csv(csv_path)
    episodes = range(len(df))  # 根据行数确定episode

    plt.figure(figsize=(10, 6))
    for model in model_types:
        if f"{model}-mean" not in df.columns or f"{model}-std" not in df.columns:
            print(f"\u26a0\ufe0f Missing columns for model: {model} in {metric_name}")
            continue

        mean_curve = df[f"{model}-mean"].values
        std_curve = df[f"{model}-std"].values

        plt.plot(episodes, mean_curve, label=f"{model}", linewidth=2)
        plt.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    plt.title(f"{metric_name.replace('_', ' ').title()} Comparison")
    plt.xlabel("Episode")
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(plot_output_dir, f"{metric_name}_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved plot: {save_path}")

