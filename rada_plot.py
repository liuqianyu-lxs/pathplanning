import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# 路径设置
aggregated_dir = "aggregated_metrics"
data_output_dir = "radar_data"
radar_output_dir = "radar_charts"
os.makedirs(data_output_dir, exist_ok=True)
os.makedirs(radar_output_dir, exist_ok=True)

# 模型和指标
model_types = ["A1", "A2", "B1", "B2"]
model_name_map = {
    "A1": "DQN-MLP",
    "A2": "DQN-CNN",
    "B1": "DDQN-MLP",
    "B2": "DDQN-CNN"
}
metrics = ["total_reward", "success", "shortest_path_ratio", "blind_step_ratio"]

# 指标方向（True=越大越好，False=越小越好）
metric_positive = {
    "total_reward": True,
    "success": True,
    "shortest_path_ratio": True,
    "blind_step_ratio": False
}

# 取最后多少episode
last_n = 1500

# 1. 手动抽取最后1000集数据，计算均值，保存新csv
radar_data = {}

for metric in metrics:
    csv_path = os.path.join(aggregated_dir, f"{metric}.csv")
    df = pd.read_csv(csv_path)
    df_last = df.tail(last_n)

    for model in model_types:
        mean_col = f"{model}-mean"
        if mean_col in df.columns:
            avg_value = df_last[mean_col].mean()
            radar_data.setdefault(model, {})[metric] = avg_value

# 将数据往外写成新表
radar_df = pd.DataFrame(radar_data).T  # models x metrics
radar_df.index.name = "Model"
radar_save_path = os.path.join(data_output_dir, "radar_data_summary.csv")
radar_df.to_csv(radar_save_path)
print(f"✅ Saved summarized radar data to: {radar_save_path}")

# 2. 从新csv里读数据，并作图
radar_df = pd.read_csv(radar_save_path, index_col=0)

# 转换为0-1规一化，考虑指标方向
for metric in metrics:
    if metric_positive[metric]:
        min_val = radar_df[metric].min()
        max_val = radar_df[metric].max()
        radar_df[metric] = (radar_df[metric] - min_val) / (max_val - min_val)
    else:
        min_val = radar_df[metric].min()
        max_val = radar_df[metric].max()
        radar_df[metric] = (max_val - radar_df[metric]) / (max_val - min_val)

# 开始作图
categories = metrics
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# 绘制全模型对比
fig, ax = plt.subplots(figsize=(8, 8), dpi=300, subplot_kw=dict(polar=True))

for idx, model in enumerate(model_types):
    values = radar_df.loc[model].tolist()
    plot_values = values + values[:1]
    ax.plot(angles, plot_values, linewidth=2, linestyle='solid', label=model_name_map[model], color=colors[idx])
    ax.fill(angles, plot_values, alpha=0.1, color=colors[idx])

plt.xticks([n / float(N) * 2 * pi for n in range(N)], [m.replace("_", " ").title() for m in categories], fontsize=12)
ax.set_rlabel_position(30)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
plt.ylim(0, 1)
plt.title("Model Performance Comparison", size=16, y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(radar_output_dir, "all_models_radar.png"))
plt.close()
print(f"✅ Saved: all_models_radar.png")

# 单模型雷达图
for idx, model in enumerate(model_types):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300, subplot_kw=dict(polar=True))
    values = radar_df.loc[model].tolist()
    plot_values = values + values[:1]

    ax.plot(angles, plot_values, linewidth=2, linestyle='solid', label=model_name_map[model], color=colors[idx])
    ax.fill(angles, plot_values, alpha=0.1, color=colors[idx])

    plt.xticks([n / float(N) * 2 * pi for n in range(N)], [m.replace("_", " ").title() for m in categories], fontsize=12)
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1)

    plt.title(f"Performance of {model_name_map[model]}", size=16, y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(radar_output_dir, f"{model}_radar.png"))
    plt.close()
    print(f"✅ Saved: {model}_radar.png")
