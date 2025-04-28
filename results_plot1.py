import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# 路径设置
aggregated_dir = "aggregated_metrics"
plot_output_dir = "aggregated_plots_smooth_styled"
os.makedirs(plot_output_dir, exist_ok=True)

# 要绘制的指标
metrics = ["total_reward", "success", "shortest_path_ratio", "blind_step_ratio", "path_len", "loss"]
model_types = ["A1", "A2", "B1", "B2"]

# 滑动平均窗口大小
smooth_window = 20

# 模型名称映射
model_name_map = {
    "A1": "DQN-MLP",
    "A2": "DQN-CNN",
    "B1": "DDQN-MLP",
    "B2": "DDQN-CNN"
}

# 线条颜色
model_colors = {
    "A1": "#1f77b4",   # 蓝色
    "A2": "#ff7f0e",   # 橙色
    "B1": "#2ca02c",   # 绿色
    "B2": "#d62728"    # 红色
}

# 全局样式设置
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['lines.linewidth'] = 2.5

# 定义滑动平均函数
def moving_average(data, window_size):
    return pd.Series(data).rolling(window=window_size, min_periods=1, center=True).mean()

for metric_name in metrics:
    csv_path = os.path.join(aggregated_dir, f"{metric_name}.csv")
    if not os.path.exists(csv_path):
        print(f"\u26a0\ufe0f Missing aggregated file for: {metric_name}")
        continue

    df = pd.read_csv(csv_path)
    episodes = range(len(df))

    plt.figure(figsize=(10, 6), dpi=300)

    for model in model_types:
        mean_col = f"{model}-mean"
        if mean_col not in df.columns:
            print(f"\u26a0\ufe0f Missing {mean_col} in {csv_path}")
            continue

        smoothed_curve = moving_average(df[mean_col].values, smooth_window)
        plt.plot(episodes, smoothed_curve, label=model_name_map[model], color=model_colors[model])

    plt.title(f"{metric_name.replace('_', ' ').title()} Comparison (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.grid(True)
    plt.legend(frameon=False, loc='upper left')
    plt.tight_layout()

    save_path = os.path.join(plot_output_dir, f"{metric_name}_smooth_styled.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved styled plot: {save_path}")
