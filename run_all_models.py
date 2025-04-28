import subprocess
import time
from plot_multi_seed_curves import plot_multi_seed_metric
from plot_all_model_comparison import plot_all_models_avg_reward
import os

model_types = ["A1", "A2", "B1", "B2"]
seeds = [0, 100, 499, 999, 5000]
episodes = 3000  
result_root = "./"
plot_output_path = "./plots"
os.makedirs(plot_output_path, exist_ok=True)

for model in model_types:
    print(f"\n=== 🚀 正在训练模型 {model} ===")
    start_time = time.time()
    for seed in seeds:
        print(f"🎯 Training {model} with seed {seed}")
        subprocess.run(["python", "main.py", "--model_type", model, "--episodes", str(episodes), "--seed", str(seed)])


    for metric in ["total_reward","success", "shortest_path_ratio", "blind_step_ratio", "path_len", "loss"]:
        plot_multi_seed_metric(result_root, model, metric, plot_output_path)

    duration = time.time() - start_time
    m, s = divmod(duration, 60)
    print(f"模型 {model} 训练完成，用时 {int(m)} 分 {int(s)} 秒")

plot_all_models_avg_reward(
    result_root="./",
    model_codes=["A1", "A2", "B1", "B2"],
    output_path="./plots"
)