import subprocess
import time

model_types = ["A1", "A2", "B1", "B2"]
episodes = 5000  

for model in model_types:
    print(f"\n=== 🚀 正在训练模型 {model} ===")
    start_time = time.time()

    subprocess.run(["python", "main.py", "--model_type", model, "--episodes", str(episodes)])

    duration = time.time() - start_time
    m, s = divmod(duration, 60)
    print(f"模型 {model} 训练完成，用时 {int(m)} 分 {int(s)} 秒")
