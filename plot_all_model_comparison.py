import os
import numpy as np
import matplotlib.pyplot as plt

def load_avg_reward_curve(result_root, model_code, seeds = [0, 100, 499, 999, 5000]):
    all_rewards = []

    for seed in seeds:
        path = os.path.join(result_root, f"results_{model_code}_seed{seed}/logs/episode_data.csv")
        if not os.path.exists(path):
            print(f"⚠️ Missing: {path}")
            continue
        df = pd.read_csv(path)
        if 'total_reward' not in df.columns:
            print(f"⚠️ 'total_reward' not found in {path}")
            continue

        rewards = df['total_reward'].values
        all_rewards.append(rewards)

    if len(all_rewards) == 0:
        return None, None

    all_rewards = np.array(all_rewards)
    avg_reward = np.mean(all_rewards, axis=0)
    return avg_reward, np.arange(len(avg_reward))


def plot_all_models_avg_reward(result_root, model_codes, output_path):
    plt.figure(figsize=(10, 6))

    for code in model_codes:
        avg_reward, x = load_avg_reward_curve(result_root, code)
        if avg_reward is not None:
            plt.plot(x, avg_reward, label=code, linewidth=2)

    plt.title("Average Reward Comparison (5 Seeds)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, "all_models_reward_compare.png"))
    plt.close()
    print(f"✅ Comparison plot saved to {os.path.join(output_path, 'all_models_reward_compare.png')}")
