import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from env.map_env import MapEnv, get_state_tensor
from utils.map_utils import generate_map, generate_test_maps, validate_map_positions
from utils.logger import *

MAX_STEPS = 300
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 10
MAP_POOL_SIZE = 100
TEST_MAPS_COUNT = 50
CURRICULUM_STEPS = 9999 #暂时不升级难度

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def train_agent(agent, episodes, render_every, base_dir, dirs):
    save_config({
        "episodes": episodes,
        "batch_size": BATCH_SIZE,
        "target_update_freq": TARGET_UPDATE_FREQ,
        "map_pool_size": MAP_POOL_SIZE,
        "test_maps_count": TEST_MAPS_COUNT,
        "curriculum_steps": CURRICULUM_STEPS,
        "max_steps": MAX_STEPS,
        "gamma": agent.gamma,
        "lr": agent.optimizer.param_groups[0]['lr'],
        "epsilon_start": agent.epsilon,
        "epsilon_decay": agent.epsilon_decay,
        "epsilon_min": agent.epsilon_min,
        "device": str(agent.device)
    }, base_dir)

    test_maps = generate_test_maps(TEST_MAPS_COUNT)
    save_test_maps(test_maps, dirs["test_maps"])

    reward_list, test_rewards, success_rate = [], [], []
    map_pool = deque(maxlen=MAP_POOL_SIZE)
    success_buffer = deque(maxlen=1000)

    current_grid_size = 50
    current_obstacle_ratio = (0.08, 0.1)
    current_blind_ratio = 0.05

    cumulative_successes = 0
    cumulative_episodes = 0

    t0 = time.time()

    for episode in range(1, episodes + 1):
        if episode % CURRICULUM_STEPS == 0:
            current_grid_size = min(current_grid_size + 10, 100)
            current_obstacle_ratio = (current_obstacle_ratio[0], min(current_obstacle_ratio[1] + 0.01, 0.1))
            current_blind_ratio = min(current_blind_ratio + 0.005, 0.08)

        if len(map_pool) > 0 and random.random() < 0.8:
            obstacle_map, surveillance_map, start_pos, goal_pos = random.choice(map_pool)
        else:
            obstacle_map, surveillance_map, start_pos, goal_pos = generate_map(
                grid_size=current_grid_size,
                obstacle_ratio_range=current_obstacle_ratio,
                blind_ratio=current_blind_ratio
            )
            if not validate_map_positions(obstacle_map, start_pos, goal_pos):
                obstacle_map, surveillance_map, start_pos, goal_pos = generate_map(
                    grid_size=current_grid_size,
                    obstacle_ratio_range=current_obstacle_ratio,
                    blind_ratio=current_blind_ratio
                )
            map_pool.append((obstacle_map, surveillance_map, start_pos, goal_pos))
        
        loss_value = 0.0 

        env = MapEnv(obstacle_map, surveillance_map, start_pos, goal_pos)
        state_img, goal_vec = get_state_tensor(env.agent_pos, env.goal_pos, obstacle_map, surveillance_map)

        total_reward, done, steps = 0, False, 0
        episode_buffer = []

        while not done and steps < MAX_STEPS:
            action = agent.select_action(state_img, goal_vec)
            _, reward, done, _ = env.step(action)
            next_img, next_goal = get_state_tensor(env.agent_pos, env.goal_pos, obstacle_map, surveillance_map)

            transition = (state_img, goal_vec, action, reward, next_img, next_goal, done)
            agent.replay_buffer.push(transition)
            episode_buffer.append(transition)
            agent.update(BATCH_SIZE)

            state_img, goal_vec = next_img, next_goal
            total_reward += reward  # 或其他你能接受的最小总 reward
            steps += 1

        is_success = done and env.agent_pos == env.goal_pos
        if is_success:
            for t in episode_buffer:
                success_buffer.append(t)
            cumulative_successes += 1

        cumulative_episodes += 1

        if len(success_buffer) >= BATCH_SIZE and episode % 5 == 0:
            batch = random.sample(success_buffer, BATCH_SIZE)
            s_img, s_goal, a, r, ns_img, ns_goal, d = zip(*batch)
            s_img, s_goal = torch.tensor(np.array(s_img), dtype=torch.float32).to(agent.device), \
                            torch.tensor(np.array(s_goal), dtype=torch.float32).to(agent.device)
            a = torch.tensor(a, dtype=torch.int64).to(agent.device)
            r = torch.tensor(r, dtype=torch.float32).to(agent.device)
            ns_img = torch.tensor(np.array(ns_img), dtype=torch.float32).to(agent.device)
            ns_goal = torch.tensor(np.array(ns_goal), dtype=torch.float32).to(agent.device)
            d = torch.tensor(d, dtype=torch.float32).to(agent.device)

            q_values = agent.q_net(s_img, s_goal).gather(1, a.unsqueeze(1)).squeeze()
            with torch.no_grad():
                next_q = agent.target_net(ns_img, ns_goal).max(1)[0]
                target_q = r + agent.gamma * next_q * (1 - d)

            loss = torch.nn.functional.mse_loss(q_values, target_q)
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

            loss_value = loss.item()
        
        total_reward = max(total_reward, -200)

        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        reward_list.append(total_reward)

        # === 🆕 新增统计指标 ===
        path_len = len(env.path)
        blind_count = sum(1 for (x, y) in env.path if surveillance_map[x, y] == 0)
        blind_ratio = blind_count / path_len if path_len > 0 else 0
        shortest_path_len = manhattan_distance(start_pos, goal_pos)
        shortest_ratio = path_len / shortest_path_len if shortest_path_len > 0 else 0
        
        episode_data = {
            "total_reward": total_reward,
            "steps": steps,
            "epsilon": agent.epsilon,
            "success": 1 if is_success else 0,
            "grid_size": current_grid_size,
            "obstacle_ratio_max": current_obstacle_ratio[1],
            "blind_ratio": current_blind_ratio,
            "success_rate": cumulative_successes / cumulative_episodes,
            "path_len": path_len,
            "blind_step_ratio": blind_ratio,
            "shortest_path_ratio": shortest_ratio,
            "loss":loss_value
        }
        log_episode_data(episode, episode_data, dirs["logs"])

        if episode % 100 == 0:
            elapsed = time.time() - t0
            print(f"[Ep {episode}] Elapsed Time: {elapsed:.2f}s")

        if episode % 400 == 0 or episode == 1:
            save_trajectory(episode, env, dirs["trajectories"])
            save_map_data(episode, obstacle_map, surveillance_map, start_pos, goal_pos, dirs["trajectories"])
            save_checkpoint(agent, episode, dirs["models"])
            fig = env.render()
            fig.savefig(os.path.join(dirs["trajectories"], f"trajectory_{episode}.png"))
            plt.close(fig)

        if episode % 50 == 0:
            avg_reward, success_count = evaluate_on_test_maps(agent, test_maps)
            test_rewards.append(avg_reward)
            success_rate.append(success_count / len(test_maps))
            log_episode_data(episode, {
                "avg_test_reward": avg_reward,
                "success_rate": success_count / len(test_maps),
                "success_count": success_count
            }, dirs["logs"], filename="test_data.csv")
    # 保存每回合 reward 曲线（种子版本）
    np.savetxt(os.path.join(dirs["logs"], f"rewards_seed{agent.epsilon:.3f}.csv"),
            np.array(reward_list), delimiter=",")

    return reward_list, test_rewards, success_rate


def run_test_episode(agent, env):
    state_img, goal_vec = get_state_tensor(env.agent_pos, env.goal_pos, env.obstacle_map, env.surveillance_map)
    agent.epsilon = 0
    steps = 0
    done = False
    while not done and steps < MAX_STEPS:
        action = agent.select_action(state_img, goal_vec)
        _, _, done, _ = env.step(action)
        state_img, goal_vec = get_state_tensor(env.agent_pos, env.goal_pos, env.obstacle_map, env.surveillance_map)
        steps += 1
    return env.path


def evaluate_on_test_maps(agent, test_maps):
    total_reward = 0
    success_count = 0
    for map_data in test_maps:
        obstacle_map, surveillance_map, start_pos, goal_pos = map_data
        env = MapEnv(obstacle_map, surveillance_map, start_pos, goal_pos)
        state_img, goal_vec = get_state_tensor(env.agent_pos, env.goal_pos, obstacle_map, surveillance_map)
        steps = 0
        done = False
        agent.epsilon = 0
        episode_reward = 0
        while not done and steps < MAX_STEPS:
            action = agent.select_action(state_img, goal_vec)
            _, reward, done, _ = env.step(action)
            state_img, goal_vec = get_state_tensor(env.agent_pos, env.goal_pos, obstacle_map, surveillance_map)
            episode_reward += reward
            steps += 1
        total_reward += episode_reward
        if done and env.agent_pos == env.goal_pos:
            success_count += 1
    return total_reward / len(test_maps), success_count

def plot_and_save_training_curves(log_dir):
    import pandas as pd
    import matplotlib.pyplot as plt

    log_file = os.path.join(log_dir, "episode_data.csv")
    df = pd.read_csv(log_file)

    def smooth(y, window=50):
        if len(y) < window:
            return y
        return np.convolve(y, np.ones(window)/window, mode='valid')

    x = df['episode']
    save_path = os.path.join(log_dir, "..", "plots")

    # ✅ Reward 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(x, df['total_reward'], alpha=0.3, label='Raw')
    plt.plot(x[49:], smooth(df['total_reward']), label='Smoothed')
    plt.title("Training Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "reward_curve.png"))
    plt.close()

    # ✅ 成功率曲线
    plt.figure(figsize=(8, 5))
    plt.plot(x, df['success'].rolling(50).mean()*100)
    plt.title("Success Rate (Moving Avg)")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "success_rate_curve.png"))
    plt.close()

    # ✅ 最短路径比率
    if 'shortest_path_ratio' in df:
        y = df['shortest_path_ratio']
        y_smooth = smooth(y)
        x_smooth = x[len(x) - len(y_smooth):]
        plt.figure(figsize=(8, 5))
        plt.plot(x_smooth, y_smooth)
        plt.title("Shortest Path Ratio")
        plt.xlabel("Episode")
        plt.ylabel("Actual / Shortest Path")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "shortest_ratio_curve.png"))
        plt.close()

    # ✅ 盲区步数占比
    if 'blind_step_ratio' in df:
        y = df['blind_step_ratio']
        y_smooth = smooth(y)
        x_smooth = x[len(x) - len(y_smooth):]
        plt.figure(figsize=(8, 5))
        plt.plot(x_smooth, y_smooth)
        plt.title("Blind Zone Step Ratio")
        plt.xlabel("Episode")
        plt.ylabel("Blind / Total Steps")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "blind_ratio_curve.png"))
        plt.close()

    # ✅ 平均路径长度
    if 'path_len' in df:
        y = df['path_len']
        y_smooth = smooth(y)
        x_smooth = x[len(x) - len(y_smooth):]
        plt.figure(figsize=(8, 5))
        plt.plot(x_smooth, y_smooth)
        plt.title("Path Length (Smoothed)")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "path_len_curve.png"))
        plt.close()

