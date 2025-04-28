import os
import json
import csv
import numpy as np
import torch
from datetime import datetime

def setup_logging_directories(model_type=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_type:
        base_dir = f"results_{model_type}"
    else:
        base_dir = f"results_{timestamp}"

    dirs = {
        "models": os.path.join(base_dir, "models"),
        "plots": os.path.join(base_dir, "plots"),
        "logs": os.path.join(base_dir, "logs"),
        "trajectories": os.path.join(base_dir, "trajectories"),
        "test_maps": os.path.join(base_dir, "test_maps")
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return base_dir, dirs


def save_config(config_dict, base_dir):
    config_dict["model_type"] = config_dict.get("model_type", "unknown")  # 添加模型类型
    config_path = os.path.join(base_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)


def log_episode_data(episode, data_dict, log_dir, filename="episode_data.csv"):
    filepath = os.path.join(log_dir, filename)
    file_exists = os.path.isfile(filepath)

    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['episode'] + list(data_dict.keys()))
        writer.writerow([episode] + list(data_dict.values()))


def save_trajectory(episode, env, traj_dir, filename_prefix="trajectory"):
    filepath = os.path.join(traj_dir, f"{filename_prefix}_{episode}.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        for pos in env.path:
            writer.writerow(pos)


def save_map_data(episode, obstacle_map, surveillance_map, start_pos, goal_pos, traj_dir):
    filepath = os.path.join(traj_dir, f"map_{episode}.npz")
    np.savez(filepath,
             obstacle_map=obstacle_map,
             surveillance_map=surveillance_map,
             start_pos=np.array(start_pos),
             goal_pos=np.array(goal_pos))


def save_checkpoint(agent, episode, model_dir):
    filepath = os.path.join(model_dir, f"model_ep{episode}.pth")
    torch.save({
        'episode': episode,
        'model_state_dict': agent.q_net.state_dict(),
        'target_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon
    }, filepath)


def save_test_maps(test_maps, test_map_dir):
    os.makedirs(test_map_dir, exist_ok=True)
    for i, (obstacle_map, surveillance_map, start_pos, goal_pos) in enumerate(test_maps):
        filepath = os.path.join(test_map_dir, f"test_map_{i}.npz")
        np.savez(filepath,
                 obstacle_map=obstacle_map,
                 surveillance_map=surveillance_map,
                 start_pos=np.array(start_pos),
                 goal_pos=np.array(goal_pos))

    with open(os.path.join(test_map_dir, "index.txt"), 'w') as f:
        f.write(f"Total test maps: {len(test_maps)}\n")
        for i in range(len(test_maps)):
            _, _, start_pos, goal_pos = test_maps[i]
            f.write(f"Map {i}: Start={start_pos}, Goal={goal_pos}\n")
