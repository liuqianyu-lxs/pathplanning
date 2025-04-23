import argparse
import torch
import os
import time

from models.qnet_cnn import QNetCNN
from models.qnet_mlp import QNetMLP
from agent.dqn_agent import DQNAgent
from agent.ddqn_agent import DDQNAgent
from utils.logger import setup_logging_directories
from core.train_utils import train_agent
from core.train_utils import plot_and_save_training_curves

def get_agent_and_model(model_type, device):
    if model_type in ['A1', 'B1']:
        q_net = QNetMLP()
    else:
        q_net = QNetCNN()

    if model_type in ['A1', 'A2']:
        agent = DQNAgent(q_net=q_net, device=device)
    else:
        agent = DDQNAgent(q_net=q_net, device=device)

    return agent, q_net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['A1', 'A2', 'B1', 'B2'])
    parser.add_argument('--episodes', type=int, default=5000)
    args = parser.parse_args()

    model_type = args.model_type
    episodes = args.episodes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 Starting training with model type: {model_type}")
    print(f"🖥️  Using device: {device}")

    base_dir, dirs = setup_logging_directories(model_type=model_type)
    agent, q_net = get_agent_and_model(model_type, device)


    from utils.logger import save_config
    save_config({
        "model_type": model_type,
        "episodes": episodes,
        "device": str(device)
    }, base_dir)

    start_time = time.time()
    rewards, test_rewards, success_rates = train_agent(
        agent=agent,
        episodes=episodes,
        render_every=500,
        base_dir=base_dir,
        dirs=dirs
    )
    elapsed_time = time.time() - start_time
    h, rem = divmod(elapsed_time, 3600)
    m, s = divmod(rem, 60)

    # 保存 summary 信息
    with open(os.path.join(base_dir, "summary.txt"), 'w') as f:
        f.write(f"Model: {model_type}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Episodes: {episodes}\n")
        f.write(f"Final test success rate: {success_rates[-1]*100:.2f}%\n")
        f.write(f"Final test avg reward: {test_rewards[-1]:.2f}\n")
        f.write(f"Max train reward: {max(rewards):.2f} @ Episode {rewards.index(max(rewards))+1}\n")
        f.write(f"Total training time: {int(h)}h {int(m)}m {int(s)}s\n")
   
    plot_and_save_training_curves(dirs["logs"])
    print(f"Training completed. Results saved to: {base_dir}")
    

if __name__ == "__main__":
    main()
