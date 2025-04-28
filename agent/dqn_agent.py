import random
import torch
import torch.nn.functional as F
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_img, goal_vec, action, reward, next_img, next_goal, done = zip(*batch)
        return (
            torch.tensor(np.array(state_img), dtype=torch.float32),
            torch.tensor(np.array(goal_vec), dtype=torch.float32),
            torch.tensor(action, dtype=torch.int64),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(np.array(next_img), dtype=torch.float32),
            torch.tensor(np.array(next_goal), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, q_net, device="cpu", gamma=0.99, lr=5e-4, epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.05):
        self.device = device
        self.q_net = q_net.to(device)
        self.target_net = type(q_net)().to(device)  # 同样结构
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state_img, goal_vec):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        state_img = torch.tensor(state_img, dtype=torch.float32).unsqueeze(0).to(self.device)
        goal_vec = torch.tensor(goal_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_img, goal_vec)
            return q_values.argmax().item()

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        s_img, s_goal, a, r, ns_img, ns_goal, d = self.replay_buffer.sample(batch_size)
        s_img, s_goal = s_img.to(self.device), s_goal.to(self.device)
        ns_img, ns_goal = ns_img.to(self.device), ns_goal.to(self.device)
        a, r, d = a.to(self.device), r.to(self.device), d.to(self.device)

        q_values = self.q_net(s_img, s_goal).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self.target_net(ns_img, ns_goal).max(1)[0]
            target_q = r + self.gamma * next_q * (1 - d)

        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
