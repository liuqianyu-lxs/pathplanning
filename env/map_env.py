import numpy as np
import matplotlib.pyplot as plt

class MapEnv:
    def __init__(self, obstacle_map, surveillance_map, start_pos, goal_pos):
        self.obstacle_map = obstacle_map
        self.surveillance_map = surveillance_map
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.grid_size = obstacle_map.shape[0]

        self.agent_pos = start_pos
        self.path = [start_pos]

    def reset(self):
        self.agent_pos = self.start_pos
        self.path = [self.start_pos]
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos
        next_x, next_y = x, y

        if action == 0:
            next_x -= 1
        elif action == 1:
            next_x += 1
        elif action == 2:
            next_y -= 1
        elif action == 3:
            next_y += 1

        if not (0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size):
            return self.agent_pos, -100, True, {"reason": "Out of bounds"}

        if self.obstacle_map[next_x, next_y] == 1:
            return self.agent_pos, -100, True, {"reason": "Hit obstacle"}

        reward = -0.05

        if self.surveillance_map[next_x, next_y] == 0:
            reward += -20

        old_dist = abs(x - self.goal_pos[0]) + abs(y - self.goal_pos[1])
        new_dist = abs(next_x - self.goal_pos[0]) + abs(next_y - self.goal_pos[1])
        reward += 5 * (old_dist - new_dist)

        self.agent_pos = (next_x, next_y)
        self.path.append(self.agent_pos)

        done = self.agent_pos == self.goal_pos
        if done:
            reward += 500
        
        reward = max(reward, -200)

        return self.agent_pos, reward, done, {}

    def render(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.surveillance_map, cmap="gray", origin="upper")

        obstacle_x, obstacle_y = np.where(self.obstacle_map == 1)
        ax.scatter(obstacle_y, obstacle_x, color="red", s=1, label="Obstacle")

        if len(self.path) > 1:
            px, py = zip(*self.path)
            ax.plot(py, px, color='orange', linewidth=2, label="Path")

        ax.scatter(self.start_pos[1], self.start_pos[0], color="lime", s=60, label="Start")
        ax.scatter(self.goal_pos[1], self.goal_pos[0], color="blue", s=60, label="Goal")

        ax.set_title("Agent Navigation")
        ax.axis("off")
        ax.legend()
        return fig


def extract_local_window(map_2d, center, window_size=5, pad_value=1):
    half = window_size // 2
    x, y = center
    padded_map = np.pad(map_2d, ((half, half), (half, half)), 'constant', constant_values=pad_value)
    window = padded_map[x:x + window_size, y:y + window_size]
    return window


def get_state_tensor(agent_pos, goal_pos, obstacle_map, surveillance_map, window_size=5):
    x, y = agent_pos
    gx, gy = goal_pos
    grid_size = obstacle_map.shape[0]
    dx = (gx - x) / grid_size
    dy = (gy - y) / grid_size

    obstacle_window = extract_local_window(obstacle_map, agent_pos, window_size, pad_value=1)
    surveillance_window = extract_local_window(surveillance_map, agent_pos, window_size, pad_value=1)

    state_image = np.stack([obstacle_window, surveillance_window]).astype(np.float32)
    goal_vector = np.array([dx, dy], dtype=np.float32)

    return state_image, goal_vector
