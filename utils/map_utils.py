import numpy as np
import random

def generate_map(grid_size=100, obstacle_ratio_range=(0.05, 0.1), blind_ratio=0.05, blob_radius=3):
    obstacle_map = np.zeros((grid_size, grid_size), dtype=np.uint8)
    surveillance_map = np.ones((grid_size, grid_size), dtype=np.uint8)

    total_obstacles_needed = int(grid_size * grid_size * np.random.uniform(*obstacle_ratio_range))
    obstacles_created = 0
    while obstacles_created < total_obstacles_needed:
        center_x = random.randint(0, grid_size - 1)
        center_y = random.randint(0, grid_size - 1)
        for dx in range(-blob_radius, blob_radius + 1):
            for dy in range(-blob_radius, blob_radius + 1):
                x = center_x + dx
                y = center_y + dy
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    if np.random.rand() < 0.8 and obstacle_map[x, y] == 0:
                        obstacle_map[x, y] = 1
                        obstacles_created += 1
                        if obstacles_created >= total_obstacles_needed:
                            break
            if obstacles_created >= total_obstacles_needed:
                break

    num_blind = int(grid_size * grid_size * blind_ratio)
    blind_indices = random.sample(range(grid_size * grid_size), num_blind)
    for idx in blind_indices:
        x, y = divmod(idx, grid_size)
        surveillance_map[x, y] = 0

    def get_random_free_position():
        while True:
            x = random.randint(0, grid_size - 1)
            y = random.randint(0, grid_size - 1)
            if obstacle_map[x, y] == 0:
                return (x, y)

    start_pos = get_random_free_position()
    goal_pos = get_random_free_position()

    return obstacle_map, surveillance_map, start_pos, goal_pos


def validate_map_positions(obstacle_map, start_pos, goal_pos):
    return (obstacle_map[start_pos[0], start_pos[1]] == 0 and
            obstacle_map[goal_pos[0], goal_pos[1]] == 0)


def generate_valid_map_until_success(grid_size, obstacle_ratio_range, blind_ratio):
    while True:
        obstacle_map, surveillance_map, start_pos, goal_pos = generate_map(
            grid_size=grid_size,
            obstacle_ratio_range=obstacle_ratio_range,
            blind_ratio=blind_ratio
        )
        if validate_map_positions(obstacle_map, start_pos, goal_pos):
            return obstacle_map, surveillance_map, start_pos, goal_pos


def generate_test_maps(count=10, difficulties=None):
    if difficulties is None:
        difficulties = [(100, (0.05, 0.1), 0.05) for _ in range(count)]

    test_maps = []
    for i in range(count):
        grid_size, obstacle_ratio, blind_ratio = difficulties[i]
        map_data = generate_valid_map_until_success(grid_size, obstacle_ratio, blind_ratio)
        test_maps.append(map_data)
    return test_maps
