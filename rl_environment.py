"""
RL Environment for Road Pathfinding (v2)
=========================================
Q-Learning agent navigates a DOWNSAMPLED cost grid cell-by-cell
from start to goal, learning to avoid obstacles and congestion zones.

Key improvements over v1:
- Grid downsampling for tractable Q-learning on large maps
- Much more training episodes with better reward shaping
- Proper cell-by-cell path recording (no straight-line shortcuts)
- Path upsampling back to original grid resolution

Part of the Hybrid RL + GA road generation pipeline.
"""

import numpy as np
import random
import math
import time


# 8 movement directions: N, S, E, W, NE, SE, NW, SW
ACTIONS = [
    (0, 1),   # N
    (0, -1),  # S
    (1, 0),   # E
    (-1, 0),  # W
    (1, 1),   # NE
    (1, -1),  # SE
    (-1, 1),  # NW
    (-1, -1), # SW
]


def downsample_grid(grid, factor):
    """
    Downsample a cost grid by the given factor.
    Each cell in the downsampled grid takes the MAX cost of the block it covers.
    This ensures obstacles are preserved.
    """
    orig_w, orig_h = grid.shape
    new_w = max(1, orig_w // factor)
    new_h = max(1, orig_h // factor)
    downsampled = np.ones((new_w, new_h), dtype=np.int32)

    for x in range(new_w):
        for y in range(new_h):
            x0 = x * factor
            y0 = y * factor
            x1 = min(x0 + factor, orig_w)
            y1 = min(y0 + factor, orig_h)
            block = grid[x0:x1, y0:y1]
            # Use max to preserve obstacles and high-cost zones
            downsampled[x][y] = int(np.max(block))

    return downsampled


def upsample_path(path, factor):
    """Convert a path on the downsampled grid back to original grid coordinates."""
    return [(x * factor + factor // 2, y * factor + factor // 2) for x, y in path]


class RoadGridEnvironment:
    """
    Grid-based RL environment for road bypass pathfinding.

    The agent moves cell-by-cell on a cost grid, receiving rewards/penalties
    based on the terrain it traverses. This produces proper grid-aligned paths.
    """

    def __init__(self, grid, start, goal, obstacle_threshold=9999):
        self.grid = grid
        self.grid_w, self.grid_h = grid.shape
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.obstacle_threshold = obstacle_threshold
        self.state = self.start
        self.steps_taken = 0
        self.visited = set()
        # Dynamic max steps based on grid size
        self.max_steps = int(math.sqrt(self.grid_w**2 + self.grid_h**2) * 4)

    def reset(self):
        """Reset environment to start state."""
        self.state = self.start
        self.steps_taken = 0
        self.visited = set()
        self.visited.add(self.start)
        return self.state

    def _distance_to_goal(self, pos):
        return math.sqrt((pos[0] - self.goal[0])**2 + (pos[1] - self.goal[1])**2)

    def step(self, action_idx):
        """
        Take one step. Returns (next_state, reward, done, info).
        """
        dx, dy = ACTIONS[action_idx]
        nx, ny = self.state[0] + dx, self.state[1] + dy
        self.steps_taken += 1

        # Out of bounds — stay in place with penalty
        if nx < 0 or nx >= self.grid_w or ny < 0 or ny >= self.grid_h:
            return self.state, -50.0, False, {"reason": "out_of_bounds"}

        cell_cost = self.grid[nx][ny]

        # Hit obstacle — episode over
        if cell_cost >= self.obstacle_threshold:
            return self.state, -500.0, True, {"reason": "obstacle"}

        # Revisiting penalty
        revisit_penalty = -20.0 if (nx, ny) in self.visited else 0.0

        # Distance shaping
        old_dist = self._distance_to_goal(self.state)
        new_dist = self._distance_to_goal((nx, ny))
        shaping = (old_dist - new_dist) * 5.0

        # Step cost (proportional to cell cost)
        move_mult = 1.414 if (dx != 0 and dy != 0) else 1.0
        step_cost = -(move_mult * cell_cost) / 5.0

        # Move
        self.state = (nx, ny)
        self.visited.add((nx, ny))

        # Goal reached?
        if abs(nx - self.goal[0]) <= 1 and abs(ny - self.goal[1]) <= 1:
            return self.state, 1000.0, True, {"reason": "goal_reached"}

        # Max steps?
        if self.steps_taken >= self.max_steps:
            # Partial reward based on how close we got
            closeness = 1.0 - (new_dist / (self._distance_to_goal(self.start) + 1e-6))
            return self.state, -200.0 + closeness * 100.0, True, {"reason": "max_steps"}

        reward = step_cost + shaping + revisit_penalty
        return self.state, reward, False, {}


class QLearningAgent:
    """
    Tabular Q-Learning agent with ε-greedy policy.
    """

    def __init__(self, n_actions=8, alpha=0.15, gamma=0.9,
                 epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.997):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def _get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self._get_q(state)))

    def choose_greedy(self, state):
        return int(np.argmax(self._get_q(state)))

    def update(self, state, action, reward, next_state, done):
        q = self._get_q(state)
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self._get_q(next_state))
        q[action] += self.alpha * (target - q[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train_rl_agent(grid, start, goal, obstacle_threshold=9999,
                   n_episodes=1500, downsample_factor=3):
    """
    Train RL agent on a downsampled cost grid and return discovered paths.

    The grid is downsampled so Q-learning can converge in reasonable time.
    Discovered paths are upsampled back to the original grid resolution.

    Args:
        grid: 2D numpy cost grid (original resolution).
        start: (x, y) start on original grid.
        goal: (x, y) goal on original grid.
        obstacle_threshold: Cost threshold for impassable cells.
        n_episodes: Number of training episodes.
        downsample_factor: Factor to reduce grid size by.

    Returns:
        List of discovered paths (each path is a list of (x,y) tuples
        on the ORIGINAL grid). Sorted by cost (best first).
    """
    start_time = time.time()

    # --- Downsample grid for tractable Q-learning ---
    if downsample_factor > 1:
        coarse_grid = downsample_grid(grid, downsample_factor)
        coarse_start = (start[0] // downsample_factor, start[1] // downsample_factor)
        coarse_goal = (goal[0] // downsample_factor, goal[1] // downsample_factor)
    else:
        coarse_grid = grid
        coarse_start = start
        coarse_goal = goal

    cw, ch = coarse_grid.shape
    # Clamp to bounds
    coarse_start = (max(0, min(cw-1, coarse_start[0])),
                    max(0, min(ch-1, coarse_start[1])))
    coarse_goal = (max(0, min(cw-1, coarse_goal[0])),
                   max(0, min(ch-1, coarse_goal[1])))

    print(f"      RL Training: {n_episodes} episodes on coarse grid "
          f"{cw}x{ch} (downsampled {downsample_factor}x from {grid.shape})")

    env = RoadGridEnvironment(coarse_grid, coarse_start, coarse_goal, obstacle_threshold)
    agent = QLearningAgent(
        n_actions=8,
        alpha=0.15,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.997
    )

    successful_paths = []
    best_reward = float('-inf')
    goals_reached = 0

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        path = [state]
        info = {}

        for _ in range(env.max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            path.append(state)
            if done:
                break

        agent.decay_epsilon()

        if info.get("reason") == "goal_reached":
            goals_reached += 1
            cost = _path_cost(coarse_grid, path)
            successful_paths.append((cost, path))
            if total_reward > best_reward:
                best_reward = total_reward

        # Progress logging every 200 episodes
        if (episode + 1) % 200 == 0:
            elapsed = time.time() - start_time
            print(f"        Episode {episode+1:>5d}/{n_episodes} | "
                  f"eps={agent.epsilon:.3f} | "
                  f"Goals: {goals_reached} | "
                  f"Q-states: {len(agent.q_table)} | "
                  f"Time: {elapsed:.1f}s")

    # Extract final greedy policy path
    greedy_path = _extract_greedy_path(env, agent)
    if greedy_path:
        cost = _path_cost(coarse_grid, greedy_path)
        successful_paths.append((cost, greedy_path))

    # Sort by cost (best first)
    successful_paths.sort(key=lambda x: x[0])

    # Deduplicate: keep paths with meaningfully different costs
    unique_paths = []
    prev_cost = None
    for cost, path in successful_paths:
        if prev_cost is None or abs(cost - prev_cost) / (prev_cost + 1e-6) > 0.05:
            # Upsample back to original grid
            if downsample_factor > 1:
                upsampled = upsample_path(path, downsample_factor)
                # Clamp to original grid bounds
                orig_w, orig_h = grid.shape
                upsampled = [(min(x, orig_w-1), min(y, orig_h-1))
                             for x, y in upsampled]
                unique_paths.append(upsampled)
            else:
                unique_paths.append(path)
            prev_cost = cost
        if len(unique_paths) >= 8:
            break

    elapsed = time.time() - start_time
    print(f"      RL Complete: {goals_reached} goals reached, "
          f"{len(unique_paths)} unique paths, {elapsed:.1f}s")

    return unique_paths


def _extract_greedy_path(env, agent):
    """Follow the greedy (no exploration) policy to extract learned path."""
    state = env.reset()
    path = [state]
    visited = {state}

    for _ in range(env.max_steps):
        action = agent.choose_greedy(state)
        next_state, reward, done, info = env.step(action)

        if next_state in visited:
            break  # Loop detected

        visited.add(next_state)
        path.append(next_state)
        state = next_state

        if done:
            break

    if info.get("reason") == "goal_reached":
        return path
    return None


def _path_cost(grid, path):
    """Compute traversal cost of a cell-by-cell path."""
    total = 0.0
    gw, gh = grid.shape
    for i in range(1, len(path)):
        x, y = path[i]
        px, py = path[i-1]
        if 0 <= x < gw and 0 <= y < gh:
            dx, dy = abs(x - px), abs(y - py)
            move = 1.414 if dx > 0 and dy > 0 else 1.0
            total += move * grid[x][y]
        else:
            total += 9999
    return total
