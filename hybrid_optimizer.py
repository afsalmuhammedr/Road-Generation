"""
Hybrid RL + GA Optimizer for Road Bypass Planning (v2)
=======================================================
Two-phase approach:
  Phase 1 — RL Exploration: Q-Learning on downsampled grid discovers feasible routes
  Phase 2 — GA Refinement: Evolves routes using cost-grid-aware interpolation

Called by road-gen.py to replace A* pathfinding.
"""

import time
from rl_environment import train_rl_agent
from ga_optimizer import optimize_routes


def find_optimal_bypass(grid, start, goal, obstacle_threshold=9999,
                        rl_episodes=1500, rl_downsample=3,
                        ga_population=80, ga_generations=100,
                        ga_waypoints=10):
    """
    Execute the hybrid RL + GA pipeline to find the optimal bypass route.

    Phase 1: Train RL agent on downsampled grid to discover initial paths.
    Phase 2: Seed GA with RL paths and evolve on full-resolution grid.

    Args:
        grid: 2D numpy array (grid_w x grid_h) of cell costs.
        start: (x, y) start position on grid.
        goal: (x, y) goal position on grid.
        obstacle_threshold: Cells >= this are impassable.
        rl_episodes: RL training episodes.
        rl_downsample: Grid downsampling factor for RL.
        ga_population: GA population size.
        ga_generations: GA generations.
        ga_waypoints: Intermediate waypoints per route.

    Returns:
        path: List of (x, y) grid coordinates, or None if no path found.
    """
    start = tuple(start)
    goal = tuple(goal)
    total_start = time.time()

    print("\n   ============================================")
    print("   === Hybrid RL + GA Route Optimization ===")
    print("   ============================================")

    # ------------------------------------------------------------------
    # PHASE 1: Reinforcement Learning Exploration
    # ------------------------------------------------------------------
    print(f"\n   [Phase 1] RL Exploration ({rl_episodes} episodes)...")

    rl_paths = train_rl_agent(
        grid, start, goal,
        obstacle_threshold=obstacle_threshold,
        n_episodes=rl_episodes,
        downsample_factor=rl_downsample
    )

    if rl_paths:
        print(f"   [Phase 1] SUCCESS: {len(rl_paths)} candidate routes discovered")
    else:
        print("   [Phase 1] WARNING: No RL routes found — GA will explore from scratch")

    # ------------------------------------------------------------------
    # PHASE 2: Genetic Algorithm Refinement
    # ------------------------------------------------------------------
    print(f"\n   [Phase 2] GA Refinement ({ga_generations} generations, "
          f"pop {ga_population})...")

    best_path, best_fitness = optimize_routes(
        grid, start, goal,
        obstacle_threshold=obstacle_threshold,
        seed_paths=rl_paths if rl_paths else None,
        population_size=ga_population,
        n_generations=ga_generations,
        n_waypoints=ga_waypoints,
        crossover_prob=0.7,
        mutation_prob=0.35
    )

    # ------------------------------------------------------------------
    # Validate & clean result
    # ------------------------------------------------------------------
    if best_path and len(best_path) >= 2:
        # Remove any remaining obstacle cells
        gw, gh = grid.shape
        valid_path = []
        for x, y in best_path:
            if (0 <= x < gw and 0 <= y < gh
                    and grid[x][y] < obstacle_threshold):
                valid_path.append((x, y))

        if len(valid_path) >= 2:
            total_time = time.time() - total_start
            print(f"\n   ============================================")
            print(f"   === Optimization Complete ===")
            print(f"   Final path: {len(valid_path)} points")
            print(f"   Fitness: {best_fitness:.1f}")
            print(f"   Total time: {total_time:.1f}s")
            print(f"   ============================================")
            return valid_path

    total_time = time.time() - total_start
    print(f"\n   === Optimization FAILED ({total_time:.1f}s) ===")
    return None
