"""
Hybrid RL + GA Optimizer for Road Bypass Planning (v2)
=======================================================
Two-phase approach:
  Phase 1 — RL Exploration: Q-Learning on downsampled grid discovers feasible routes
  Phase 2 — GA Refinement: Evolves routes on a moderately downsampled grid

Called by road-gen.py to replace A* pathfinding.
"""

import time
import numpy as np
from rl_environment import train_rl_agent, downsample_grid, upsample_path
from ga_optimizer import optimize_routes


def find_optimal_bypass(grid, start, goal, obstacle_threshold=9999,
                        rl_episodes=1500, rl_downsample=3,
                        ga_population=50, ga_generations=60,
                        ga_waypoints=10, ga_downsample=2):
    """
    Execute the hybrid RL + GA pipeline to find the optimal bypass route.

    Phase 1: Train RL agent on heavily downsampled grid to discover initial paths.
    Phase 2: Seed GA with RL paths and evolve on moderately downsampled grid.

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
        ga_downsample: Grid downsampling factor for GA (2 = 4x fewer cells).

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
    # PHASE 2: Genetic Algorithm Refinement (on downsampled grid)
    # ------------------------------------------------------------------
    print(f"\n   [Phase 2] GA Refinement ({ga_generations} generations, "
          f"pop {ga_population})...")

    # Downsample grid for GA to run much faster
    if ga_downsample > 1:
        ga_grid = downsample_grid(grid, ga_downsample)
        ga_start = (start[0] // ga_downsample, start[1] // ga_downsample)
        ga_goal = (goal[0] // ga_downsample, goal[1] // ga_downsample)

        gw, gh = ga_grid.shape
        ga_start = (max(0, min(gw - 1, ga_start[0])),
                    max(0, min(gh - 1, ga_start[1])))
        ga_goal = (max(0, min(gw - 1, ga_goal[0])),
                   max(0, min(gh - 1, ga_goal[1])))

        # Also downsample RL seed paths to match GA grid
        ga_seed_paths = None
        if rl_paths:
            ga_seed_paths = []
            for path in rl_paths:
                ds_path = [(x // ga_downsample, y // ga_downsample) for x, y in path]
                # Clamp
                ds_path = [(max(0, min(gw - 1, x)), max(0, min(gh - 1, y)))
                           for x, y in ds_path]
                ga_seed_paths.append(ds_path)

        print(f"   GA grid: {ga_grid.shape} (downsampled {ga_downsample}x from {grid.shape})")
    else:
        ga_grid = grid
        ga_start = start
        ga_goal = goal
        ga_seed_paths = rl_paths if rl_paths else None

    best_path, best_fitness = optimize_routes(
        ga_grid, ga_start, ga_goal,
        obstacle_threshold=obstacle_threshold,
        seed_paths=ga_seed_paths,
        population_size=ga_population,
        n_generations=ga_generations,
        n_waypoints=ga_waypoints,
        crossover_prob=0.7,
        mutation_prob=0.35
    )

    # ------------------------------------------------------------------
    # Upsample GA result back to original grid and validate
    # ------------------------------------------------------------------
    if best_path and len(best_path) >= 2:
        # Upsample if we downsampled
        if ga_downsample > 1:
            best_path = upsample_path(best_path, ga_downsample)

        # Remove any remaining obstacle cells
        gw, gh = grid.shape
        valid_path = []
        for x, y in best_path:
            x = max(0, min(gw - 1, x))
            y = max(0, min(gh - 1, y))
            if grid[x][y] < obstacle_threshold:
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
