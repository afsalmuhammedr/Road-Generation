"""
Genetic Algorithm Optimizer for Road Route Evolution (v2)
==========================================================
Uses DEAP to evolve a population of candidate bypass routes.

Key improvements over v1:
- Cost-grid-aware interpolation between waypoints (NOT Bresenham straight lines)
- Each path segment walks cell-by-cell, greedily choosing the lowest-cost
  neighbor that moves toward the next waypoint
- This ensures paths NEVER cut through buildings or water
- More waypoints and tighter mutation for fine-grained control

Part of the Hybrid RL + GA road generation pipeline.
"""

import random
import math
import numpy as np
import time
from deap import base, creator, tools, algorithms


# ---------------------------------------------------------------------------
# DEAP types (guarded for reloads)
# ---------------------------------------------------------------------------
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


# 8 directions for grid walking
_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]


# ---------------------------------------------------------------------------
# Cost-grid-aware path interpolation (replaces Bresenham)
# ---------------------------------------------------------------------------

def _grid_walk(grid, start, end, obstacle_threshold, max_iters=5000):
    """
    Walk cell-by-cell from start toward end on the cost grid.
    At each step, pick the neighbor that:
      1. Is not an obstacle
      2. Is closest to the target 'end' (greedy)
      3. Has the lowest cell cost (tie-breaker)

    This produces a path that naturally avoids obstacles and high-cost zones.
    """
    gw, gh = grid.shape
    x, y = int(start[0]), int(start[1])
    ex, ey = int(end[0]), int(end[1])

    path = [(x, y)]
    visited = {(x, y)}
    iters = 0

    while iters < max_iters:
        # Close enough to target
        if abs(x - ex) <= 1 and abs(y - ey) <= 1:
            if (ex, ey) != (x, y):
                path.append((ex, ey))
            break

        # Evaluate all neighbors
        best_next = None
        best_score = float('inf')

        for dx, dy in _DIRS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < gw and 0 <= ny < gh):
                continue
            if grid[nx][ny] >= obstacle_threshold:
                continue
            if (nx, ny) in visited:
                continue

            # Score = weighted distance to target + cell cost penalty
            dist = math.sqrt((nx - ex)**2 + (ny - ey)**2)
            cost_weight = grid[nx][ny] / 10.0  # Normalize cost influence
            score = dist + cost_weight

            if score < best_score:
                best_score = score
                best_next = (nx, ny)

        if best_next is None:
            # Stuck — allow revisiting with higher penalty
            for dx, dy in _DIRS:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < gw and 0 <= ny < gh):
                    continue
                if grid[nx][ny] >= obstacle_threshold:
                    continue
                dist = math.sqrt((nx - ex)**2 + (ny - ey)**2)
                if dist < best_score:
                    best_score = dist
                    best_next = (nx, ny)

        if best_next is None:
            break  # Truly stuck

        x, y = best_next
        visited.add((x, y))
        path.append((x, y))
        iters += 1

    return path


def interpolate_waypoints_on_grid(waypoints, grid, obstacle_threshold):
    """
    Connect waypoints with cost-grid-aware cell-by-cell walking.
    This replaces Bresenham and ensures paths respect the grid.
    """
    if len(waypoints) < 2:
        return list(waypoints)

    full_path = []
    for i in range(len(waypoints) - 1):
        segment = _grid_walk(grid, waypoints[i], waypoints[i + 1], obstacle_threshold)
        if i > 0 and segment:
            segment = segment[1:]  # Skip duplicate junction point
        full_path.extend(segment)

    def remove_tangles(path):
        if len(path) < 3:
            return path
            
        last_seen = {}
        for idx, p in enumerate(path):
            last_seen[p] = idx
            
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            cx, cy = path[i]
            best_j = i + 1
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor = (cx + dx, cy + dy)
                    if neighbor in last_seen:
                        idx = last_seen[neighbor]
                        if idx > best_j:
                            best_j = idx
                            
            smoothed.append(path[best_j])
            i = best_j
            
        return smoothed

    # Apply tangle removal iteratively until path length stabilizes
    prev_len = len(full_path)
    for _ in range(3):
        full_path = remove_tangles(full_path)
        if len(full_path) == prev_len:
            break
        prev_len = len(full_path)

    return full_path


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def evaluate_route(individual, grid, start, goal, obstacle_threshold):
    """
    Evaluate fitness of an individual (list of waypoints).
    Lower is better (minimization).
    """
    gw, gh = grid.shape

    # Build waypoint chain: start → individual waypoints → goal
    all_waypoints = [start]
    for wp in individual:
        wx, wy = int(wp[0]), int(wp[1])
        wx = max(0, min(gw - 1, wx))
        wy = max(0, min(gh - 1, wy))
        all_waypoints.append((wx, wy))
    all_waypoints.append(goal)

    # Interpolate on the cost grid (cell-by-cell)
    path = interpolate_waypoints_on_grid(all_waypoints, grid, obstacle_threshold)

    if len(path) < 2:
        return (1e8,)

    # 1. Traversal cost (sum of cell costs along path)
    traversal_cost = 0.0
    obstacle_hits = 0
    for i in range(1, len(path)):
        x, y = path[i]
        px, py = path[i - 1]
        if 0 <= x < gw and 0 <= y < gh:
            cell_cost = grid[x][y]
            if cell_cost >= obstacle_threshold:
                obstacle_hits += 1
            else:
                dx, dy = abs(x - px), abs(y - py)
                move = 1.414 if dx > 0 and dy > 0 else 1.0
                traversal_cost += move * cell_cost
        else:
            traversal_cost += 500

    # 2. Obstacle penalty
    obstacle_penalty = obstacle_hits * 50000

    # 3. Path length factor (prefer shorter paths, but not too aggressively)
    length_cost = len(path) * 0.5

    # 4. Smoothness (penalize sharp turns at waypoints)
    smoothness_penalty = 0.0
    if len(all_waypoints) >= 3:
        for i in range(1, len(all_waypoints) - 1):
            p0 = all_waypoints[i - 1]
            p1 = all_waypoints[i]
            p2 = all_waypoints[i + 1]
            v1 = (p1[0] - p0[0], p1[1] - p0[1])
            v2 = (p2[0] - p1[0], p2[1] - p1[1])
            m1 = math.sqrt(v1[0]**2 + v1[1]**2)
            m2 = math.sqrt(v2[0]**2 + v2[1]**2)
            if m1 > 0 and m2 > 0:
                cos_a = max(-1, min(1, (v1[0]*v2[0] + v1[1]*v2[1]) / (m1 * m2)))
                angle = math.acos(cos_a)
                if angle > math.pi / 2:
                    smoothness_penalty += (angle - math.pi / 2) * 1000

    # 5. Goal proximity (penalize if path doesn't reach goal)
    last = path[-1]
    goal_dist = math.sqrt((last[0] - goal[0])**2 + (last[1] - goal[1])**2)
    goal_penalty = goal_dist * 100 if goal_dist > 3 else 0

    # 6. Self-Intersection / Loop Penalty
    duplicate_cells = len(path) - len(set(path))
    loop_penalty = duplicate_cells * 50000

    # 7. Micro-path Reversal Penalty
    reversal_penalty = 0.0
    for i in range(2, len(path)):
        if path[i] == path[i - 2]:
            reversal_penalty += 5000

    return (traversal_cost + obstacle_penalty + length_cost + smoothness_penalty + goal_penalty + loop_penalty + reversal_penalty,)


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def create_random_individual(start, goal, grid_w, grid_h, n_waypoints):
    """Create individual with waypoints distributed between start and goal."""
    waypoints = []
    for i in range(n_waypoints):
        t = (i + 1) / (n_waypoints + 1)
        bx = start[0] + t * (goal[0] - start[0])
        by = start[1] + t * (goal[1] - start[1])

        # Random offset perpendicular to the start→goal line
        spread = max(grid_w, grid_h) * 0.2
        wx = int(max(0, min(grid_w - 1, bx + random.gauss(0, spread))))
        wy = int(max(0, min(grid_h - 1, by + random.gauss(0, spread))))
        waypoints.append((wx, wy))

    return creator.Individual(waypoints)


def create_from_path(path, n_waypoints):
    """Create individual by sampling waypoints from an existing path."""
    if len(path) < 3:
        return None
    indices = np.linspace(1, len(path) - 2, n_waypoints, dtype=int)
    waypoints = [path[idx] for idx in indices]
    return creator.Individual(waypoints)


def mutate_waypoint(individual, grid_w, grid_h, indpb=0.35, sigma_frac=0.06):
    """Gaussian mutation of waypoints — smaller sigma for finer control."""
    sigma = max(grid_w, grid_h) * sigma_frac
    for i in range(len(individual)):
        if random.random() < indpb:
            x, y = individual[i]
            x = int(max(0, min(grid_w - 1, x + random.gauss(0, sigma))))
            y = int(max(0, min(grid_h - 1, y + random.gauss(0, sigma))))
            individual[i] = (x, y)
    return (individual,)


def crossover_waypoints(ind1, ind2):
    """Single-point crossover of waypoint sequences."""
    size = min(len(ind1), len(ind2))
    if size < 2:
        return ind1, ind2
    cx = random.randint(1, size - 1)
    ind1[cx:], ind2[cx:] = ind2[cx:], ind1[cx:]
    return ind1, ind2


# ---------------------------------------------------------------------------
# Main GA optimization
# ---------------------------------------------------------------------------

def optimize_routes(grid, start, goal, obstacle_threshold=9999,
                    seed_paths=None, population_size=80,
                    n_generations=100, n_waypoints=10,
                    crossover_prob=0.7, mutation_prob=0.35):
    """
    Evolve optimal bypass routes using a Genetic Algorithm.

    Waypoints are connected by cost-grid-aware walks, so paths always
    respect obstacles and the cost field.
    """
    start_time = time.time()
    grid_w, grid_h = grid.shape
    start = tuple(start)
    goal = tuple(goal)

    print(f"      GA Optimization: pop={population_size}, gen={n_generations}, "
          f"waypoints={n_waypoints}, grid={grid_w}x{grid_h}")

    # --- DEAP toolbox ---
    toolbox = base.Toolbox()
    toolbox.register("individual", create_random_individual,
                     start, goal, grid_w, grid_h, n_waypoints)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_route,
                     grid=grid, start=start, goal=goal,
                     obstacle_threshold=obstacle_threshold)
    toolbox.register("mate", crossover_waypoints)
    toolbox.register("mutate", mutate_waypoint,
                     grid_w=grid_w, grid_h=grid_h,
                     indpb=0.35, sigma_frac=0.06)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # --- Create population ---
    pop = toolbox.population(n=population_size)

    # Seed from RL paths
    n_seeded = 0
    if seed_paths:
        for path in seed_paths:
            if n_seeded >= population_size // 2:
                break
            seeded = create_from_path(path, n_waypoints)
            if seeded:
                pop[n_seeded] = seeded
                n_seeded += 1
                # Add mutated variants
                for _ in range(3):
                    if n_seeded >= population_size // 2:
                        break
                    variant = creator.Individual(list(seeded))
                    mutate_waypoint(variant, grid_w, grid_h, indpb=0.5, sigma_frac=0.1)
                    pop[n_seeded] = variant
                    n_seeded += 1
        print(f"        Seeded {n_seeded}/{population_size} individuals from RL paths")

    # --- Evolution with manual loop for progress logging ---
    hof = tools.HallOfFame(3)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    print(f"        Gen   0 | Best: {hof[0].fitness.values[0]:>10.1f} | "
          f"Time: {time.time()-start_time:.1f}s")

    for gen in range(1, n_generations + 1):
        # Selection
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < crossover_prob:
                toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Mutation
        for i in range(len(offspring)):
            if random.random() < mutation_prob:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # Evaluate new individuals
        invalids = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalids))
        for ind, fit in zip(invalids, fitnesses):
            ind.fitness.values = fit

        # Elitism: keep best from previous generation
        pop[:] = offspring
        hof.update(pop)

        # Progress logging
        if gen % 20 == 0 or gen == n_generations:
            best_fit = hof[0].fitness.values[0]
            avg_fit = np.mean([ind.fitness.values[0] for ind in pop])
            elapsed = time.time() - start_time
            print(f"        Gen {gen:>3d} | Best: {best_fit:>10.1f} | "
                  f"Avg: {avg_fit:>10.1f} | Time: {elapsed:.1f}s")

    # --- Extract best path ---
    best_ind = hof[0]
    best_fitness = best_ind.fitness.values[0]

    # Reconstruct full path
    all_wp = [start] + [(int(wp[0]), int(wp[1])) for wp in best_ind] + [goal]
    best_path = interpolate_waypoints_on_grid(all_wp, grid, obstacle_threshold)

    elapsed = time.time() - start_time
    print(f"      GA Complete: fitness={best_fitness:.1f}, "
          f"path={len(best_path)} pts, {elapsed:.1f}s")

    return best_path, best_fitness
