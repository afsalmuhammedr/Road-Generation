# Project Presentation: AI-Driven Road Generation & Traffic Optimization

## 1. Project Overview
**Title:** Automated Urban Congestion Relief using AI
**Objective:** To create an intelligent system that autonomously monitors traffic, detects heavy congestion, and proposes infrastructure solutions (new roads) or operational optimizations (signal timing) to mitigate traffic jams.

### Core Technologies
*   **Simulation**: SUMO (Simulation of Urban MObility) for realistic traffic modeling.
*   **Map Data**: OpenStreetMap (OSM) for real-world road networks.
*   **Languages**: Python 3.10+
*   **Algorithms**:
    1.  **Reinforcement Learning (Q-Learning)**: For exploring feasible bypass routes on traffic grids.
    2.  **Genetic Algorithm (GA)**: For evolving and optimizing road paths and traffic signal phases.
    3.  **Hybrid RL + GA Pipeline**: A two-phase approach where RL discovers initial routes and GA refines them.

---

## 2. System Architecture & Implementation

The project consists of three main modules that work sequentially to analyze and solve traffic problems.

### Module 1: Environment Generation
**File:** `extract_env.py`
**Purpose:** Converts raw OpenStreetMap data into a format understandable by the SUMO simulator and extracts visual features like buildings and water bodies.

**Key Functions:**
*   `generate_network(osm_file, net_file)`: Uses `netconvert` to turn OSM XML into a SUMO network (`.net.xml`), handling geometry and topology.
*   `extract_features_from_osm(osm_file)`: Parses the OSM file to find non-road features (buildings, parks, water).
*   `get_feature_style(tags)`: Classifies features (e.g., 'building' = Grey, 'water' = Blue) to create a visual map (`.poly.xml`) used for obstacle avoidance.
*   `transform_and_write_poly(...)`: Converts geographic coordinates (Lat/Lon) to SUMO's Cartesian (X/Y) coordinate system.

### Module 2: Traffic Analysis & Hybrid Bypass Planning
**Files:** `road-gen.py`, `rl_environment.py`, `ga_optimizer.py`, `hybrid_optimizer.py`
**Purpose:** Runs the traffic simulation, identifies the worst congested road, and plans a new bypass road using a **hybrid RL + GA** approach.

**Key Functions:**
1.  **`run_simulation_pipeline()`** (`road-gen.py`):
    *   Generates random traffic trips using SUMO's `randomTrips.py`.
    *   Runs a headless simulation to collect data.
    *   Produces `edge_data.xml` containing congestion metrics.

2.  **`find_worst_congestion()`** (`road-gen.py`):
    *   **Logic**: Parses `edge_data.xml` and compares the `timeLoss` (seconds lost due to traffic) for every road.
    *   **Output**: Identifies the Edge ID with the maximum time loss (The "Problem").

3.  **`plan_road(edge_id)`** (`road-gen.py`):
    *   Constructs a cost grid with repulsion fields around the congested edge.
    *   Marks buildings, water, and existing roads as obstacles or high-cost zones.
    *   Calls the **hybrid optimizer** to find the optimal bypass.

4.  **`train_rl_agent()`** (`rl_environment.py`):
    *   **Phase 1** of the hybrid approach.
    *   Implements Q-Learning on the cost grid.
    *   Agent navigates from start to goal, learning to avoid obstacles and congestion.
    *   Produces multiple candidate paths ranked by quality.

5.  **`optimize_routes()`** (`ga_optimizer.py`):
    *   **Phase 2** of the hybrid approach.
    *   Uses **DEAP** library for evolutionary optimization.
    *   Individuals are represented as sequences of waypoints between start and goal.
    *   RL-discovered paths seed 1/3 of the initial population.
    *   Evolves routes through selection, crossover, and mutation over multiple generations.

6.  **`find_optimal_bypass()`** (`hybrid_optimizer.py`):
    *   Orchestrates the two-phase pipeline: RL → GA.
    *   Returns the best validated path as grid coordinates.

### Module 3: Traffic Signal Optimization (Genetic Algorithm)
**Purpose:** Optimizes the green light durations for traffic signals to improve flow at intersections.

**Implementation Logic:**
*   **Representation**: Each "Individual" is a set of green light durations for the traffic lights.
*   **Evolution**: The system evolves these timings over generations to find the most efficient combination.

---

## 3. Reward & Cost Function Explanation

This section details the mathematical "goals" used by the AI agents to make decisions.

### A. RL Reward Function (Q-Learning Agent)
Used in `rl_environment.py` where the agent learns to navigate the cost grid.

**State:** Current $(x, y)$ position on the discretized grid.
**Actions:** 8 directional moves (N, S, E, W, NE, NW, SE, SW).

**Reward Components:**

1.  **Step Penalty**: $R_{step} = -\frac{\text{move\_cost} \times \text{cell\_cost}}{10}$
    *   Move cost = $1.0$ (cardinal) or $\sqrt{2}$ (diagonal)
    *   Cell cost varies: 1 (empty) → 200 (congestion core)
    *   *Explanation*: Discourages traversing high-cost zones (congestion, near existing roads).

2.  **Distance Shaping**: $R_{shape} = 2 \times (d_{old} - d_{new})$
    *   Euclidean distance improvement toward the goal.
    *   *Explanation*: Guides the agent toward the destination.

3.  **Goal Reward**: $R_{goal} = +500$ when reaching within 1 cell of the destination.

4.  **Obstacle Penalty**: $R_{obstacle} = -1000$ (episode terminates) for hitting impassable cells.

### B. GA Fitness Function (Route Evolution)
Used in `ga_optimizer.py` to evaluate how "good" a candidate bypass route is.

$$ \text{Fitness} = C_{traversal} + P_{obstacles} + P_{smoothness} + P_{goal} $$

1.  **Traversal Cost** ($C_{traversal}$): Sum of $\text{move\_cost} \times \text{cell\_cost}$ for each step along the interpolated path.
    *   *Explanation*: Lower cost means the path avoids congestion and high-cost zones.

2.  **Obstacle Penalty** ($P_{obstacles}$): $10000 \times \text{obstacle\_hits}$
    *   *Explanation*: Extremely high penalty ensures paths never cross buildings or water.

3.  **Smoothness Penalty** ($P_{smoothness}$): $50 \times (\theta - \frac{\pi}{2})$ for turns sharper than 90°
    *   *Explanation*: Discourages sharp turns to produce realistic, buildable road geometry.

4.  **Goal Penalty** ($P_{goal}$): $10 \times d_{goal}$ if the path doesn't reach the destination.

**Genetic Operators:**
*   **Selection**: Tournament (size 3) — picks the fittest from random groups.
*   **Crossover**: Single-point swap of waypoint subsequences between two parents.
*   **Mutation**: Gaussian perturbation of waypoint positions.
*   **Seeding**: 1/3 of the initial population is seeded from RL-discovered routes.

### C. Signal Optimization Reward Function (Genetic Algorithm)
Used to evaluate how "good" a specific set of traffic signal timings is.

$$ \text{Fitness} = \frac{1}{\text{Total Average Waiting Time}} $$

*   **Explanation**: Lower wait time → Higher fitness → Higher chance of survival.

---

## 4. Hybrid Pipeline Summary

```
┌──────────────────────────────────────────────────────────┐
│  1. SUMO Traffic Simulation (randomTrips → edge_data)    │
├──────────────────────────────────────────────────────────┤
│  2. Congestion Detection (find worst edge by timeLoss)   │
├──────────────────────────────────────────────────────────┤
│  3. Cost Grid Construction                               │
│     - Buildings/water → OBSTACLE (9999)                  │
│     - Existing roads  → HIGH COST (60)                   │
│     - Congestion core → FORBIDDEN (200)                  │
│     - Congestion near → DRAG (40)                        │
│     - Empty land      → FREE (1)                         │
├──────────────────────────────────────────────────────────┤
│  4. PHASE 1: RL Exploration (Q-Learning)                 │
│     → Discovers initial feasible routes                  │
├──────────────────────────────────────────────────────────┤
│  5. PHASE 2: GA Refinement (DEAP)                        │
│     → Evolves optimal route from RL seeds                │
├──────────────────────────────────────────────────────────┤
│  6. Output: proposal_layer.xml (Red=congested, Green=new)│
└──────────────────────────────────────────────────────────┘
```

## 5. Main Functions Summary

| Function | File | Description |
| :--- | :--- | :--- |
| `extract_features_from_osm` | `extract_env.py` | Scans OSM data to identify buildings and water for collision checking. |
| `run_simulation_pipeline` | `road-gen.py` | Orchestrates the SUMO simulation to generate traffic data. |
| `find_worst_congestion` | `road-gen.py` | Analytic function that pinpoints the road segment with the highest delay. |
| `plan_road` | `road-gen.py` | Constructs cost grid and calls the hybrid optimizer. |
| `train_rl_agent` | `rl_environment.py` | Q-Learning agent explores the grid to discover bypass routes. |
| `optimize_routes` | `ga_optimizer.py` | Genetic Algorithm evolves route population for optimal fitness. |
| `find_optimal_bypass` | `hybrid_optimizer.py` | Orchestrates the RL → GA two-phase pipeline. |

---

## 6. Visual Demonstration

To verify the results of the project, use the following command to visualize the generated road in SUMO GUI:

```bash
sumo-gui -n area.net.xml -a area.poly.xml,proposal_layer.xml
```

*   **Red Road**: The detected congested segment.
*   **Green Road**: The RL+GA generated bypass avoiding buildings and congestion.
