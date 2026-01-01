# Project Presentation: AI-Driven Road Generation & Traffic Signal Optimization

## 1. Project Overview
**Title:** Automated Urban Congestion Relief using AI
**Objective:** To create an intelligent system that autonomously monitors traffic, detects heavy congestion, and proposes infrastructure solutions (new roads) or operational optimizations (signal timing) to mitigate traffic jams.

### Core Technologies
*   **Simulation**: SUMO (Simulation of Urban MObility) for realistic traffic modeling.
*   **Map Data**: OpenStreetMap (OSM) for real-world road networks.
*   **Languages**: Python 3.10+
*   **Algorithms**:
    1.  **A* Search Algorithm**: For optimal road path planning.
    2.  **Genetic Algorithm (GA)**: For optimizing traffic signal phases.

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

### Module 2: Traffic Analysis & Road Proposal
**File:** `traffic_optimizer.py`
**Purpose:** Runs the traffic simulation, identifies the "worst" road, and plans a new bypass road avoiding physical obstacles.

**Key Functions:**
1.  **`run_simulation_pipeline()`**:
    *   Generates random traffic trips using SUMO's `randomTrips.py`.
    *   Runs a headless simulation to collect data.
    *   Produces `edge_data.xml` containing congestion metrics.

2.  **`find_worst_congestion()`**:
    *   **Logic**: Parses `edge_data.xml` and compares the `timeLoss` (seconds lost due to traffic) for every road.
    *   **Output**: Identifies the Edge ID with the maximum time loss (The "Problem").

3.  **`plan_road(start_coord, end_coord)`**:
    *   **Logic**: Implements the A* Search algorithm.
    *   **Grid Mapping**: Discretizes the world into a 4x4 meter grid.
    *   **Obstacle Parsing**: Reads `area.poly.xml` and marks buildings/water as "Blocked" (1) and empty land as "Free" (0).
    *   **Search**: Finds the shortest path from the start of the congested edge to its end through "Free" cells.

### Module 3: Traffic Signal Optimization (Genetic Algorithm)
**Purpose:** Optimizes the green light durations for traffic signals to improve flow at intersections.

**Implementation Logic:**
*   **Representation**: Each "Individual" is a set of green light durations for the traffic lights.
*   **Evolution**: The system evolves these timings over generations to find the most efficient combination.

---

## 3. Reward & Cost Function Explanation

This section details the mathematical "goals" used by the AI agents to make decisions.

### A. Road Generation Cost Function (A* Algorithm)
Used in `traffic_optimizer.py` to draw the road. The algorithm minimizes the Total Cost $f(n)$.

$$ f(n) = g(n) + h(n) $$

1.  **Road Cost ($g(n)$)**: The actual "price" of building road up to current point.
    *   **Straight Move**: Cost = 1.0
    *   **Diagonal Move**: Cost = $\sqrt{2} \approx 1.414$
    *   *Explanation*: This ensures the road takes the physical shortest path and doesn't zigzag unnecessarily.

2.  **Heuristic ($h(n)$)**: The estimated remaining distance.
    *   **Formula**: Euclidean Distance $= \sqrt{(x_{goal} - x_{current})^2 + (y_{goal} - y_{current})^2}$
    *   *Explanation*: This "pulls" the road search towards the destination, making the search efficient.

### B. Signal Optimization Reward Function (Genetic Algorithm)
Used to evaluate how "good" a specific set of traffic signal timings is.

1.  **The Objective**: Minimize the time drivers spend waiting at red lights.
2.  **Fitness Function**:
    
    $$ \text{Fitness} = \frac{1}{\text{Total Average Waiting Time}} $$

    *   **Explanation**: 
        *   The simulation runs with a specific set of green light timings.
        *   We measure the **Average Waiting Time** (in seconds) for all cars.
        *   Since we want to *minimize* wait time, but Genetic Algorithms typically *maximize* fitness, we take the inverse.
        *   **Lower Wait Time** $\rightarrow$ **Higher Fitness** $\rightarrow$ **Higher chance of survival**.

---

## 4. Main Functions Summary

| Function | File | Description |
| :--- | :--- | :--- |
| `extract_features_from_osm` | `extract_env.py` | Scans OSM data to identify buildings and water for collision checking. |
| `run_simulation_pipeline` | `traffic_optimizer.py` | Orchestrates the SUMO simulation to generate traffic data. |
| `find_worst_congestion` | `traffic_optimizer.py` | Analytic function that pinpoints the road segment with the highest delay. |
| `plan_road` | `traffic_optimizer.py` | The "Brain" of road generation; uses A* to generate the geometry of the new bypass. |

---

## 5. Visual Demonstration

To verify the results of the project, use the following command to visualize the generated road in SUMO GUI:

```bash
sumo-gui -n area.net.xml -a area.poly.xml,proposal_layer.xml
```

*   **Red Road**: The detected congested segment.
*   **Green Road**: The A* generated bypass avoiding buildings.
