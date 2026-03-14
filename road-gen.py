import os
import sys
import subprocess
import xml.etree.ElementTree as ET
import sumolib
import numpy as np
import math
import json
from shapely.geometry import Polygon, Point, LineString
from hybrid_optimizer import find_optimal_bypass

# --- Configuration ---
NET_FILE = "area.net.xml"
POLY_FILE = "area.poly.xml"
OUTPUT_PROPOSAL = "proposal_layer.xml"
TRIPS_FILE = "trips.xml"
ROUTE_FILE = "routes.xml"
EDGE_DATA_FILE = "edge_data.xml"

# Load config from file if it exists (written by Flask backend)
_config = {}
_config_path = os.path.join(os.getcwd(), "road_gen_config.json")
if os.path.exists(_config_path):
    with open(_config_path, 'r') as _cf:
        _config = json.load(_cf)
    print(f"Loaded config from {_config_path}: {_config}")

GRID_RES = float(_config.get("GRID_RES", 5.0))  # 5 meters per cell

# PENALTY CONFIGURATION
COST_OBSTACLE = int(_config.get("COST_OBSTACLE", 9999))    # Buildings/Water
COST_EXISTING_ROAD = int(_config.get("COST_EXISTING_ROAD", 60)) # Don't overlap with other roads
COST_CONGESTION_CORE = int(_config.get("COST_CONGESTION_CORE", 200)) # The red zone (forbidden)
COST_CONGESTION_NEAR = int(_config.get("COST_CONGESTION_NEAR", 40))  # Close to red zone (high drag)
COST_CONGESTION_FAR = int(_config.get("COST_CONGESTION_FAR", 5))    # Influence zone (slight drag)
COST_EMPTY = int(_config.get("COST_EMPTY", 1))          # Green field (cheapest)

# HYBRID RL + GA CONFIGURATION
RL_EPISODES = int(_config.get("RL_EPISODES", 1500))
RL_DOWNSAMPLE = int(_config.get("RL_DOWNSAMPLE", 3))
GA_POPULATION = int(_config.get("GA_POPULATION", 80))
GA_GENERATIONS = int(_config.get("GA_GENERATIONS", 100))
GA_WAYPOINTS = int(_config.get("GA_WAYPOINTS", 10))


def check_sumo_env():
    if 'SUMO_HOME' not in os.environ:
        print("Error: SUMO_HOME is not set.")
        sys.exit(1)
    return os.environ['SUMO_HOME']

def run_simulation_pipeline(sumo_home):
    print("--- Phase 1: Traffic Simulation ---")
    if not os.path.exists(TRIPS_FILE):
        print("1. Generating random traffic...")
        subprocess.run([
            "python", os.path.join(sumo_home, "tools", "randomTrips.py"),
            "-n", NET_FILE, "-o", TRIPS_FILE, "-r", ROUTE_FILE, 
            "-e", "600", "-p", "2.0", "--validate"
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("2. Simulating traffic flow...")
    subprocess.run([
        "sumo", "-n", NET_FILE, "-r", ROUTE_FILE, 
        "--edgedata-output", EDGE_DATA_FILE, 
        "--begin", "0", "--end", "800", "--no-step-log"
    ], check=True)

def find_worst_congestion():
    print("3. Analyzing congestion data...")
    tree = ET.parse(EDGE_DATA_FILE)
    root = tree.getroot()
    worst_edge = None
    max_time_loss = -1.0
    
    for interval in root.findall('interval'):
        for edge in interval.findall('edge'):
            loss = float(edge.get('timeLoss', 0))
            if loss > max_time_loss:
                max_time_loss = loss
                worst_edge = edge.get('id')
    
    if worst_edge:
        print(f"   >> CRITICAL BOTTLENECK: Edge {worst_edge} (Lag: {max_time_loss:.2f}s)")
    return worst_edge

# --- HYBRID RL + GA PATHFINDING ---

def get_valid_node(grid, x, y, w, h):
    """Finds nearest non-obstacle node."""
    if 0 <= x < w and 0 <= y < h and grid[x][y] < COST_OBSTACLE:
        return x, y
    for radius in range(1, 10): # Search radius
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if grid[nx][ny] < COST_OBSTACLE:
                        return nx, ny
    return None

def rasterize_polygon(grid, polygon, val, min_x, min_y, grid_w, grid_h):
    b_min_x, b_min_y, b_max_x, b_max_y = polygon.bounds
    g_min_x = max(0, int((b_min_x - min_x)/GRID_RES))
    g_max_x = min(grid_w, int((b_max_x - min_x)/GRID_RES)+1)
    g_min_y = max(0, int((b_min_y - min_y)/GRID_RES))
    g_max_y = min(grid_h, int((b_max_y - min_y)/GRID_RES)+1)
    
    for x in range(g_min_x, g_max_x):
        for y in range(g_min_y, g_max_y):
            wx = min_x + x*GRID_RES + GRID_RES/2
            wy = min_y + y*GRID_RES + GRID_RES/2
            if polygon.contains(Point(wx, wy)):
                # Only overwrite if new cost is higher (to keep obstacles supreme)
                if val > grid[x][y]:
                    grid[x][y] = val

def find_upstream_downstream_anchors(net, edge_id, depth=4):
    """Finds anchor points several nodes upstream and downstream to create a wider arc."""
    center_edge = net.getEdge(edge_id)
    
    # Trace Backwards (Upstream)
    start_node = center_edge.getFromNode()
    for _ in range(depth):
        incoming = list(start_node.getIncoming())
        if not incoming: break
        # Choose longest incoming edge to avoid tiny connector edges
        best_prev = max(incoming, key=lambda e: e.getLength())
        start_node = best_prev.getFromNode()

    # Trace Forwards (Downstream)
    end_node = center_edge.getToNode()
    for _ in range(depth):
        outgoing = list(end_node.getOutgoing())
        if not outgoing: break
        best_next = max(outgoing, key=lambda e: e.getLength())
        end_node = best_next.getToNode()
        
    return start_node, end_node

def plan_road(edge_id):
    print("\n--- Phase 2: Strategic Bypass Planning ---")
    
    net = sumolib.net.readNet(NET_FILE)
    bbox = net.getBBoxXY()
    if isinstance(bbox[0], (list, tuple)):
        (min_x, min_y), (max_x, max_y) = bbox
    else:
        min_x, min_y, max_x, max_y = bbox
        
    w_m, h_m = max_x - min_x, max_y - min_y
    grid_w, grid_h = int(w_m/GRID_RES)+1, int(h_m/GRID_RES)+1
    
    # 1. Initialize Grid with lowest cost
    grid = np.ones((grid_w, grid_h), dtype=np.int32) * COST_EMPTY
    
    # 2. Apply Repulsion Field (The "Wide Swing" Logic)
    # We buffer the congested road at different distances with decreasing costs
    print("   Applying repulsion gradients...")
    bad_edge = net.getEdge(edge_id)
    bad_poly = LineString(bad_edge.getShape())
    
    # Order matters: Paint broad strokes first, then tight strokes
    # 150m buffer -> Slight drag
    rasterize_polygon(grid, bad_poly.buffer(150.0), COST_CONGESTION_FAR, min_x, min_y, grid_w, grid_h)
    # 60m buffer -> High drag
    rasterize_polygon(grid, bad_poly.buffer(60.0), COST_CONGESTION_NEAR, min_x, min_y, grid_w, grid_h)
    # 20m buffer -> Forbidden core
    rasterize_polygon(grid, bad_poly.buffer(20.0), COST_CONGESTION_CORE, min_x, min_y, grid_w, grid_h)

    # 3. Mark ALL existing roads (Prevent overlap)
    print("   Masking existing road network...")
    for edge in net.getEdges():
        # Buffer existing roads by 8m to prevent touching them
        shape = edge.getShape()
        if len(shape) < 2: continue
        road_poly = LineString(shape).buffer(8.0)
        rasterize_polygon(grid, road_poly, COST_EXISTING_ROAD, min_x, min_y, grid_w, grid_h)

    # 4. Mark Buildings (Strict Obstacles)
    print("   Marking buildings...")
    tree = ET.parse(POLY_FILE)
    root = tree.getroot()
    for poly in root.findall('poly'):
        ptype = poly.get('type', 'unknown')
        if ptype in ['building', 'water', 'forest']:
            shape = poly.get('shape')
            coords = [tuple(map(float, p.split(','))) for p in shape.split()]
            if len(coords) >= 3:
                rasterize_polygon(grid, Polygon(coords), COST_OBSTACLE, min_x, min_y, grid_w, grid_h)

    # 5. Find Distant Anchors
    start_node, end_node = find_upstream_downstream_anchors(net, edge_id, depth=4)
    print(f"   Anchor Search: Start moved to {start_node.getID()}, End moved to {end_node.getID()}")
    
    start_coord = start_node.getCoord()
    end_coord = end_node.getCoord()
    
    raw_sx, raw_sy = int((start_coord[0]-min_x)/GRID_RES), int((start_coord[1]-min_y)/GRID_RES)
    raw_ex, raw_ey = int((end_coord[0]-min_x)/GRID_RES), int((end_coord[1]-min_y)/GRID_RES)

    start_pt = get_valid_node(grid, raw_sx, raw_sy, grid_w, grid_h)
    end_pt = get_valid_node(grid, raw_ex, raw_ey, grid_w, grid_h)

    if not start_pt or not end_pt:
        print("Error: Could not find valid start/end points.")
        return

    # 6. Hybrid RL + GA Search
    print(f"   Running Hybrid RL + GA Optimizer...")
    
    optimal_path = find_optimal_bypass(
        grid, start_pt, end_pt,
        obstacle_threshold=COST_OBSTACLE,
        rl_episodes=RL_EPISODES,
        rl_downsample=RL_DOWNSAMPLE,
        ga_population=GA_POPULATION,
        ga_generations=GA_GENERATIONS,
        ga_waypoints=GA_WAYPOINTS
    )

    if optimal_path:
        print(f"   SUCCESS: Bypass found! Path points: {len(optimal_path)}")
        path_str = []
        for i, (gx, gy) in enumerate(optimal_path):
            # Sub-sample points to smooth the line slightly
            if i % 2 == 0:
                wx = min_x + gx * GRID_RES
                wy = min_y + gy * GRID_RES
                path_str.append(f"{wx:.2f},{wy:.2f}")
        
        # Save Output
        with open(OUTPUT_PROPOSAL, 'w') as f:
            f.write('<additional>\n')
            # Visualizing the congestion (RED)
            orig_shape = bad_edge.getShape()
            f.write(f'    <poly id="congestion_red" color="255,0,0" lineWidth="10" layer="100" '
                    f'shape="{orig_shape[0][0]},{orig_shape[0][1]} {orig_shape[-1][0]},{orig_shape[-1][1]}"/>\n')
            # The New Bypass (GREEN)
            f.write(f'    <poly id="bypass_green" color="0,255,0" lineWidth="8" layer="101" '
                    f'shape="{" ".join(path_str)}"/>\n')
            f.write('</additional>\n')
        print(f"   Proposal saved to {OUTPUT_PROPOSAL}")
    else:
        print("   FAILED: No path found.")

def main():
    sumo = check_sumo_env()
    run_simulation_pipeline(sumo)
    
    worst_edge_id = find_worst_congestion()
    if worst_edge_id:
        plan_road(worst_edge_id)
        print(f"\nVisualize: sumo-gui -n {NET_FILE} -a {POLY_FILE},{OUTPUT_PROPOSAL}")

if __name__ == "__main__":
    main()