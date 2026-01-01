import os
import sys
import subprocess
import xml.etree.ElementTree as ET
import sumolib
import numpy as np
import heapq
import math
from shapely.geometry import Polygon, Point, LineString

# --- Configuration ---
NET_FILE = "area.net.xml"
POLY_FILE = "area.poly.xml"
OUTPUT_PROPOSAL = "proposal_layer.xml"

TRIPS_FILE = "trips.xml"
ROUTE_FILE = "routes.xml"
EDGE_DATA_FILE = "edge_data.xml"

GRID_RES = 5.0  # 5 meters per cell

# PENALTY CONFIGURATION
COST_OBSTACLE = 9999  # Effectively infinite (Buildings/Water)
COST_CONGESTION = 50  # High cost to discourage using the existing road
COST_EMPTY = 1        # Low cost for using empty space

def check_sumo_env():
    if 'SUMO_HOME' not in os.environ:
        print("Error: SUMO_HOME is not set.")
        sys.exit(1)
    return os.environ['SUMO_HOME']

def run_simulation_pipeline(sumo_home):
    print("--- Phase 1: Traffic Simulation ---")
    print("1. Generating random traffic...")
    subprocess.run([
        "python", os.path.join(sumo_home, "tools", "randomTrips.py"),
        "-n", NET_FILE, "-o", TRIPS_FILE, "-r", ROUTE_FILE, 
        "-e", "600", "-p", "2.0", "--validate"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("2. Simulating traffic flow...")
    # FIX: Changed '--edge-data-output' to '--edgedata-output'
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

# --- ADVANCED A* WITH PENALTIES ---

class Node:
    def __init__(self, x, y, parent=None, g=0, h=0):
        self.x, self.y = x, y
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h
    def __lt__(self, other): return self.f < other.f

def get_valid_node(grid, x, y, w, h):
    """Finds nearest non-obstacle node."""
    if 0 <= x < w and 0 <= y < h and grid[x][y] < COST_OBSTACLE:
        return x, y
    
    # Search outwards
    for radius in range(1, 15):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if grid[nx][ny] < COST_OBSTACLE:
                        return nx, ny
    return None

def rasterize_polygon(grid, polygon, val, min_x, min_y, grid_w, grid_h):
    """Helper to mark grid cells"""
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
                grid[x][y] = val

def plan_road(edge_id):
    print("\n--- Phase 2: Strategic Bypass Planning ---")
    
    # 1. Network & Grid Setup
    net = sumolib.net.readNet(NET_FILE)
    bbox = net.getBBoxXY()
    if isinstance(bbox[0], (list, tuple)):
        (min_x, min_y), (max_x, max_y) = bbox
    else:
        min_x, min_y, max_x, max_y = bbox
        
    w_m, h_m = max_x - min_x, max_y - min_y
    grid_w, grid_h = int(w_m/GRID_RES)+1, int(h_m/GRID_RES)+1
    
    # Grid now stores COSTS, not just 0/1
    # Initialize with 1 (Empty Space Cost)
    grid = np.ones((grid_w, grid_h), dtype=np.int32) * COST_EMPTY
    
    # 2. Mark Obstacles (Buildings/Water) -> Infinite Cost
    print("   Marking buildings as OBSTACLES...")
    tree = ET.parse(POLY_FILE)
    root = tree.getroot()
    for poly in root.findall('poly'):
        ptype = poly.get('type', 'unknown')
        if ptype in ['building', 'water']:
            shape = poly.get('shape')
            coords = [tuple(map(float, p.split(','))) for p in shape.split()]
            if len(coords) >= 3:
                rasterize_polygon(grid, Polygon(coords), COST_OBSTACLE, min_x, min_y, grid_w, grid_h)

    # 3. Mark the Congested Road -> High Penalty Cost
    # We buffer the line to create a "zone of avoidance"
    print("   Marking current road as HIGH COST zone...")
    bad_edge = net.getEdge(edge_id)
    shape_coords = bad_edge.getShape() # List of (x,y)
    
    # Create a polygon buffer around the road (15 meters wide)
    road_line = LineString(shape_coords)
    road_buffer = road_line.buffer(20.0)  # Increased buffer to 20m to force wider bypass
    rasterize_polygon(grid, road_buffer, COST_CONGESTION, min_x, min_y, grid_w, grid_h)

    # 4. Determine Start/End Points (Extended)
    start_coord = bad_edge.getFromNode().getCoord()
    end_coord = bad_edge.getToNode().getCoord()
    
    raw_sx, raw_sy = int((start_coord[0]-min_x)/GRID_RES), int((start_coord[1]-min_y)/GRID_RES)
    raw_ex, raw_ey = int((end_coord[0]-min_x)/GRID_RES), int((end_coord[1]-min_y)/GRID_RES)

    start_pt = get_valid_node(grid, raw_sx, raw_sy, grid_w, grid_h)
    end_pt = get_valid_node(grid, raw_ex, raw_ey, grid_w, grid_h)
    
    if not start_pt or not end_pt:
        print("   Error: Could not find valid endpoints.")
        return

    # 5. Weighted A* Search
    print(f"   Searching for bypass path (avoiding cost {COST_CONGESTION})...")
    open_list = []
    closed_set = set()
    heapq.heappush(open_list, Node(start_pt[0], start_pt[1]))
    
    final_node = None
    visited = 0
    
    while open_list:
        curr = heapq.heappop(open_list)
        visited += 1
        
        if visited % 5000 == 0:
            print(f"   ... visited {visited} nodes ...")
        
        if abs(curr.x - end_pt[0]) <= 1 and abs(curr.y - end_pt[1]) <= 1:
            final_node = curr
            break
        
        if (curr.x, curr.y) in closed_set: continue
        closed_set.add((curr.x, curr.y))
        
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nx, ny = curr.x+dx, curr.y+dy
            
            if 0 <= nx < grid_w and 0 <= ny < grid_h:
                cell_cost = grid[nx][ny]
                
                # If obstacle, skip
                if cell_cost >= COST_OBSTACLE: continue
                if (nx, ny) in closed_set: continue
                
                # Movement cost (1.0 or 1.41)
                move_cost = 1.414 if dx!=0 and dy!=0 else 1.0
                
                # TOTAL COST = Steps + Cell Weight (Congestion Penalty)
                new_g = curr.g + move_cost + (cell_cost * 2.0) # Higher multiplier to enforce avoidance
                
                # Heuristic (Euclidean)
                # Reduced weight (1.2) to allow 'swinging wide' instead of beelining
                h = math.sqrt((nx-end_pt[0])**2 + (ny-end_pt[1])**2) * 1.2
                
                heapq.heappush(open_list, Node(nx, ny, curr, new_g, h))

    if final_node:
        print(f"   SUCCESS: Bypass found! (Length: {visited} nodes)")
        path_str = []
        c = final_node
        while c:
            wx = min_x + c.x*GRID_RES
            wy = min_y + c.y*GRID_RES
            path_str.append(f"{wx:.2f},{wy:.2f}")
            c = c.parent
        
        # Save Output
        with open(OUTPUT_PROPOSAL, 'w') as f:
            f.write('<additional>\n')
            # RED: The Congested Road
            f.write(f'    <poly id="congestion_red" color="255,0,0" lineWidth="8" layer="100" '
                    f'shape="{start_coord[0]},{start_coord[1]} {end_coord[0]},{end_coord[1]}"/>\n')
            # GREEN: The Bypass (Should now bow outwards)
            f.write(f'    <poly id="bypass_green" color="0,255,0" lineWidth="6" layer="101" '
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