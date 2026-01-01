import os
import sys
import subprocess
import xml.etree.ElementTree as ET
import sumolib
import numpy as np
import heapq
import math
from shapely.geometry import Polygon, Point

# --- Configuration ---
# INPUTS (Read-Only)
NET_FILE = "area.net.xml"
POLY_FILE = "area.poly.xml"  # Your colored map with buildings/water/compounds

# OUTPUT (New Layer)
OUTPUT_PROPOSAL = "proposal_layer.xml" # Defines the new road and congestion

# SIMULATION CONFIG
TRIPS_FILE = "trips.xml"
ROUTE_FILE = "routes.xml"
EDGE_DATA_FILE = "edge_data.xml"
GRID_RES = 4.0  # Grid precision in meters

def check_sumo_env():
    if 'SUMO_HOME' not in os.environ:
        print("Error: SUMO_HOME is not set.")
        sys.exit(1)
    return os.environ['SUMO_HOME']

def run_simulation_pipeline(sumo_home):
    """Generates traffic and runs simulation to find congestion."""
    print("--- Phase 1: Traffic Simulation ---")
    
    # 1. Generate random trips
    print("1. Generating random traffic...")
    subprocess.run([
        "python", os.path.join(sumo_home, "tools", "randomTrips.py"),
        "-n", NET_FILE, "-o", TRIPS_FILE, "-r", ROUTE_FILE, 
        "-e", "600", "-p", "1.5", "--validate"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. Run Simulation (Headless)
    print("2. Simulating traffic flow...")
    sumo_bin = os.path.join(sumo_home, "bin", "sumo")
    subprocess.run([
        sumo_bin, "-n", NET_FILE, "-r", ROUTE_FILE,
        "--edgedata-output", EDGE_DATA_FILE,
        "--begin", "0", "--end", "800", "--no-step-log"
    ], check=True)

def find_worst_congestion():
    """Reads the simulation output to find the slowest road."""
    print("3. analyzing congestion data...")
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
        print(f"   >> CRITICAL CONGESTION DETECTED on Edge: {worst_edge} (Time Loss: {max_time_loss:.2f}s)")
    return worst_edge

# --- A* Path Finding ---

class Node:
    def __init__(self, x, y, parent=None):
        self.x, self.y, self.parent = x, y, parent
        self.g = self.h = self.f = 0
    def __lt__(self, other): return self.f < other.f

def plan_road(start_coord, end_coord):
    print("\n--- Phase 2: A* Road Planning ---")
    print(f"Planning bypass from {start_coord} to {end_coord}...")
    
    # 1. Setup Grid
    net = sumolib.net.readNet(NET_FILE)
    # sumolib/net.getBBoxXY may return either 4 values or two coordinate pairs
    bbox = net.getBBoxXY()
    if isinstance(bbox, (list, tuple)) and len(bbox) == 2 and isinstance(bbox[0], (list, tuple)):
        (min_x, min_y), (max_x, max_y) = bbox
    else:
        min_x, min_y, max_x, max_y = bbox
    w_m, h_m = max_x - min_x, max_y - min_y
    grid_w, grid_h = int(w_m/GRID_RES)+1, int(h_m/GRID_RES)+1
    
    grid = np.zeros((grid_w, grid_h), dtype=int)
    
    # 2. Load Obstacles from poly file (READ ONLY)
    print(f"Reading obstacles from {POLY_FILE}...")
    tree = ET.parse(POLY_FILE)
    root = tree.getroot()
    
    blocked_count = 0
    for poly in root.findall('poly'):
        ptype = poly.get('type', 'unknown')
        
        # INTELLIGENT FILTERING:
        # We BLOCK 'building' (Grey) and 'water' (Blue).
        # We ALLOW 'industrial', 'education', 'compound' (Pink/Yellow).
        if ptype not in ['building', 'water']:
            continue
            
        shape = poly.get('shape')
        coords = [tuple(map(float, p.split(','))) for p in shape.split()]
        if len(coords) < 3: continue
        
        polygon = Polygon(coords)
        blocked_count += 1
        
        # Rasterize this polygon onto the grid
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
                    grid[x][y] = 1 # Blocked

    print(f"Mapped {blocked_count} critical obstacles (buildings/water).")
    
    # 3. A* Search
    start_n = (int((start_coord[0]-min_x)/GRID_RES), int((start_coord[1]-min_y)/GRID_RES))
    end_n = (int((end_coord[0]-min_x)/GRID_RES), int((end_coord[1]-min_y)/GRID_RES))
    
    open_list = []
    closed_set = set()
    heapq.heappush(open_list, Node(*start_n))
    
    final_node = None
    while open_list:
        curr = heapq.heappop(open_list)
        if abs(curr.x - end_n[0]) <= 1 and abs(curr.y - end_n[1]) <= 1:
            final_node = curr
            break
        
        closed_set.add((curr.x, curr.y))
        
        # 8-direction movement
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nx, ny = curr.x+dx, curr.y+dy
            if 0<=nx<grid_w and 0<=ny<grid_h:
                if (nx, ny) in closed_set or grid[nx][ny] == 1: continue
                
                cost = 1.414 if dx!=0 and dy!=0 else 1.0
                neighbor = Node(nx, ny, curr)
                neighbor.g = curr.g + cost
                neighbor.h = math.sqrt((nx-end_n[0])**2 + (ny-end_n[1])**2)
                neighbor.f = neighbor.g + neighbor.h
                heapq.heappush(open_list, neighbor)

    # 4. Save Result to NEW File
    if final_node:
        path = []
        c = final_node
        while c:
            path.append(f"{min_x + c.x*GRID_RES:.2f},{min_y + c.y*GRID_RES:.2f}")
            c = c.parent
        
        with open(OUTPUT_PROPOSAL, 'w') as f:
            f.write('<additional>\n')
            
            # 1. Highlight the PROBLEM (Congested Road) in RED
            f.write(f'    <poly id="congestion_zone" color="255,0,0" lineWidth="6" layer="100" '
                    f'shape="{start_coord[0]},{start_coord[1]} {end_coord[0]},{end_coord[1]}"/>\n')
            
            # 2. Highlight the SOLUTION (New Road) in BRIGHT GREEN
            f.write(f'    <poly id="proposed_bypass" color="0,255,0" lineWidth="5" layer="101" '
                    f'shape="{" ".join(path)}"/>\n')
            
            f.write('</additional>\n')
        print(f"\nSUCCESS: Proposal saved to '{OUTPUT_PROPOSAL}'")
    else:
        print("Failed: No path found.")

def main():
    sumo = check_sumo_env()
    run_simulation_pipeline(sumo)
    
    worst_edge_id = find_worst_congestion()
    if worst_edge_id:
        net = sumolib.net.readNet(NET_FILE)
        edge = net.getEdge(worst_edge_id)
        # Plan path from Start of edge to End of edge
        plan_road(edge.getFromNode().getCoord(), edge.getToNode().getCoord())
        
        print("\nTo visualize the difference, run:")
        print(f"sumo-gui -n {NET_FILE} -a {POLY_FILE},{OUTPUT_PROPOSAL}")

if __name__ == "__main__":
    main()