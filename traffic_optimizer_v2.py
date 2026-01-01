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
NET_FILE = "area.net.xml"
POLY_FILE = "area.poly.xml"
OUTPUT_PROPOSAL = "proposal_layer.xml"

TRIPS_FILE = "trips.xml"
ROUTE_FILE = "routes.xml"
EDGE_DATA_FILE = "edge_data.xml"

# OPTIMIZATION: 5.0 meters is a good balance for speed/accuracy
GRID_RES = 5.0  

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

# --- ROBUST A* IMPLEMENTATION ---

class Node:
    def __init__(self, x, y, parent=None, g=0, h=0):
        self.x, self.y = x, y
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h
    def __lt__(self, other): return self.f < other.f

def get_valid_node(grid, x, y, w, h):
    """
    Safety Mechanism: If the start/end point is inside a building,
    spiral out to find the nearest FREE cell.
    """
    if 0 <= x < w and 0 <= y < h and grid[x][y] == 0:
        return x, y
    
    # Search radius of 10 cells (approx 50m)
    print(f"   ! Point ({x},{y}) is blocked/out of bounds. Snapping to nearest valid road...")
    for radius in range(1, 10):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if grid[nx][ny] == 0:
                        return nx, ny
    return None

def get_extended_endpoint(net, start_node, is_upstream=True, distance_threshold=200):
    """
    Traverses the graph upstream or downstream to find a point further away from the congestion.
    """
    current_node = start_node
    distance_covered = 0
    
    # Safety breakout
    for _ in range(10): 
        if distance_covered >= distance_threshold:
            break
            
        edges = current_node.getIncoming() if is_upstream else current_node.getOutgoing()
        if not edges:
            break
            
        # Pick the 'main' continuation (simplistic approach: longest edge or straightest)
        # Here we just take the first one for simplicity, or the one that isn't a turnaround
        next_edge = list(edges)[0]
        distance_covered += next_edge.getLength()
        current_node = next_edge.getFromNode() if is_upstream else next_edge.getToNode()
        
    return current_node.getCoord()

def apply_repulsion_field(grid, net, congested_edge_id, grid_res, min_x, min_y, penalty_cost=20, radius_m=40):
    """
    Adds a high cost to grid cells near the congested edge to discourage the pathfinder
    from just hugging the existing road.
    """
    print(f"   Applying REPULSION FIELD (Penalty: +{penalty_cost}) around edge {congested_edge_id}...")
    edge = net.getEdge(congested_edge_id)
    shape = edge.getShape() # List of (x, y)
    
    grid_w, grid_h = grid.shape
    radius_cells = int(radius_m / grid_res)
    
    # Rasterize the line and apply dilation
    # Simplified approach: Bounding box of line segments + radius
    for i in range(len(shape) - 1):
        x1, y1 = shape[i]
        x2, y2 = shape[i+1]
        
        # Grid range for this segment
        gx1 = int((min(x1, x2) - min_x) / grid_res) - radius_cells
        gx2 = int((max(x1, x2) - min_x) / grid_res) + radius_cells
        gy1 = int((min(y1, y2) - min_y) / grid_res) - radius_cells
        gy2 = int((max(y1, y2) - min_y) / grid_res) + radius_cells
        
        # Clamp to grid
        gx1, gx2 = max(0, gx1), min(grid_w, gx2)
        gy1, gy2 = max(0, gy1), min(grid_h, gy2)
        
        # Apply penalty
        for x in range(gx1, gx2):
            for y in range(gy1, gy2):
                # Distance check could be added for circular field, but box is faster/sufficient
                if grid[x][y] != 1: # Don't overwrite walls
                    # We use a separate cost grid or just encoded high values? 
                    # Let's use negative values for 'penalty' in the same grid -> Wait, grid is int8 0/1.
                    # Let's verify grid dtype. It's int8. 
                    # We can use values > 1 for penalties.
                    grid[x][y] = max(grid[x][y], 10) # 10 = High Cost Zone
    return grid

def plan_road(congested_edge_id):
    print("\n--- Phase 2: Traffic-Aware Road Planning ---")
    
    net = sumolib.net.readNet(NET_FILE)
    edge = net.getEdge(congested_edge_id)
    
    # 1. EXTENDED ENDPOINTS LOGIC
    # Find start/end points that are ~250m away from the bottleneck to create a true bypass
    print("   Searching for extended connection points...")
    start_coord = get_extended_endpoint(net, edge.getFromNode(), is_upstream=True, distance_threshold=250)
    end_coord = get_extended_endpoint(net, edge.getToNode(), is_upstream=False, distance_threshold=250)
    print(f"   New Start: {start_coord}")
    print(f"   New End:   {end_coord}")

    # 2. Setup Grid
    bbox = net.getBBoxXY()
    if isinstance(bbox[0], (list, tuple)):
        (min_x, min_y), (max_x, max_y) = bbox
    else:
        min_x, min_y, max_x, max_y = bbox
        
    w_m, h_m = max_x - min_x, max_y - min_y
    grid_w, grid_h = int(w_m/GRID_RES)+1, int(h_m/GRID_RES)+1
    
    print(f"   Map Size: {int(w_m)}x{int(h_m)}m | Grid: {grid_w}x{grid_h}")
    # Use int16 to allow for penalty values > 127 if needed
    grid = np.zeros((grid_w, grid_h), dtype=np.int16) 
    
    # 3. Rasterize Obstacles
    print(f"   Mapping obstacles from {POLY_FILE}...")
    tree = ET.parse(POLY_FILE)
    root = tree.getroot()
    
    count = 0
    for poly in root.findall('poly'):
        ptype = poly.get('type', 'unknown')
        if ptype not in ['building', 'water']: continue 
            
        shape = poly.get('shape')
        coords = [tuple(map(float, p.split(','))) for p in shape.split()]
        if len(coords) < 3: continue
        
        polygon = Polygon(coords)
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
                    grid[x][y] = 1 # 1 = Absolute Wall
                    count += 1
    
    print(f"   Grid ready. {count} cells blocked.")

    # 4. APPLY REPULSION FIELD
    grid = apply_repulsion_field(grid, net, congested_edge_id, GRID_RES, min_x, min_y, penalty_cost=20, radius_m=40)

    # 5. A* Setup
    raw_sx, raw_sy = int((start_coord[0]-min_x)/GRID_RES), int((start_coord[1]-min_y)/GRID_RES)
    raw_ex, raw_ey = int((end_coord[0]-min_x)/GRID_RES), int((end_coord[1]-min_y)/GRID_RES)
    
    start_pt = get_valid_node(grid, raw_sx, raw_sy, grid_w, grid_h)
    end_pt = get_valid_node(grid, raw_ex, raw_ey, grid_w, grid_h)
    
    if not start_pt or not end_pt:
        print("   CRITICAL ERROR: Could not find valid start/end points.")
        return

    # 6. Running Weighted A* with Repulsion
    print(f"   Starting Search...")
    
    open_list = []
    closed_set = set()
    heapq.heappush(open_list, Node(start_pt[0], start_pt[1]))
    
    final_node = None
    nodes_visited = 0
    
    while open_list:
        curr = heapq.heappop(open_list)
        nodes_visited += 1
        
        if nodes_visited % 10000 == 0:
            print(f"   ... visited {nodes_visited} nodes ...")
        
        if abs(curr.x - end_pt[0]) <= 1 and abs(curr.y - end_pt[1]) <= 1:
            final_node = curr
            break
        
        if (curr.x, curr.y) in closed_set: continue
        closed_set.add((curr.x, curr.y))
        
        # 8-direction movement
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nx, ny = curr.x+dx, curr.y+dy
            if 0 <= nx < grid_w and 0 <= ny < grid_h:
                cell_val = grid[nx][ny]
                
                # Check if Wall (1)
                if cell_val == 1: continue
                if (nx, ny) in closed_set: continue
                
                # Base Cost + Repulsion Penalty
                step_cost = 1.414 if dx!=0 and dy!=0 else 1.0
                
                # If cell_val > 1, it's a penalty zone (e.g., 10)
                # We add that penalty to the step cost to discourage walking here
                if cell_val > 1:
                    step_cost += int(cell_val)
                
                h = math.sqrt((nx-end_pt[0])**2 + (ny-end_pt[1])**2)
                f_score = (curr.g + step_cost) + h
                
                heapq.heappush(open_list, Node(nx, ny, curr, curr.g+step_cost, h))

    if final_node:
        print(f"   SUCCESS: Path found! Length: {nodes_visited} steps computed.")
        path_str = []
        c = final_node
        while c:
            wx = min_x + c.x*GRID_RES
            wy = min_y + c.y*GRID_RES
            path_str.append(f"{wx:.2f},{wy:.2f}")
            c = c.parent
        
        with open(OUTPUT_PROPOSAL, 'w') as f:
            f.write('<additional>\n')
            f.write(f'    <poly id="congestion_red" color="255,0,0" lineWidth="8" layer="100" shape="{start_coord[0]},{start_coord[1]} {end_coord[0]},{end_coord[1]}"/>\n')
            f.write(f'    <poly id="bypass_green" color="0,255,0" lineWidth="6" layer="101" shape="{" ".join(path_str)}"/>\n')
            f.write('</additional>\n')
        print(f"   Proposal saved to {OUTPUT_PROPOSAL}")
    else:
        print("   FAILED: No path found.")

def main():
    sumo = check_sumo_env()
    # run_simulation_pipeline(sumo) # Skip sim if already ran to save time during dev
    
    worst_edge_id = find_worst_congestion()
    if worst_edge_id:
        plan_road(worst_edge_id)
        print(f"\nVisualize: sumo-gui -n {NET_FILE} -a {POLY_FILE},{OUTPUT_PROPOSAL}")

if __name__ == "__main__":
    main()