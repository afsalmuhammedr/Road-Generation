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

# --- IMPROVED HELPER FUNCTIONS ---

def find_upstream_downstream(net, edge_id, depth=2):
    """
    Moves 'depth' edges backward from the start and forward from the end
    to find better anchor points for the bypass.
    """
    center_edge = net.getEdge(edge_id)
    
    # 1. Find Upstream Start (Backwards)
    current_node = center_edge.getFromNode()
    for _ in range(depth):
        incoming = list(current_node.getIncoming())
        if not incoming: break
        # Heuristic: Pick the longest incoming edge to avoid tiny connector edges
        best_prev = max(incoming, key=lambda e: e.getLength())
        current_node = best_prev.getFromNode()
    start_node = current_node

    # 2. Find Downstream End (Forwards)
    current_node = center_edge.getToNode()
    for _ in range(depth):
        outgoing = list(current_node.getOutgoing())
        if not outgoing: break
        best_next = max(outgoing, key=lambda e: e.getLength())
        current_node = best_next.getToNode()
    end_node = current_node

    return start_node, end_node

def rasterize_all_roads(grid, net, min_x, min_y, grid_w, grid_h):
    """Marks ALL existing roads as expensive to prevent parallel overlaps."""
    print("   Marking existing road network as high cost...")
    for edge in net.getEdges():
        # Skip internal edges usually
        if edge.getFunction() == "internal": continue
        
        shape = edge.getShape()
        line = LineString(shape)
        # Buffer actual road width (approx 10m)
        poly = line.buffer(5.0) 
        rasterize_polygon(grid, poly, 50, min_x, min_y, grid_w, grid_h) # Cost 50 (High-ish)

# --- IMPROVED PLAN ROAD FUNCTION ---

def plan_road(edge_id):
    print("\n--- Phase 2: Strategic Bypass Planning (Improved) ---")
    
    net = sumolib.net.readNet(NET_FILE)
    bbox = net.getBBoxXY()
    if isinstance(bbox[0], (list, tuple)):
        (min_x, min_y), (max_x, max_y) = bbox
    else:
        min_x, min_y, max_x, max_y = bbox
        
    w_m, h_m = max_x - min_x, max_y - min_y
    grid_w, grid_h = int(w_m/GRID_RES)+1, int(h_m/GRID_RES)+1
    
    # Init Grid with Empty Cost (1)
    grid = np.ones((grid_w, grid_h), dtype=np.int32) * COST_EMPTY
    
    # 1. Mark Static Obstacles (Buildings) -> Infinite Cost
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

    # 2. Mark ALL existing roads (Prevents parallel lanes)
    rasterize_all_roads(grid, net, min_x, min_y, grid_w, grid_h)

    # 3. Create the "Repulsion Field" around the Congestion
    # We use concentric buffers to force the path to swing wide.
    # The closer to the bad road, the higher the cost.
    print("   Creating congestion repulsion field...")
    bad_edge = net.getEdge(edge_id)
    road_line = LineString(bad_edge.getShape())
    
    # Layer 1: Immediate Zone (Do not enter) - 20m
    rasterize_polygon(grid, road_line.buffer(20.0), 200, min_x, min_y, grid_w, grid_h)
    # Layer 2: Nearby Zone (High Drag) - 50m
    rasterize_polygon(grid, road_line.buffer(50.0), 40, min_x, min_y, grid_w, grid_h)
    # Layer 3: Influence Zone (Slight Drag) - 100m
    rasterize_polygon(grid, road_line.buffer(100.0), 5, min_x, min_y, grid_w, grid_h)

    # 4. Determine Dynamic Start/End Points
    # Instead of the bad edge's nodes, we search upstream/downstream
    start_node, end_node = find_upstream_downstream(net, edge_id, depth=3) # Look 3 edges away
    
    start_coord = start_node.getCoord()
    end_coord = end_node.getCoord()

    print(f"   Anchor Points Moved: Start {bad_edge.getFromNode().getID()} -> {start_node.getID()}")

    raw_sx, raw_sy = int((start_coord[0]-min_x)/GRID_RES), int((start_coord[1]-min_y)/GRID_RES)
    raw_ex, raw_ey = int((end_coord[0]-min_x)/GRID_RES), int((end_coord[1]-min_y)/GRID_RES)

    start_pt = get_valid_node(grid, raw_sx, raw_sy, grid_w, grid_h)
    end_pt = get_valid_node(grid, raw_ex, raw_ey, grid_w, grid_h)

    # 5. A* Search (Unchanged logic, just running on new grid)
    print(f"   Searching for wide bypass...")
    open_list = []
    closed_set = set()
    heapq.heappush(open_list, Node(start_pt[0], start_pt[1]))
    
    final_node = None
    visited = 0
    
    while open_list:
        curr = heapq.heappop(open_list)
        visited += 1
        
        # Optimization: Heuristic Inflation (1.5) makes it greedier/faster
        # but in a high-cost field, it encourages taking the longer "cheap" route 
        # rather than the short "expensive" route.
        
        if abs(curr.x - end_pt[0]) <= 1 and abs(curr.y - end_pt[1]) <= 1:
            final_node = curr
            break
        
        if (curr.x, curr.y) in closed_set: continue
        closed_set.add((curr.x, curr.y))
        
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nx, ny = curr.x+dx, curr.y+dy
            
            if 0 <= nx < grid_w and 0 <= ny < grid_h:
                cell_cost = grid[nx][ny]
                if cell_cost >= COST_OBSTACLE: continue
                if (nx, ny) in closed_set: continue
                
                move_cost = 1.414 if dx!=0 and dy!=0 else 1.0
                
                # KEY: The cell_cost is now variable (1, 5, 40, 50, 200)
                new_g = curr.g + move_cost * cell_cost 
                
                h = math.sqrt((nx-end_pt[0])**2 + (ny-end_pt[1])**2)
                heapq.heappush(open_list, Node(nx, ny, curr, new_g, h))

    if final_node:
        print(f"   SUCCESS: Path found.")
        path_str = []
        c = final_node
        # Simple smoothing: skip every 2nd node to reduce jaggedness in output
        step_counter = 0
        while c:
            if step_counter % 2 == 0: 
                wx = min_x + c.x*GRID_RES
                wy = min_y + c.y*GRID_RES
                path_str.append(f"{wx:.2f},{wy:.2f}")
            step_counter += 1
            c = c.parent
        
        # Write XML (Standard)
        with open(OUTPUT_PROPOSAL, 'w') as f:
            f.write('<additional>\n')
            f.write(f'    <poly id="congestion_red" color="255,0,0" lineWidth="8" layer="100" shape="{bad_edge.getShape()[0][0]},{bad_edge.getShape()[0][1]} {bad_edge.getShape()[-1][0]},{bad_edge.getShape()[-1][1]}"/>\n')
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