import os
import sys
import subprocess
import xml.etree.ElementTree as ET
import sumolib
import numpy as np
import math
import json
from shapely.geometry import Polygon, Point, LineString, MultiLineString
from shapely.ops import linemerge
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
COST_EXISTING_ROAD = int(_config.get("COST_EXISTING_ROAD", 80))
COST_CONGESTION_CORE = int(_config.get("COST_CONGESTION_CORE", 800))
COST_CONGESTION_NEAR = int(_config.get("COST_CONGESTION_NEAR", 300))
COST_CONGESTION_FAR = int(_config.get("COST_CONGESTION_FAR", 50))
COST_EMPTY = int(_config.get("COST_EMPTY", 1))

# HYBRID RL + GA CONFIGURATION
RL_EPISODES = int(_config.get("RL_EPISODES", 1500))
RL_DOWNSAMPLE = int(_config.get("RL_DOWNSAMPLE", 3))
GA_POPULATION = int(_config.get("GA_POPULATION", 50))
GA_GENERATIONS = int(_config.get("GA_GENERATIONS", 60))
GA_WAYPOINTS = int(_config.get("GA_WAYPOINTS", 10))
GA_DOWNSAMPLE = int(_config.get("GA_DOWNSAMPLE", 2))


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


# =========================================================================
# CORRIDOR-AWARE CONGESTION DETECTION (MAIN-ROAD-AWARE)
# =========================================================================

def _get_road_identity(net, edge_id):
    """Get the road name and highway type for an edge."""
    try:
        edge = net.getEdge(edge_id)
        name = edge.getName() or ""
        etype = edge.getType() or ""
        return name, etype
    except:
        return "", ""


def _order_corridor_edges(net, corridor_edge_ids):
    """
    Topologically sort corridor edges by walking from one end to the other
    along the chain of connected edges. This produces a smooth, contiguous
    line instead of random-order edge segments.
    """
    if len(corridor_edge_ids) <= 1:
        return list(corridor_edge_ids)
    
    corridor_set = set(corridor_edge_ids)
    
    # Build adjacency: edge -> set of corridor neighbor edges
    adj = {eid: set() for eid in corridor_edge_ids}
    for eid in corridor_edge_ids:
        try:
            edge = net.getEdge(eid)
        except:
            continue
        for node in [edge.getFromNode(), edge.getToNode()]:
            for neighbor in list(node.getIncoming()) + list(node.getOutgoing()):
                nid = neighbor.getID()
                if nid != eid and nid in corridor_set:
                    adj[eid].add(nid)
    
    # Find an endpoint edge (degree 1 in the corridor graph, or fewest neighbors)
    start_edge = min(adj, key=lambda e: len(adj[e]))
    
    # Walk the chain
    ordered = [start_edge]
    visited = {start_edge}
    current = start_edge
    
    while True:
        next_edges = [n for n in adj[current] if n not in visited]
        if not next_edges:
            break
        # Pick the neighbor with the best adjacency continuation
        current = next_edges[0]
        ordered.append(current)
        visited.add(current)
    
    return ordered


def find_congested_corridor(net):
    """
    Detects the entire congested corridor along the MAIN ROAD.
    
    Strategy:
    1. Find the worst congested edge
    2. Identify the road name and highway type of that edge
    3. BFS outward, but STRONGLY prefer edges on the same road
       Side streets only included if extremely congested (>75% of worst)
    4. Return edges in topological (ordered) sequence
    
    Returns: (worst_edge_id, [ordered list of corridor edge IDs])
    """
    print("3. Analyzing congestion data...")
    tree = ET.parse(EDGE_DATA_FILE)
    root = tree.getroot()
    
    # Build congestion map: edge_id -> max time loss
    congestion_map = {}
    for interval in root.findall('interval'):
        for edge in interval.findall('edge'):
            eid = edge.get('id')
            loss = float(edge.get('timeLoss', 0))
            if eid not in congestion_map or loss > congestion_map[eid]:
                congestion_map[eid] = loss
    
    if not congestion_map:
        print("   No congestion data found.")
        return None, []
    
    # Find worst edge
    worst_edge = max(congestion_map, key=congestion_map.get)
    max_loss = congestion_map[worst_edge]
    print(f"   >> CRITICAL BOTTLENECK: Edge {worst_edge} (Lag: {max_loss:.2f}s)")
    
    # Identify the MAIN ROAD identity from the worst edge
    main_name, main_type = _get_road_identity(net, worst_edge)
    print(f"   >> Main road: name=\"{main_name}\", type=\"{main_type}\"")
    
    # Thresholds: different for same-road vs side-street edges
    same_road_threshold = max_loss * 0.20   # Lenient for same road
    side_street_threshold = max_loss * 0.75  # Strict for side streets
    
    # BFS from worst edge — main-road-aware
    corridor_edges = set()
    visited = set()
    queue = [worst_edge]
    
    while queue:
        eid = queue.pop(0)
        if eid in visited:
            continue
        visited.add(eid)
        
        try:
            edge = net.getEdge(eid)
        except:
            continue
        
        # Check congestion level
        if eid not in congestion_map:
            continue
        edge_loss = congestion_map[eid]
        
        # Determine if this edge is on the same road as the main corridor
        edge_name, edge_type = _get_road_identity(net, eid)
        is_same_road = False
        if main_name and edge_name and main_name.lower() == edge_name.lower():
            is_same_road = True
        elif main_type and edge_type and main_type == edge_type and not main_name:
            # Fallback: match by highway type if no road name
            is_same_road = True
        
        # Apply appropriate threshold
        if eid == worst_edge:
            pass  # Always include the worst edge
        elif is_same_road:
            if edge_loss < same_road_threshold:
                continue  # Not congested enough on the main road
        else:
            if edge_loss < side_street_threshold:
                continue  # Side street not congested enough to include
        
        corridor_edges.add(eid)
        
        # Explore neighbors
        for node in [edge.getFromNode(), edge.getToNode()]:
            for neighbor in list(node.getIncoming()) + list(node.getOutgoing()):
                nid = neighbor.getID()
                if nid not in visited:
                    queue.append(nid)
    
    # Order corridor edges topologically for smooth line output
    corridor_list = _order_corridor_edges(net, corridor_edges)
    
    print(f"   >> Congested corridor: {len(corridor_list)} edges "
          f"(same-road threshold: {same_road_threshold:.1f}s, "
          f"side-street threshold: {side_street_threshold:.1f}s)")
    return worst_edge, corridor_list, main_name, main_type


# =========================================================================
# ANCHOR SEARCH: Walk along the SAME ROAD past the corridor
# =========================================================================

def _edge_midpoint(edge):
    """Get the midpoint coordinate of an edge."""
    shape = edge.getShape()
    mx = sum(p[0] for p in shape) / len(shape)
    my = sum(p[1] for p in shape) / len(shape)
    return (mx, my)

def _dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def find_corridor_anchors(net, corridor_edge_ids, main_name="", main_type="", extra_distance=400):
    """
    Finds anchor nodes by tracing ALONG the road past the corridor endpoints.
    
    Strategy:
    1. Find the two extreme nodes of the corridor (start/end of the chain)
    2. From each extreme, walk ALONG the road (following longest edges) 
       for 'extra_distance' meters beyond the corridor
    3. These nodes are where traffic can divert onto the bypass
    
    This ensures anchors are on the same road, upstream and downstream.
    """
    if not corridor_edge_ids:
        return None, None
    
    corridor_set = set(corridor_edge_ids)
    
    # Collect all corridor nodes
    corridor_nodes = set()
    for eid in corridor_edge_ids:
        try:
            edge = net.getEdge(eid)
            corridor_nodes.add(edge.getFromNode().getID())
            corridor_nodes.add(edge.getToNode().getID())
        except:
            continue
    
    # Find the two BOUNDARY nodes of the corridor:
    # boundary = nodes where at least one connected edge is NOT in the corridor
    boundary_nodes = []
    for nid in corridor_nodes:
        node = net.getNode(nid)
        all_edges = list(node.getIncoming()) + list(node.getOutgoing())
        non_corridor = [e for e in all_edges if e.getID() not in corridor_set]
        if non_corridor:
            boundary_nodes.append(node)
    
    if len(boundary_nodes) < 2:
        # Fallback: use first and last edge nodes
        print("   WARNING: Not enough boundary nodes, using corridor edge endpoints.")
        first = net.getEdge(corridor_edge_ids[0])
        last = net.getEdge(corridor_edge_ids[-1])
        boundary_nodes = [first.getFromNode(), last.getToNode()]
    
    # Pick the two nodes farthest apart as the corridor start/end
    max_dist = -1
    start_boundary = boundary_nodes[0]
    end_boundary = boundary_nodes[-1]
    for i in range(len(boundary_nodes)):
        for j in range(i+1, len(boundary_nodes)):
            d = _dist(boundary_nodes[i].getCoord(), boundary_nodes[j].getCoord())
            if d > max_dist:
                max_dist = d
                start_boundary = boundary_nodes[i]
                end_boundary = boundary_nodes[j]
    
    print(f"   Corridor endpoints: {start_boundary.getID()} <-> {end_boundary.getID()} ({max_dist:.0f}m)")
    
    # Now walk ALONG the road past each boundary for extra_distance meters
    def _walk_past(node, away_from_coord, min_dist):
        """
        Walk along the road from 'node', heading away from the corridor,
        until we've traveled at least min_dist meters.
        """
        current = node
        traveled = 0
        visited_nodes = set(corridor_nodes)  # Don't re-enter corridor
        visited_nodes.add(current.getID())
        
        for step in range(15):  # max 15 edges
            if traveled >= min_dist:
                break
            
            # Get all edges leaving this node (both directions)
            candidates = []
            for edge in list(current.getIncoming()) + list(current.getOutgoing()):
                if edge.getID() in corridor_set:
                    continue  # Skip corridor edges
                other_node = edge.getToNode() if edge.getFromNode().getID() == current.getID() else edge.getFromNode()
                if other_node.getID() in visited_nodes:
                    continue
                candidates.append((edge, other_node))
            
            if not candidates:
                break
            
            # Pick the edge that takes us farthest from the corridor center
            # AND is the longest (prefer major roads)
            best_edge = None
            best_score = -float('inf')
            for edge, other in candidates:
                dist_away = _dist(other.getCoord(), away_from_coord)
                
                # Bonus for staying on the same road
                edge_name, edge_type = _get_road_identity(net, edge.getID())
                road_bonus = 0
                if main_name and edge_name and main_name.lower() == edge_name.lower():
                    road_bonus = 5000  # Massive bonus for same road name
                elif main_type and edge_type and main_type == edge_type:
                    road_bonus = 1000  # Good bonus for same highway type
                
                score = edge.getLength() + dist_away * 0.3 + road_bonus
                if score > best_score:
                    best_score = score
                    best_edge = (edge, other)
            
            if best_edge is None:
                break
            
            edge, next_node = best_edge
            traveled += edge.getLength()
            visited_nodes.add(next_node.getID())
            current = next_node
        
        return current, traveled
    
    # Corridor center (to walk away from)
    corridor_center = (
        (start_boundary.getCoord()[0] + end_boundary.getCoord()[0]) / 2,
        (start_boundary.getCoord()[1] + end_boundary.getCoord()[1]) / 2
    )
    
    # Walk upstream from start boundary
    start_anchor, start_traveled = _walk_past(start_boundary, corridor_center, extra_distance)
    # Walk downstream from end boundary  
    end_anchor, end_traveled = _walk_past(end_boundary, corridor_center, extra_distance)
    
    start_d = _dist(start_anchor.getCoord(), corridor_center)
    end_d = _dist(end_anchor.getCoord(), corridor_center)
    
    print(f"   Upstream anchor: {start_anchor.getID()} ({start_traveled:.0f}m walked, {start_d:.0f}m from center)")
    print(f"   Downstream anchor: {end_anchor.getID()} ({end_traveled:.0f}m walked, {end_d:.0f}m from center)")
    
    return start_anchor, end_anchor


# =========================================================================
# GRID HELPERS
# =========================================================================

def get_valid_node(grid, x, y, w, h):
    """Finds nearest non-obstacle, non-high-cost node."""
    if 0 <= x < w and 0 <= y < h and grid[x][y] < COST_OBSTACLE:
        return x, y
    for radius in range(1, 20):
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
                if val > grid[x][y]:
                    grid[x][y] = val


# =========================================================================
# MAIN PLANNING
# =========================================================================

def plan_road(net, corridor_edge_ids, main_name, main_type):
    print("\n--- Phase 2: Strategic Bypass Planning ---")
    
    bbox = net.getBBoxXY()
    if isinstance(bbox[0], (list, tuple)):
        (min_x, min_y), (max_x, max_y) = bbox
    else:
        min_x, min_y, max_x, max_y = bbox
        
    w_m, h_m = max_x - min_x, max_y - min_y
    grid_w, grid_h = int(w_m/GRID_RES)+1, int(h_m/GRID_RES)+1
    
    # 1. Initialize Grid
    grid = np.ones((grid_w, grid_h), dtype=np.int32) * COST_EMPTY
    
    # 2. Build unified corridor geometry
    print("   Building corridor geometry...")
    corridor_lines = []
    for eid in corridor_edge_ids:
        try:
            edge = net.getEdge(eid)
            shape = edge.getShape()
            if len(shape) >= 2:
                corridor_lines.append(LineString(shape))
        except:
            continue
    
    if not corridor_lines:
        print("   ERROR: No valid corridor geometry.")
        return
    
    if len(corridor_lines) == 1:
        corridor_geom = corridor_lines[0]
    else:
        corridor_geom = linemerge(MultiLineString(corridor_lines))
    
    corridor_length = corridor_geom.length
    print(f"   Corridor: {len(corridor_edge_ids)} edges, ~{corridor_length:.0f}m total")
    
    # 3. Apply repulsion around the ENTIRE corridor
    # Scale buffers based on corridor length
    print("   Applying corridor repulsion...")
    scale = max(1.0, corridor_length / 200.0)
    
    # Larger buffers to push bypass further out
    far_buffer = max(300, min(1200, 300 * scale))
    near_buffer = max(150, min(800, 150 * scale))
    core_buffer = max(50, min(300, 60 * scale))
    
    print(f"   Buffers: core={core_buffer:.0f}m, near={near_buffer:.0f}m, far={far_buffer:.0f}m")
    
    rasterize_polygon(grid, corridor_geom.buffer(far_buffer), COST_CONGESTION_FAR, min_x, min_y, grid_w, grid_h)
    rasterize_polygon(grid, corridor_geom.buffer(near_buffer), COST_CONGESTION_NEAR, min_x, min_y, grid_w, grid_h)
    rasterize_polygon(grid, corridor_geom.buffer(core_buffer), COST_CONGESTION_CORE, min_x, min_y, grid_w, grid_h)

    # 4. Mark ALL existing roads
    print("   Masking existing road network...")
    for edge in net.getEdges():
        shape = edge.getShape()
        if len(shape) < 2: continue
        road_poly = LineString(shape).buffer(8.0)
        rasterize_polygon(grid, road_poly, COST_EXISTING_ROAD, min_x, min_y, grid_w, grid_h)

    # 5. Mark Buildings
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

    # 6. Find anchors by walking ALONG the road past the corridor
    extra_dist = max(300, corridor_length * 0.5)
    start_node, end_node = find_corridor_anchors(net, corridor_edge_ids, main_name, main_type, extra_distance=extra_dist)
    
    if not start_node or not end_node:
        print("   ERROR: Could not find bypass anchor points.")
        return
    
    start_coord = start_node.getCoord()
    end_coord = end_node.getCoord()
    anchor_dist = _dist(start_coord, end_coord)
    print(f"   Anchor separation: {anchor_dist:.0f}m")
    
    raw_sx = int((start_coord[0]-min_x)/GRID_RES)
    raw_sy = int((start_coord[1]-min_y)/GRID_RES)
    raw_ex = int((end_coord[0]-min_x)/GRID_RES)
    raw_ey = int((end_coord[1]-min_y)/GRID_RES)

    start_pt = get_valid_node(grid, raw_sx, raw_sy, grid_w, grid_h)
    end_pt = get_valid_node(grid, raw_ex, raw_ey, grid_w, grid_h)

    if not start_pt or not end_pt:
        print("Error: Could not find valid start/end points on grid.")
        return

    # 7. Hybrid RL + GA Search
    print(f"   Running Hybrid RL + GA Optimizer...")
    print(f"   Grid: {grid_w}x{grid_h}, Start: {start_pt}, End: {end_pt}")
    
    optimal_path = find_optimal_bypass(
        grid, start_pt, end_pt,
        obstacle_threshold=COST_OBSTACLE,
        rl_episodes=RL_EPISODES,
        rl_downsample=RL_DOWNSAMPLE,
        ga_population=GA_POPULATION,
        ga_generations=GA_GENERATIONS,
        ga_waypoints=GA_WAYPOINTS,
        ga_downsample=GA_DOWNSAMPLE
    )

    if optimal_path:
        print(f"   SUCCESS: Bypass found! Path points: {len(optimal_path)}")
        path_str = []
        for gx, gy in optimal_path:
            wx = min_x + gx * GRID_RES
            wy = min_y + gy * GRID_RES
            path_str.append(f"{wx:.2f},{wy:.2f}")
        
        # Save Output
        with open(OUTPUT_PROPOSAL, 'w') as f:
            f.write('<additional>\n')
            
            # Visualize the ENTIRE congested corridor (RED)
            all_corridor_pts = []
            for eid in corridor_edge_ids:
                try:
                    edge = net.getEdge(eid)
                    for pt in edge.getShape():
                        all_corridor_pts.append(pt)
                except:
                    continue
            
            if all_corridor_pts:
                congestion_shape_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in all_corridor_pts)
                f.write(f'    <poly id="congestion_red" color="255,0,0" lineWidth="12" layer="100" '
                        f'shape="{congestion_shape_str}"/>\n')
            
            # The bypass (GREEN)
            f.write(f'    <poly id="bypass_green" color="0,200,0" lineWidth="10" layer="101" '
                    f'shape="{" ".join(path_str)}"/>\n')
            f.write('</additional>\n')
        print(f"   Proposal saved to {OUTPUT_PROPOSAL}")
    else:
        print("   FAILED: No path found.")

def main():
    sumo = check_sumo_env()
    run_simulation_pipeline(sumo)
    
    net = sumolib.net.readNet(NET_FILE)
    res = find_congested_corridor(net)
    
    if res and res[0] and res[1]:
        worst_edge_id, corridor_edge_ids, main_name, main_type = res
        plan_road(net, corridor_edge_ids, main_name, main_type)
        print(f"\nVisualize: sumo-gui -n {NET_FILE} -a {POLY_FILE},{OUTPUT_PROPOSAL}")
    else:
        print("No congestion detected.")

if __name__ == "__main__":
    main()