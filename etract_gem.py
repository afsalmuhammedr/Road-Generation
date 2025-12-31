import os
import subprocess
import xml.etree.ElementTree as ET
import sumolib
from shapely.geometry import Polygon
import geopandas as gpd

# --- Configuration ---
INPUT_OSM = "map.osm"        # Replace with your actual OSM file name
OUTPUT_NET = "area.net.xml"
OUTPUT_POLY = "area.poly.xml"

def generate_network(osm_file, net_file):
    """
    Step 1: Convert OSM to SUMO Network (net.xml).
    This establishes the coordinate system (projection) for the simulation.
    """
    print(f"Generating {net_file} from {osm_file}...")
    
    # We use subprocess to call SUMO's native netconvert tool
    # This ensures the road topology is correct before we add buildings
    cmd = [
        "netconvert",
        "--osm-files", osm_file,
        "-o", net_file,
        "--no-turnarounds",       # Cleaner for visualization
        "--geometry.remove",      # Optimization
        "--roundabouts.guess",    # Better junction interpretation
        "--ramps.guess"
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        print("Network generation successful.")
    except subprocess.CalledProcessError as e:
        print(f"Error running netconvert: {e}")
        exit(1)
    except FileNotFoundError:
        print("Error: 'netconvert' not found. Ensure SUMO bin/ is in your PATH.")
        exit(1)

def extract_features_from_osm(osm_file):
    """
    Step 2: Parse raw OSM XML to find Buildings and Water.
    We manually parse XML to ensure we get local data without API calls.
    """
    print("Parsing OSM file for features...")
    tree = ET.parse(osm_file)
    root = tree.getroot()

    nodes = {}  # Store id -> (lon, lat)
    features = [] # Store {type, geometry_nodes}

    # 1. Cache all nodes
    for node in root.findall('node'):
        nid = node.get('id')
        lon = float(node.get('lon'))
        lat = float(node.get('lat'))
        nodes[nid] = (lon, lat)

    # 2. Process ways (potential polygons)
    for way in root.findall('way'):
        way_nodes = []
        tags = {}
        
        # Get tags
        for tag in way.findall('tag'):
            tags[tag.get('k')] = tag.get('v')
        
        # Get geometry node IDs
        for nd in way.findall('nd'):
            ref = nd.get('ref')
            if ref in nodes:
                way_nodes.append(nodes[ref])

        # Requirements: Must be a closed loop (polygon)
        if len(way_nodes) < 3 or way_nodes[0] != way_nodes[-1]:
            continue

        # Classification Logic
        feat_type = None
        
        # Check for Building
        if 'building' in tags:
            feat_type = 'building'
        
        # Check for Water
        elif tags.get('natural') == 'water' or 'waterway' in tags:
            feat_type = 'water'

        if feat_type:
            features.append({
                'type': feat_type,
                'raw_nodes': way_nodes  # List of (lon, lat)
            })

    print(f"Extracted {len(features)} features.")
    return features

def transform_and_write_poly(features, net_file, poly_file):
    """
    Step 3 & 4: Convert Lat/Lon to SUMO X/Y and write .poly.xml.
    This ensures buildings don't overlap differently than the roads.
    """
    print("Transforming coordinates and writing output...")
    
    # Load the network to access the projection
    net = sumolib.net.readNet(net_file)
    
    with open(poly_file, 'w') as f:
        f.write('<additional>\n')
        
        for idx, feat in enumerate(features):
            sumo_coords = []
            
            # Convert every node from Geo (Lon, Lat) to SUMO (X, Y)
            for lon, lat in feat['raw_nodes']:
                x, y = net.convertLonLat2XY(lon, lat)
                sumo_coords.append(f"{x:.2f},{y:.2f}")
            
            shape_str = " ".join(sumo_coords)
            
            # Visualization Attributes
            if feat['type'] == 'building':
                # Grey/Brown for buildings, higher layer to stand out
                color = "160,160,160" 
                layer = "1"
                fill = "true"
            else:
                # Blue for water, lower layer
                color = "50,50,200"
                layer = "0"
                fill = "true"

            # Write the polygon element
            f.write(f'    <poly id="{feat["type"]}_{idx}" '
                    f'type="{feat["type"]}" '
                    f'color="{color}" '
                    f'fill="{fill}" '
                    f'layer="{layer}" '
                    f'shape="{shape_str}"/>\n')

        f.write('</additional>\n')
    print(f"Successfully wrote {poly_file}")

def main():
    if not os.path.exists(INPUT_OSM):
        print(f"Error: Input file '{INPUT_OSM}' not found.")
        return

    # 1. Create the Road Network
    generate_network(INPUT_OSM, OUTPUT_NET)
    
    # 2. Extract Buildings and Water
    features = extract_features_from_osm(INPUT_OSM)
    
    # 3. Transform and Save Visuals
    transform_and_write_poly(features, OUTPUT_NET, OUTPUT_POLY)
    
    print("\nProcess Complete.")
    print("Run the simulation with:")
    print(f"sumo-gui -n {OUTPUT_NET} -a {OUTPUT_POLY}")

if __name__ == "__main__":
    main()