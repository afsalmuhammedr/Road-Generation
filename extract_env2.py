import os
import subprocess
import xml.etree.ElementTree as ET
import sumolib

# --- Configuration ---
INPUT_OSM = "map.osm"
OUTPUT_NET = "area.net.xml"
OUTPUT_POLY = "area.poly.xml"

# --- OSM Color Palette (R,G,B) ---
# These approximate the standard OSM map rendering colors
COLORS = {
    # Water
    'water':       "170,211,223",  # Light Blue
    
    # Buildings
    'building':    "200,200,200",  # Standard Grey
    
    # Landuse / Compounds
    'industrial':  "235,219,233",  # Pinkish Purple (e.g., KSRTC Works)
    'commercial':  "242,218,218",  # Reddish Pink
    'education':   "255,255,228",  # Light Yellow/Beige (e.g., Colleges)
    'residential': "220,220,220",  # Light Grey
    'park':        "200,250,200",  # Light Green
    'forest':      "173,209,158",  # Green
    
    # Fallback
    'default':     "230,230,230"
}

def generate_network(osm_file, net_file):
    print(f"Generating {net_file} from {osm_file}...")
    cmd = [
        "netconvert",
        "--osm-files", osm_file,
        "-o", net_file,
        "--no-turnarounds",
        "--geometry.remove",
        "--roundabouts.guess",
        "--ramps.guess"
    ]
    # Suppress output for cleaner console
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_feature_style(tags):
    """
    Determines type, color, and layer based on OSM tags.
    """
    # 1. Buildings (Top Layer)
    if 'building' in tags:
        return 'building', COLORS['building'], "2"
    
    # 2. Water (Bottom Layer)
    if tags.get('natural') == 'water' or 'waterway' in tags:
        return 'water', COLORS['water'], "0"
    
    # 3. Compounds / Landuse (Background Layer -1)
    layer = "-1"
    
    # Check Landuse tags
    landuse = tags.get('landuse')
    if landuse in ['industrial', 'construction']:
        return 'industrial', COLORS['industrial'], layer
    if landuse in ['commercial', 'retail']:
        return 'commercial', COLORS['commercial'], layer
    if landuse in ['residential']:
        return 'residential', COLORS['residential'], layer
    if landuse in ['forest', 'grass']:
        return 'forest', COLORS['forest'], layer
        
    # Check Amenity tags (Colleges, Hospitals often marked here)
    amenity = tags.get('amenity')
    if amenity in ['college', 'school', 'university', 'kindergarten']:
        return 'education', COLORS['education'], layer
    if amenity in ['parking']:
        return 'commercial', COLORS['default'], layer
    if amenity in ['bus_station']:
        return 'industrial', COLORS['industrial'], layer

    # Leisure tags
    leisure = tags.get('leisure')
    if leisure in ['park', 'garden', 'pitch']:
        return 'park', COLORS['park'], layer

    return None, None, None

def extract_features_from_osm(osm_file):
    print("Parsing OSM file for detailed features...")
    tree = ET.parse(osm_file)
    root = tree.getroot()

    nodes = {}
    features = []

    # Cache Nodes
    for node in root.findall('node'):
        nodes[node.get('id')] = (float(node.get('lon')), float(node.get('lat')))

    # Process Ways
    for way in root.findall('way'):
        way_nodes = []
        tags = {tag.get('k'): tag.get('v') for tag in way.findall('tag')}
        
        for nd in way.findall('nd'):
            ref = nd.get('ref')
            if ref in nodes:
                way_nodes.append(nodes[ref])

        if len(way_nodes) < 3 or way_nodes[0] != way_nodes[-1]:
            continue

        # Get style based on specific tags
        f_type, f_color, f_layer = get_feature_style(tags)

        if f_type:
            features.append({
                'type': f_type,
                'color': f_color,
                'layer': f_layer,
                'raw_nodes': way_nodes
            })

    print(f"Extracted {len(features)} features.")
    return features

def transform_and_write_poly(features, net_file, poly_file):
    print("Transforming coordinates and writing output...")
    net = sumolib.net.readNet(net_file)
    
    with open(poly_file, 'w') as f:
        f.write('<additional>\n')
        
        for idx, feat in enumerate(features):
            sumo_coords = []
            for lon, lat in feat['raw_nodes']:
                x, y = net.convertLonLat2XY(lon, lat)
                sumo_coords.append(f"{x:.2f},{y:.2f}")
            
            shape_str = " ".join(sumo_coords)
            
            f.write(f'    <poly id="{feat["type"]}_{idx}" '
                    f'type="{feat["type"]}" '
                    f'color="{feat["color"]}" '
                    f'fill="true" '
                    f'layer="{feat["layer"]}" '
                    f'shape="{shape_str}"/>\n')

        f.write('</additional>\n')
    print(f"Successfully wrote {poly_file}")

def main():
    if not os.path.exists(INPUT_OSM):
        print(f"Error: {INPUT_OSM} not found.")
        return

    generate_network(INPUT_OSM, OUTPUT_NET)
    features = extract_features_from_osm(INPUT_OSM)
    transform_and_write_poly(features, OUTPUT_NET, OUTPUT_POLY)
    
    print("\nProcessing complete.")
    print(f"Run: sumo-gui -n {OUTPUT_NET} -a {OUTPUT_POLY}")

if __name__ == "__main__":
    main()