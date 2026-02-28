import os
import sys
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import xml.etree.ElementTree as ET

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(BASE_DIR, 'projects')

# Ensure projects directory exists
os.makedirs(PROJECTS_DIR, exist_ok=True)


def download_osm(north, south, east, west, output_path):
    """Download OSM data for the given bounding box using Overpass API with fallbacks."""
    import requests
    import time
    
    print(f"  [1/4] Fetching OSM data: N={north:.4f}, S={south:.4f}, E={east:.4f}, W={west:.4f}")
    
    bbox_str = f"{south},{west},{north},{east}"  # Overpass QL uses S,W,N,E
    bbox_map = f"{west},{south},{east},{north}"  # /api/map uses W,S,E,N
    
    # Overpass QL query: get all data in the bounding box (roads, buildings, etc.)
    overpass_query = f"""
    [out:xml][timeout:90];
    (
      way["highway"]({bbox_str});
      way["building"]({bbox_str});
      way["landuse"]({bbox_str});
      way["natural"]({bbox_str});
      way["waterway"]({bbox_str});
      relation["boundary"]({bbox_str});
    );
    (._;>;);
    out body;
    """
    
    # List of Overpass API endpoints to try (primary + mirrors)
    overpass_endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]
    
    # Strategy 1: Try Overpass QL interpreter (fetches only relevant data, much faster)
    for i, endpoint in enumerate(overpass_endpoints):
        try:
            print(f"  [1/4] Trying Overpass QL endpoint ({i+1}/{len(overpass_endpoints)}): {endpoint.split('/')[2]}...")
            response = requests.post(
                endpoint,
                data={"data": overpass_query},
                timeout=90,
                headers={"User-Agent": "RoadGenTool/1.0"}
            )
            
            if response.status_code == 200 and len(response.content) > 100:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                size_kb = len(response.content) / 1024
                print(f"  [1/4] OSM data saved ({size_kb:.1f} KB) via Overpass QL")
                return True
            else:
                print(f"  [1/4] Endpoint returned status {response.status_code} (body: {len(response.content)} bytes), trying next...")
                
        except requests.exceptions.Timeout:
            print(f"  [1/4] Endpoint timed out, trying next...")
        except requests.exceptions.ConnectionError:
            print(f"  [1/4] Connection error, trying next...")
        except Exception as e:
            print(f"  [1/4] Error: {e}, trying next...")
        
        time.sleep(1)  # Brief pause before trying next endpoint
    
    # Strategy 2: Fall back to /api/map endpoint (downloads ALL data, larger but simpler)
    map_endpoints = [
        "https://overpass-api.de/api/map",
        "https://overpass.kumi.systems/api/map",
    ]
    
    for i, endpoint in enumerate(map_endpoints):
        try:
            print(f"  [1/4] Fallback: trying /api/map endpoint ({i+1}/{len(map_endpoints)})...")
            response = requests.get(
                endpoint,
                params={"bbox": bbox_map},
                timeout=90,
                headers={"User-Agent": "RoadGenTool/1.0"}
            )
            
            if response.status_code == 200 and len(response.content) > 100:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                size_kb = len(response.content) / 1024
                print(f"  [1/4] OSM data saved ({size_kb:.1f} KB) via /api/map fallback")
                return True
            else:
                print(f"  [1/4] Fallback returned status {response.status_code}, trying next...")
                
        except requests.exceptions.Timeout:
            print(f"  [1/4] Fallback timed out, trying next...")
        except Exception as e:
            print(f"  [1/4] Fallback error: {e}, trying next...")
        
        time.sleep(1)
    
    raise Exception(
        "Failed to download OSM data from all Overpass API endpoints. "
        "The servers may be overloaded. Please try again in a few minutes."
    )



def run_extract_env(area_path):
    """Run extract_env.py in the given area directory."""
    print(f"  [2/4] Running network extraction (netconvert + poly)...")
    
    script_path = os.path.join(BASE_DIR, "extract_env.py")
    env = os.environ.copy()
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=area_path,
        capture_output=True,
        text=True,
        env=env
    )
    
    if result.returncode != 0:
        print(f"  [2/4] STDERR: {result.stderr}")
        raise Exception(f"extract_env.py failed: {result.stderr}")
    
    print(f"  [2/4] Network extraction complete.")
    return True


def write_road_gen_config(area_path, params):
    """Write a config JSON that road-gen.py can optionally read."""
    config = {
        "GRID_RES": params.get("gridRes", 5.0),
        "COST_OBSTACLE": params.get("costObstacle", 9999),
        "COST_EXISTING_ROAD": params.get("costExistingRoad", 60),
        "COST_CONGESTION_CORE": params.get("costCongestionCore", 200),
        "COST_CONGESTION_NEAR": params.get("costCongestionNear", 40),
        "COST_CONGESTION_FAR": params.get("costCongestionFar", 5),
        "COST_EMPTY": params.get("costEmpty", 1),
    }
    config_path = os.path.join(area_path, "road_gen_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return config_path


def run_road_generation(area_path):
    """Run road-gen.py in the given area directory."""
    print(f"  [3/4] Running road generation algorithm...")
    
    script_path = os.path.join(BASE_DIR, "road-gen.py")
    env = os.environ.copy()
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=area_path,
        capture_output=True,
        text=True,
        env=env
    )
    
    if result.returncode != 0:
        print(f"  [3/4] STDERR: {result.stderr}")
        raise Exception(f"road-gen.py failed: {result.stderr}")
    
    print(f"  [3/4] Road generation complete.")
    print(f"  [3/4] STDOUT: {result.stdout}")
    return True


def parse_proposal_xml(area_path):
    """Parse proposal_layer.xml and convert SUMO coordinates to lat/lon."""
    import sumolib
    
    print(f"  [4/4] Parsing results and converting to geo-coordinates...")
    
    proposal_path = os.path.join(area_path, "proposal_layer.xml")
    net_path = os.path.join(area_path, "area.net.xml")
    
    if not os.path.exists(proposal_path):
        print(f"  [4/4] Warning: proposal_layer.xml not found.")
        return None
    
    if not os.path.exists(net_path):
        print(f"  [4/4] Warning: area.net.xml not found for coordinate conversion.")
        return None
    
    # Load the network for coordinate conversion
    net = sumolib.net.readNet(net_path)
    
    tree = ET.parse(proposal_path)
    root = tree.getroot()
    
    results = {}
    
    for poly in root.findall('poly'):
        poly_id = poly.get('id', 'unknown')
        shape_str = poly.get('shape', '')
        color = poly.get('color', '128,128,128')
        
        if not shape_str:
            continue
        
        # Parse SUMO shape: "x1,y1 x2,y2 x3,y3 ..."
        sumo_coords = []
        for pair in shape_str.strip().split():
            parts = pair.split(',')
            if len(parts) == 2:
                try:
                    sumo_coords.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        
        # Convert SUMO XY -> Lon/Lat
        geo_coords = []
        for x, y in sumo_coords:
            lon, lat = net.convertXY2LonLat(x, y)
            geo_coords.append([lat, lon])  # [lat, lng] for Leaflet
        
        # Parse color "R,G,B" to hex
        try:
            r, g, b = [int(c) for c in color.split(',')]
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
        except:
            hex_color = "#888888"
        
        results[poly_id] = {
            "coordinates": geo_coords,
            "color": hex_color
        }
    
    print(f"  [4/4] Found {len(results)} overlay(s) in results.")
    return results


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "road-generation-backend"})


@app.route('/api/generate', methods=['POST'])
def generate_network():
    try:
        data = request.json
        project_id = data.get('projectId')
        area_id = data.get('areaId')
        bounds = data.get('bounds')
        params = data.get('params', {})
        
        if not all([project_id, area_id, bounds]):
            return jsonify({"error": "Missing required fields: projectId, areaId, bounds"}), 400

        # Sanitize area_id for filesystem
        safe_area_id = "".join([c for c in area_id if c.isalnum() or c in ('-', '_')]).strip()
        if not safe_area_id:
            safe_area_id = "area_default"

        # Create project/area directory structure
        project_path = os.path.join(PROJECTS_DIR, project_id)
        area_path = os.path.join(project_path, safe_area_id)
        os.makedirs(area_path, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Road Generation Request")
        print(f"  Project: {project_id}")
        print(f"  Area: {safe_area_id}")
        print(f"  Bounds: {bounds}")
        print(f"  Params: {params}")
        print(f"{'='*60}")
        
        north = float(bounds['north'])
        south = float(bounds['south'])
        east = float(bounds['east'])
        west = float(bounds['west'])
        
        # Step 1: Download OSM data
        osm_path = os.path.join(area_path, "map.osm")
        download_osm(north, south, east, west, osm_path)
        
        # Step 2: Run extract_env.py (generates area.net.xml & area.poly.xml)
        run_extract_env(area_path)
        
        # Step 3: Write config and run road-gen.py
        if params:
            write_road_gen_config(area_path, params)
        run_road_generation(area_path)
        
        # Step 4: Parse results and convert to geo-coordinates
        overlays = parse_proposal_xml(area_path)
        
        # List all generated files
        generated_files = []
        if os.path.exists(area_path):
            for f in os.listdir(area_path):
                fpath = os.path.join(area_path, f)
                if os.path.isfile(fpath):
                    generated_files.append({
                        "name": f,
                        "size": os.path.getsize(fpath)
                    })
        
        print(f"\n  Generated files: {[f['name'] for f in generated_files]}")
        print(f"  Output directory: {area_path}")
        print(f"{'='*60}\n")
        
        return jsonify({
            "status": "success",
            "message": "Road network generated successfully.",
            "outputDir": area_path,
            "files": generated_files,
            "overlays": overlays  # Contains geo-coordinates for map visualization
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/projects/<project_id>/areas', methods=['GET'])
def get_project_areas(project_id):
    """List areas for a project that have been processed."""
    project_path = os.path.join(PROJECTS_DIR, project_id)
    if not os.path.exists(project_path):
        return jsonify({"areas": []})
    
    areas = []
    for area_name in os.listdir(project_path):
        area_path = os.path.join(project_path, area_name)
        if os.path.isdir(area_path):
            files = os.listdir(area_path)
            areas.append({
                "id": area_name,
                "files": files,
                "hasProposal": "proposal_layer.xml" in files
            })
    
    return jsonify({"areas": areas})


if __name__ == '__main__':
    print("=" * 60)
    print("  Road Generation Backend Server")
    print(f"  Projects dir: {PROJECTS_DIR}")
    print(f"  SUMO_HOME: {os.environ.get('SUMO_HOME', 'NOT SET')}")
    print("=" * 60)
    app.run(debug=True, port=6000)
