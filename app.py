import os
import sys
import json
import traceback
import threading
import uuid
import time
import io
from flask import Flask, request, jsonify, Response, send_from_directory
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

# ---------------------------------------------------------------------------
# Job tracking system
# ---------------------------------------------------------------------------
# Each job: {status, logs[], result, error}
jobs = {}
jobs_lock = threading.Lock()


def create_job():
    """Create a new job entry and return its ID."""
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {
            "status": "running",   # running | complete | failed
            "logs": [],
            "result": None,
            "error": None,
        }
    return job_id


def job_log(job_id, message, level="info"):
    """Append a log entry to a job. Thread-safe."""
    timestamp = time.strftime("%H:%M:%S")
    entry = {"timestamp": timestamp, "level": level, "message": message}
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["logs"].append(entry)
    # Also print to server console for debugging
    print(f"  [{timestamp}] [{level}] {message}")


def job_complete(job_id, result):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["status"] = "complete"
            jobs[job_id]["result"] = result


def job_fail(job_id, error_msg):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = error_msg


# ---------------------------------------------------------------------------
# Pipeline steps (updated to log to job)
# ---------------------------------------------------------------------------

def download_osm(north, south, east, west, output_path, job_id):
    """Download OSM data for the given bounding box using Overpass API with fallbacks."""
    import requests

    job_log(job_id, f"Fetching OSM data: N={north:.4f}, S={south:.4f}, E={east:.4f}, W={west:.4f}")

    bbox_str = f"{south},{west},{north},{east}"
    bbox_map = f"{west},{south},{east},{north}"

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

    overpass_endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]

    for i, endpoint in enumerate(overpass_endpoints):
        try:
            job_log(job_id, f"Trying Overpass endpoint ({i+1}/{len(overpass_endpoints)}): {endpoint.split('/')[2]}...")
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
                job_log(job_id, f"✓ OSM data saved ({size_kb:.1f} KB)", "success")
                return True
            else:
                job_log(job_id, f"Endpoint returned status {response.status_code}, trying next...")

        except requests.exceptions.Timeout:
            job_log(job_id, f"Endpoint timed out, trying next...")
        except requests.exceptions.ConnectionError:
            job_log(job_id, f"Connection error, trying next...")
        except Exception as e:
            job_log(job_id, f"Error: {e}, trying next...")

        time.sleep(1)

    # Fallback
    map_endpoints = [
        "https://overpass-api.de/api/map",
        "https://overpass.kumi.systems/api/map",
    ]

    for i, endpoint in enumerate(map_endpoints):
        try:
            job_log(job_id, f"Fallback: trying /api/map endpoint ({i+1}/{len(map_endpoints)})...")
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
                job_log(job_id, f"✓ OSM data saved ({size_kb:.1f} KB) via fallback", "success")
                return True
            else:
                job_log(job_id, f"Fallback returned status {response.status_code}, trying next...")

        except requests.exceptions.Timeout:
            job_log(job_id, f"Fallback timed out, trying next...")
        except Exception as e:
            job_log(job_id, f"Fallback error: {e}, trying next...")

        time.sleep(1)

    raise Exception(
        "Failed to download OSM data from all Overpass API endpoints. "
        "The servers may be overloaded. Please try again in a few minutes."
    )


def run_extract_env(area_path, job_id):
    """Run extract_env.py in the given area directory."""
    job_log(job_id, "Running network extraction (netconvert + poly)...")

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
        raise Exception(f"extract_env.py failed: {result.stderr}")

    job_log(job_id, "✓ Network extraction complete", "success")
    return True


def write_road_gen_config(area_path, params):
    """Write a config JSON that road-gen.py can optionally read."""
    config = {
        "GRID_RES": params.get("gridRes", 5.0),
        "COST_OBSTACLE": params.get("costObstacle", 9999),
        "COST_EXISTING_ROAD": params.get("costExistingRoad", 80),
        "COST_CONGESTION_CORE": params.get("costCongestionCore", 500),
        "COST_CONGESTION_NEAR": params.get("costCongestionNear", 150),
        "COST_CONGESTION_FAR": params.get("costCongestionFar", 30),
        "COST_EMPTY": params.get("costEmpty", 1),

        # RL + GA Parameters
        "RL_EPISODES": params.get("rlEpisodes", 1500),
        "RL_DOWNSAMPLE": params.get("rlDownsample", 3),
        "GA_POPULATION": params.get("gaPopulation", 50),
        "GA_GENERATIONS": params.get("gaGenerations", 60),
        "GA_WAYPOINTS": params.get("gaWaypoints", 10),
        "GA_DOWNSAMPLE": params.get("gaDownsample", 2),
    }
    config_path = os.path.join(area_path, "road_gen_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return config_path


def run_road_generation(area_path, job_id):
    """Run road-gen.py with real-time stdout streaming to job logs."""
    job_log(job_id, "Starting road generation (RL + GA optimization)...")

    script_path = os.path.join(BASE_DIR, "road-gen.py")
    env = os.environ.copy()

    # Use Popen for real-time output streaming
    process = subprocess.Popen(
        [sys.executable, "-u", script_path],  # -u for unbuffered output
        cwd=area_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,  # Line buffered
    )

    # Read stdout line-by-line in real time
    for line in process.stdout:
        line = line.rstrip()
        if line:
            # Detect RL/GA progress lines for better level tagging
            if "Episode" in line or "Gen " in line:
                job_log(job_id, line.strip(), "info")
            elif "SUCCESS" in line or "Complete" in line:
                job_log(job_id, line.strip(), "success")
            elif "WARNING" in line or "Error" in line or "FAILED" in line:
                job_log(job_id, line.strip(), "warning")
            else:
                job_log(job_id, line.strip(), "info")

    process.wait()

    if process.returncode != 0:
        stderr = process.stderr.read() if process.stderr else ""
        raise Exception(f"road-gen.py failed (exit {process.returncode}): {stderr}")

    job_log(job_id, "✓ Road generation complete", "success")
    return True


def parse_proposal_xml(area_path, job_id):
    """Parse proposal_layer.xml and convert SUMO coordinates to lat/lon."""
    import sumolib

    job_log(job_id, "Parsing results and converting to geo-coordinates...")

    proposal_path = os.path.join(area_path, "proposal_layer.xml")
    net_path = os.path.join(area_path, "area.net.xml")

    if not os.path.exists(proposal_path):
        job_log(job_id, "Warning: proposal_layer.xml not found.", "warning")
        return None

    if not os.path.exists(net_path):
        job_log(job_id, "Warning: area.net.xml not found.", "warning")
        return None

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

        sumo_coords = []
        for pair in shape_str.strip().split():
            parts = pair.split(',')
            if len(parts) == 2:
                try:
                    sumo_coords.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue

        geo_coords = []
        for x, y in sumo_coords:
            lon, lat = net.convertXY2LonLat(x, y)
            geo_coords.append([lat, lon])

        try:
            r, g, b = [int(c) for c in color.split(',')]
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
        except:
            hex_color = "#888888"

        # Add display metadata based on overlay type
        overlay_meta = {
            "coordinates": geo_coords,
            "color": hex_color,
        }
        if "congestion" in poly_id or "red" in poly_id:
            overlay_meta["label"] = "Congested Road"
            overlay_meta["weight"] = 6
            overlay_meta["opacity"] = 0.9
        elif "bypass" in poly_id or "green" in poly_id:
            overlay_meta["label"] = "Proposed Bypass"
            overlay_meta["weight"] = 6
            overlay_meta["opacity"] = 0.95
            overlay_meta["dashArray"] = "12 6"

        results[poly_id] = overlay_meta

    job_log(job_id, f"✓ Found {len(results)} overlay(s) in results", "success")
    return results


# ---------------------------------------------------------------------------
# Background pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline_async(job_id, project_id, area_id, bounds, params):
    """Runs the full generation pipeline in a background thread."""
    try:
        safe_area_id = "".join([c for c in area_id if c.isalnum() or c in ('-', '_')]).strip()
        if not safe_area_id:
            safe_area_id = "area_default"

        project_path = os.path.join(PROJECTS_DIR, project_id)
        area_path = os.path.join(project_path, safe_area_id)
        os.makedirs(area_path, exist_ok=True)

        north = float(bounds['north'])
        south = float(bounds['south'])
        east = float(bounds['east'])
        west = float(bounds['west'])

        # Step 1: Download OSM data
        osm_path = os.path.join(area_path, "map.osm")
        download_osm(north, south, east, west, osm_path, job_id)

        # Step 2: Run extract_env.py
        run_extract_env(area_path, job_id)

        # Step 3: Write config and run road-gen.py
        if params:
            write_road_gen_config(area_path, params)
        run_road_generation(area_path, job_id)

        # Step 4: Parse results
        overlays = parse_proposal_xml(area_path, job_id)

        # List generated files
        generated_files = []
        if os.path.exists(area_path):
            for f in os.listdir(area_path):
                fpath = os.path.join(area_path, f)
                if os.path.isfile(fpath):
                    generated_files.append({
                        "name": f,
                        "size": os.path.getsize(fpath)
                    })

        job_log(job_id, f"Generated {len(generated_files)} files", "info")
        job_log(job_id, "Road generation completed successfully!", "success")

        job_complete(job_id, {
            "status": "success",
            "message": "Road network generated successfully.",
            "outputDir": area_path,
            "files": generated_files,
            "overlays": overlays
        })

    except Exception as e:
        traceback.print_exc()
        job_log(job_id, f"Error: {str(e)}", "error")
        job_fail(job_id, str(e))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "road-generation-backend"})


@app.route('/api/generate', methods=['POST'])
def generate_network():
    """Start road generation in background, return job ID immediately."""
    try:
        data = request.json
        project_id = data.get('projectId')
        area_id = data.get('areaId')
        bounds = data.get('bounds')
        params = data.get('params', {})

        if not all([project_id, area_id, bounds]):
            return jsonify({"error": "Missing required fields: projectId, areaId, bounds"}), 400

        # Create job and start pipeline in background thread
        job_id = create_job()
        job_log(job_id, f"Road generation started for area: {area_id}")

        thread = threading.Thread(
            target=run_pipeline_async,
            args=(job_id, project_id, area_id, bounds, params),
            daemon=True
        )
        thread.start()

        return jsonify({
            "jobId": job_id,
            "message": "Road generation started. Connect to SSE stream for progress."
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate/<job_id>/stream')
def stream_job(job_id):
    """SSE endpoint — streams job logs and final result in real time."""
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({"error": "Job not found"}), 404

    def event_stream():
        last_index = 0

        while True:
            with jobs_lock:
                job = jobs.get(job_id)
                if not job:
                    break

                # Send any new log entries
                current_logs = job["logs"]
                new_logs = current_logs[last_index:]
                status = job["status"]
                result = job["result"]
                error = job["error"]

            for entry in new_logs:
                data = json.dumps({"type": "log", **entry})
                yield f"data: {data}\n\n"
                last_index += 1

            # Check if job finished
            if status == "complete":
                data = json.dumps({"type": "complete", "result": result})
                yield f"data: {data}\n\n"
                break
            elif status == "failed":
                data = json.dumps({"type": "error", "error": error})
                yield f"data: {data}\n\n"
                break

            time.sleep(0.5)  # Poll interval

    return Response(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Access-Control-Allow-Origin': '*',
        }
    )


@app.route('/api/generate/<job_id>/status')
def job_status(job_id):
    """Fallback status polling endpoint."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        return jsonify({
            "status": job["status"],
            "logCount": len(job["logs"]),
            "result": job["result"],
            "error": job["error"],
        })


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


@app.route('/api/projects/<project_id>/areas/<area_id>/results', methods=['GET'])
def get_area_results(project_id, area_id):
    """Get past results and files for a specific area."""
    try:
        project_path = os.path.join(PROJECTS_DIR, project_id)
        area_path = os.path.join(project_path, area_id)
        
        if not os.path.exists(area_path):
            return jsonify({"error": "Area not found or not processed yet"}), 404
            
        files = []
        for f in os.listdir(area_path):
            fpath = os.path.join(area_path, f)
            if os.path.isfile(fpath):
                files.append({
                    "name": f,
                    "size": os.path.getsize(fpath)
                })

        overlays = parse_proposal_xml(area_path, "api_request")
        
        return jsonify({
            "files": files,
            "overlays": overlays
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/projects/<project_id>/areas/<area_id>/files/<path:filename>', methods=['GET'])
def download_file(project_id, area_id, filename):
    project_path = os.path.join(PROJECTS_DIR, project_id)
    area_path = os.path.join(project_path, area_id)
    if not os.path.exists(area_path):
        return jsonify({"error": "Area not found"}), 404
    return send_from_directory(area_path, filename, as_attachment=True)


if __name__ == '__main__':
    print("=" * 60)
    print("  Road Generation Backend Server")
    print(f"  Projects dir: {PROJECTS_DIR}")
    print(f"  SUMO_HOME: {os.environ.get('SUMO_HOME', 'NOT SET')}")
    print("=" * 60)
    app.run(debug=True, port=6001, threaded=True)
