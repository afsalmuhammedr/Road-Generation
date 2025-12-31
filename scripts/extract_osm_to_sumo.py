#!/usr/bin/env python3
"""
extract_osm_to_sumo.py

Extract buildings & water from OSM and generate SUMO poly.xml
aligned with the SUMO road network.
"""

import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union


# ---------------------------------------------------
# Read SUMO projection
# ---------------------------------------------------
def read_sumo_projection(net_xml):
    tree = ET.parse(net_xml)
    root = tree.getroot()
    loc = root.find("location")
    if loc is None or loc.get("projParameter") is None:
        raise RuntimeError("SUMO net.xml missing projection info")
    return loc.get("projParameter")


# ---------------------------------------------------
# Load OSM (OSMnx ≥ 2.0 compatible)
# ---------------------------------------------------
def load_osm(osm_path):
    if hasattr(ox, "geometries_from_xml"):
        gdf = ox.geometries_from_xml(osm_path)
    else:
        gdf = ox.features_from_xml(osm_path)

    if gdf.empty:
        raise RuntimeError("No OSM features found")
    return gdf


# ---------------------------------------------------
# Split features manually
# ---------------------------------------------------
def split_features(gdf):
    buildings = gdf[gdf.get("building").notna()].copy()
    roads = gdf[gdf.get("highway").notna()].copy()

    water_parts = []
    if "natural" in gdf.columns:
        water_parts.append(gdf[gdf["natural"] == "water"])
    if "waterway" in gdf.columns:
        water_parts.append(gdf[gdf["waterway"].notna()])

    water = gpd.GeoDataFrame(pd.concat(water_parts), crs=gdf.crs) if water_parts else gpd.GeoDataFrame(crs=gdf.crs)
    return buildings, water, roads


# ---------------------------------------------------
# Convert waterways to polygons
# ---------------------------------------------------
def water_to_polygons(water_m, buffer_m=5):
    polys = []
    for g in water_m.geometry:
        if isinstance(g, (Polygon, MultiPolygon)):
            polys.append(g)
        elif isinstance(g, LineString):
            polys.append(g.buffer(buffer_m))
    if not polys:
        return gpd.GeoDataFrame(crs=water_m.crs)
    return gpd.GeoDataFrame(geometry=[unary_union(polys)], crs=water_m.crs)


# ---------------------------------------------------
# Remove road overlap
# ---------------------------------------------------
def subtract_roads(polys, roads, buffer_m=2):
    if polys.empty or roads.empty:
        return polys

    road_union = unary_union(roads.geometry.buffer(buffer_m))
    cleaned = []

    for g in polys.geometry:
        diff = g.difference(road_union)
        if isinstance(diff, Polygon):
            cleaned.append(diff)
        elif isinstance(diff, MultiPolygon):
            cleaned.extend(diff.geoms)

    return gpd.GeoDataFrame(geometry=cleaned, crs=polys.crs)


# ---------------------------------------------------
# Polygon → SUMO shape
# ---------------------------------------------------
def polygon_to_shape(geom):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in geom.exterior.coords)


# ---------------------------------------------------
# Write poly.xml
# ---------------------------------------------------
def write_poly(out_file, buildings, water):
    from lxml import etree

    root = etree.Element("additional")

    def add(gdf, prefix, color):
        idx = 0
        for geom in gdf.geometry:
            if isinstance(geom, Polygon):
                geoms = [geom]
            elif isinstance(geom, MultiPolygon):
                geoms = geom.geoms
            else:
                continue

            for g in geoms:
                p = etree.SubElement(root, "poly")
                p.set("id", f"{prefix}_{idx}")
                p.set("color", color)
                p.set("layer", "1")
                p.set("shape", polygon_to_shape(g))
                idx += 1

    add(buildings, "building", "160,160,160")
    add(water, "water", "0,120,255")

    etree.ElementTree(root).write(
        out_file,
        pretty_print=True,
        xml_declaration=True,
        encoding="utf-8"
    )


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--osm", required=True)
    parser.add_argument("--net", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    osm = Path(args.osm)
    net = Path(args.net)

    if not osm.exists() or not net.exists():
        sys.exit("OSM or NET file not found")

    print("🔹 Reading SUMO projection...")
    sumo_proj = read_sumo_projection(net)

    print("🔹 Loading OSM...")
    gdf = load_osm(osm)

    print("🔹 Extracting features...")
    buildings, water, roads = split_features(gdf)

    print("🔹 Projecting to SUMO CRS...")
    buildings_m = buildings.to_crs(sumo_proj)
    water_m = water.to_crs(sumo_proj)
    roads_m = roads.to_crs(sumo_proj)

    print("🔹 Processing water...")
    water_poly = water_to_polygons(water_m)

    print("🔹 Removing road overlap...")
    buildings_clean = subtract_roads(buildings_m, roads_m)
    water_clean = subtract_roads(water_poly, roads_m)

    print("🔹 Writing poly.xml...")
    write_poly(args.out, buildings_clean, water_clean)

    print("✅ DONE")
    print(f"Open with:\n  sumo-gui -n {args.net} -a {args.out}")


if __name__ == "__main__":
    main()
