#!/usr/bin/env python3
"""
Generate Deep Zoom overlays for annotations and a mock heatmap.

- Annotation overlay: rasterizes ASAP XML polygons into Deep Zoom tiles.
- Heatmap overlay: creates a deterministic pseudo-heatmap on a coarse grid
  (replace with model confidences later) and rasterizes to Deep Zoom tiles.

Example:
python scripts/generate_overlays.py \
  --slide data/camelyon17/training/center_0/patient_010_node_4.tif \
  --xml data/camelyon17/training/center_0/patient_010_node_4.xml \
  --output public/overlays/patient_010_node_4 \
  --tile-size 256 --min-level 8
"""

import argparse
import math
import random
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator


def write_dzi(width: int, height: int, tile_size: int, overlap: int, fmt: str, path: Path) -> None:
    """Write a DZI metadata file."""
    dzi = f"""<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" TileSize="{tile_size}" Overlap="{overlap}" Format="{fmt}">
    <Size Width="{width}" Height="{height}"/>
</Image>
"""
    path.write_text(dzi, encoding="utf-8")


def load_polygons(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    polys = []
    for ann in root.findall(".//Annotation"):
        coords = [
            (float(c.attrib["X"]), float(c.attrib["Y"]))
            for c in ann.findall(".//Coordinate")
        ]
        if len(coords) < 3:
            continue
        label = ann.attrib.get("PartOfGroup") or ann.attrib.get("Name") or "unknown"
        color = ann.attrib.get("Color")
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        polys.append(
            {
                "label": label,
                "coords": coords,
                "color": color,
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
            }
        )
    return polys


def compute_level_tiles_for_polys(polys, levels, tile_size, l0_downsamples, offset):
    """Map polygons to tile indices per level to avoid iterating empty tiles."""
    tiles_by_level = {lvl: set() for lvl in levels}
    for lvl in levels:
        down = l0_downsamples[lvl]
        for poly in polys:
            x0, y0, x1, y1 = poly["bbox"]
            # convert bbox to deepzoom level coords
            zx0 = (x0 - offset[0]) / down
            zy0 = (y0 - offset[1]) / down
            zx1 = (x1 - offset[0]) / down
            zy1 = (y1 - offset[1]) / down
            if zx1 < 0 or zy1 < 0:
                continue
            min_col = max(0, int(math.floor(zx0 / tile_size)))
            min_row = max(0, int(math.floor(zy0 / tile_size)))
            max_col = int(math.floor(zx1 / tile_size))
            max_row = int(math.floor(zy1 / tile_size))
            for col in range(min_col, max_col + 1):
                for row in range(min_row, max_row + 1):
                    tiles_by_level[lvl].add((col, row))
    return tiles_by_level


def draw_annotations_tile(polys, tile, lvl, col, row, tile_size, l0_downsamples, offset):
    draw = ImageDraw.Draw(tile, "RGBA")
    down = l0_downsamples[lvl]
    origin_x = col * tile_size
    origin_y = row * tile_size
    for poly in polys:
        # quick reject by bbox
        x0, y0, x1, y1 = poly["bbox"]
        zx0 = (x0 - offset[0]) / down
        zy0 = (y0 - offset[1]) / down
        zx1 = (x1 - offset[0]) / down
        zy1 = (y1 - offset[1]) / down
        if zx1 < origin_x or zy1 < origin_y:
            continue
        if zx0 > origin_x + tile.width or zy0 > origin_y + tile.height:
            continue
        coords = [
            ((x - offset[0]) / down - origin_x, (y - offset[1]) / down - origin_y)
            for x, y in poly["coords"]
        ]
        color = poly.get("color") or "#F4FA58"
        draw.polygon(coords, outline=color, fill=color + "55")


def draw_heatmap_tile(tile, lvl, col, row, tile_size, l0_downsamples, offset, cell_size, colormap):
    """Draw a simple deterministic heatmap on a coarse grid."""
    draw = ImageDraw.Draw(tile, "RGBA")
    down = l0_downsamples[lvl]
    origin_x = col * tile_size
    origin_y = row * tile_size

    # tile bounds in level-0 coords
    l0_min_x = origin_x * down + offset[0]
    l0_min_y = origin_y * down + offset[1]
    l0_max_x = (origin_x + tile.width) * down + offset[0]
    l0_max_y = (origin_y + tile.height) * down + offset[1]

    cell_min_x = int(math.floor(l0_min_x / cell_size))
    cell_min_y = int(math.floor(l0_min_y / cell_size))
    cell_max_x = int(math.floor((l0_max_x - 1) / cell_size))
    cell_max_y = int(math.floor((l0_max_y - 1) / cell_size))

    for cx in range(cell_min_x, cell_max_x + 1):
        for cy in range(cell_min_y, cell_max_y + 1):
            rng = random.Random((cx * 73856093) ^ (cy * 19349663))
            score = rng.random()
            fill = colormap(score)
            # cell bounds in level coords
            z_min_x = (cx * cell_size - offset[0]) / down - origin_x
            z_min_y = (cy * cell_size - offset[1]) / down - origin_y
            z_max_x = ((cx + 1) * cell_size - offset[0]) / down - origin_x
            z_max_y = ((cy + 1) * cell_size - offset[1]) / down - origin_y
            draw.rectangle([z_min_x, z_min_y, z_max_x, z_max_y], fill=fill, outline=None)


def simple_colormap(score: float):
    """Map 0..1 to RGBA (blue->green->yellow->red)."""
    score = max(0.0, min(1.0, score))
    if score < 0.33:
        # blue to green
        t = score / 0.33
        r, g, b = 0, int(255 * t), int(255 * (1 - t))
    elif score < 0.66:
        # green to yellow
        t = (score - 0.33) / 0.33
        r, g, b = int(255 * t), 255, 0
    else:
        # yellow to red
        t = (score - 0.66) / 0.34
        r, g, b = 255, int(255 * (1 - t)), 0
    return (r, g, b, int(180 * score + 40))


def main():
    parser = argparse.ArgumentParser(description="Generate Deep Zoom overlays (annotations + mock heatmap).")
    parser.add_argument("--slide", required=True, help="Path to WSI (.tif)")
    parser.add_argument("--xml", required=True, help="Path to ASAP XML annotations")
    parser.add_argument("--output", required=True, help="Output base dir for overlays")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size for overlays")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap for overlays")
    parser.add_argument("--min-level", type=int, default=None, help="Min Deep Zoom level to render (inclusive)")
    parser.add_argument("--max-level", type=int, default=None, help="Max Deep Zoom level to render (inclusive)")
    parser.add_argument("--heatmap-cell-size", type=int, default=1024, help="Heatmap grid cell size in level-0 pixels")
    args = parser.parse_args()

    slide_path = Path(args.slide)
    xml_path = Path(args.xml)
    out_base = Path(args.output)
    out_base.mkdir(parents=True, exist_ok=True)

    slide = OpenSlide(str(slide_path))
    dz = DeepZoomGenerator(
        slide,
        tile_size=args.tile_size,
        overlap=args.overlap,
        limit_bounds=True,
    )

    levels = list(range(dz.level_count))
    if args.min_level is not None:
        levels = [l for l in levels if l >= args.min_level]
    if args.max_level is not None:
        levels = [l for l in levels if l <= args.max_level]
    if not levels:
        raise SystemExit("No levels selected for overlays.")

    tile_size = dz._z_t_downsample
    l0_offset = dz._l0_offset
    l0_downsamples = tuple(2 ** (dz.level_count - dz_level - 1) for dz_level in range(dz.level_count))

    polys = load_polygons(xml_path)
    tiles_for_polys = compute_level_tiles_for_polys(polys, levels, tile_size, l0_downsamples, l0_offset)

    # Annotation overlay
    ann_tiles_dir = out_base / "annotations_files"
    for lvl in levels:
        tiles = tiles_for_polys[lvl]
        if not tiles:
            continue
        lvl_dir = ann_tiles_dir / str(lvl)
        lvl_dir.mkdir(parents=True, exist_ok=True)
        for col, row in tiles:
            _, z_size = dz._get_tile_info(lvl, (col, row))
            tile = Image.new("RGBA", z_size, (0, 0, 0, 0))
            draw_annotations_tile(polys, tile, lvl, col, row, tile_size, l0_downsamples, l0_offset)
            tile.save(lvl_dir / f"{col}_{row}.png")
    write_dzi(
        slide.dimensions[0],
        slide.dimensions[1],
        tile_size,
        args.overlap,
        "png",
        out_base / "annotations.dzi",
    )

    # Heatmap overlay (render all tiles in selected levels to cover full extent)
    heat_tiles_dir = out_base / "heatmap_files"
    for lvl in levels:
        cols, rows = dz.level_tiles[lvl]
        lvl_dir = heat_tiles_dir / str(lvl)
        lvl_dir.mkdir(parents=True, exist_ok=True)
        for col in range(cols):
            for row in range(rows):
                _, z_size = dz._get_tile_info(lvl, (col, row))
                tile = Image.new("RGBA", z_size, (0, 0, 0, 0))
                draw_heatmap_tile(
                    tile,
                    lvl,
                    col,
                    row,
                    tile_size,
                    l0_downsamples,
                    l0_offset,
                    args.heatmap_cell_size,
                    simple_colormap,
                )
                tile.save(lvl_dir / f"{col}_{row}.png")
    write_dzi(
        slide.dimensions[0],
        slide.dimensions[1],
        tile_size,
        args.overlap,
        "png",
        out_base / "heatmap.dzi",
    )

    print(f"Annotation overlay saved under {ann_tiles_dir.parent}")
    print(f"Heatmap overlay saved under {heat_tiles_dir.parent}")


if __name__ == "__main__":
    main()
