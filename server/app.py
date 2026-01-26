#!/usr/bin/env python3
"""
On-demand Deep Zoom tile server for CAMELYON WSIs and overlays.

Endpoints (under /api):
- /api/dzi                          -> DZI XML for the base WSI
- /api/tile/{level}/{col}_{row}.jpeg -> Base WSI tile (JPEG)
- /api/overlay/annotations/dzi       -> DZI XML for annotation overlay (PNG)
- /api/overlay/annotations/tile/{level}/{col}_{row}.png -> Annotation tile
- /api/overlay/heatmap/dzi           -> DZI XML for heatmap overlay (PNG)
- /api/overlay/heatmap/tile/{level}/{col}_{row}.png     -> Heatmap tile

Static files (web UI) are served from ../web at "/".
"""

from __future__ import annotations

import io
import math
import random
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import hashlib

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image, ImageDraw, ImageFilter

# --- Configuration ---
ROOT = Path(__file__).resolve().parent.parent
SLIDE_PATH = ROOT / "data/camelyon17/training/center_0/patient_010_node_4.tif"
ANNOTATION_XML = ROOT / "data/camelyon17/training/center_0/patient_010_node_4.xml"
TILE_SIZE = 256
OVERLAP = 0
LIMIT_BOUNDS = True
HEATMAP_CELL_SIZE = 2048  # spacing between noise grid points (level-0 pixels)
HEATMAP_OFFSET = (-2048, 0)  # shift heatmap noise field in level-0 coords (x, y)
HEATMAP_PALETTES = {
    "metastasis": [
        (0.0, (40, 70, 170)),
        (0.33, (0, 180, 180)),
        (0.66, (255, 220, 80)),
        (1.0, (255, 50, 50)),
    ],
    "epithelial": [
        (0.0, (20, 100, 130)),
        (0.33, (20, 170, 150)),
        (0.66, (190, 210, 120)),
        (1.0, (240, 190, 90)),
    ],
    "normal": [
        (0.0, (40, 120, 70)),
        (0.33, (80, 180, 80)),
        (0.66, (200, 210, 120)),
        (1.0, (230, 190, 110)),
    ],
}


def _load_slide() -> tuple[OpenSlide, DeepZoomGenerator]:
    slide = OpenSlide(str(SLIDE_PATH))
    dz = DeepZoomGenerator(
        slide, tile_size=TILE_SIZE, overlap=OVERLAP, limit_bounds=LIMIT_BOUNDS
    )
    return slide, dz


def _l0_downsamples_for_dz(dz: DeepZoomGenerator) -> tuple[int, ...]:
    """DeepZoomGenerator exposes level_count; derive l0 downsample per deep zoom level."""
    return tuple(2 ** (dz.level_count - dz_level - 1) for dz_level in range(dz.level_count))


def _load_polygons(xml_path: Path):
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
        color = ann.attrib.get("Color") or "#F4FA58"
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


def _polys_bbox(polys):
    xs, ys = [], []
    for p in polys:
        x0, y0, x1, y1 = p["bbox"]
        xs.extend([x0, x1])
        ys.extend([y0, y1])
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


@lru_cache(maxsize=1)
def _polygons():
    return _load_polygons(ANNOTATION_XML)


def _dzi_xml(width: int, height: int, tile_size: int, overlap: int, fmt: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" TileSize="{tile_size}" Overlap="{overlap}" Format="{fmt}">
    <Size Width="{width}" Height="{height}"/>
</Image>
"""


def _draw_annotations_tile(polys, tile, lvl, col, row, tile_size, l0_downsamples, offset):
    draw = ImageDraw.Draw(tile, "RGBA")
    down = l0_downsamples[lvl]
    origin_x = col * tile_size
    origin_y = row * tile_size
    for poly in polys:
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


def _draw_tissue_tile(tile, lvl, col, row, tile_size, l0_downsamples, offset, bbox):
    """Render a simple tissue mask based on the union bbox of annotations (placeholder)."""
    if bbox is None:
        return
    draw = ImageDraw.Draw(tile, "RGBA")
    down = l0_downsamples[lvl]
    origin_x = col * tile_size
    origin_y = row * tile_size
    x0, y0, x1, y1 = bbox
    pad = 4096  # add padding to cover nearby tissue
    x0 -= pad
    y0 -= pad
    x1 += pad
    y1 += pad
    zx0 = (x0 - offset[0]) / down - origin_x
    zy0 = (y0 - offset[1]) / down - origin_y
    zx1 = (x1 - offset[0]) / down - origin_x
    zy1 = (y1 - offset[1]) / down - origin_y
    draw.rectangle([zx0, zy0, zx1, zy1], fill=(180, 220, 255, 50), outline=None)


def _model_seed(model: str) -> int:
    digest = hashlib.sha256(model.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _smooth_noise_score(x_l0: float, y_l0: float, spacing: float, model_seed: int) -> float:
    """Deterministic smooth noise via bilinear interpolation on a coarse grid."""
    gx = x_l0 / spacing
    gy = y_l0 / spacing
    x0 = math.floor(gx)
    y0 = math.floor(gy)
    tx = gx - x0
    ty = gy - y0

    def val(ix, iy):
        rng = random.Random((ix * 73856093) ^ (iy * 19349663) ^ model_seed)
        return rng.random()

    v00 = val(x0, y0)
    v10 = val(x0 + 1, y0)
    v01 = val(x0, y0 + 1)
    v11 = val(x0 + 1, y0 + 1)
    v0 = v00 * (1 - tx) + v10 * tx
    v1 = v01 * (1 - tx) + v11 * tx
    return v0 * (1 - ty) + v1 * ty


def _draw_heatmap_tile(tile, lvl, col, row, tile_size, l0_downsamples, offset, spacing, model_seed, palette, bbox=None, field_offset=(0, 0)):
    down = l0_downsamples[lvl]
    origin_x = col * tile_size
    origin_y = row * tile_size
    pix = tile.load()

    bx0 = by0 = bx1 = by1 = None
    if bbox:
        bx0, by0, bx1, by1 = bbox
        pad = spacing * 2
        bx0 -= pad
        by0 -= pad
        bx1 += pad
        by1 += pad

    for y in range(tile.height):
        l0_y = (origin_y + y) * down + offset[1]
        if bbox and (l0_y < by0 or l0_y > by1):
            continue
        for x in range(tile.width):
            l0_x = (origin_x + x) * down + offset[0]
            if bbox and (l0_x < bx0 or l0_x > bx1):
                continue
            score = _smooth_noise_score(l0_x + field_offset[0], l0_y + field_offset[1], spacing, model_seed)
            r, g, b, a = _colormap(score, palette)
            pix[x, y] = (r, g, b, a)


def _colormap(score: float, palette_name: str = "metastasis"):
    score = max(0.0, min(1.0, score))
    stops = HEATMAP_PALETTES.get(palette_name, HEATMAP_PALETTES["metastasis"])
    for i in range(len(stops) - 1):
        s0, c0 = stops[i]
        s1, c1 = stops[i + 1]
        if score <= s1:
            t = (score - s0) / (s1 - s0 + 1e-9)
            r = int(c0[0] + t * (c1[0] - c0[0]))
            g = int(c0[1] + t * (c1[1] - c0[1]))
            b = int(c0[2] + t * (c1[2] - c0[2]))
            break
    else:
        r, g, b = stops[-1][1]
    alpha = int(70 + 90 * score)  # softer by default
    return (r, g, b, alpha)


app = FastAPI(title="CAMELYON Deep Zoom Server", docs_url="/api/docs", openapi_url="/api/openapi.json")

# Serve static web UI under /web; root ("/") returns index.html for convenience
web_root = ROOT / "web"
if web_root.exists():
    app.mount("/web", StaticFiles(directory=web_root, html=True), name="web")


@app.get("/")
def root():
    if not web_root.exists():
        return {"message": "web UI not found", "visit": "/api/docs"}
    index_path = web_root / "index.html"
    if not index_path.exists():
        return {"message": "index.html not found in web/", "visit": "/api/docs"}
    return FileResponse(index_path)

_slide, _dz = _load_slide()
_l0_downsamples = _l0_downsamples_for_dz(_dz)


@app.get("/api/dzi")
def get_dzi():
    xml = _dz.get_dzi("jpeg")
    return Response(xml, media_type="application/xml")


@app.get("/api/tile/{level}/{col}_{row}.jpeg")
def get_tile(level: int, col: int, row: int):
    if level < 0 or level >= _dz.level_count:
        raise HTTPException(status_code=404, detail="Invalid level")
    cols, rows = _dz.level_tiles[level]
    if col < 0 or row < 0 or col >= cols or row >= rows:
        raise HTTPException(status_code=404, detail="Invalid tile address")
    tile = _dz.get_tile(level, (col, row))
    buf = io.BytesIO()
    tile.save(buf, format="JPEG")
    return Response(buf.getvalue(), media_type="image/jpeg")


@app.get("/api/overlay/annotations/dzi")
def get_annotations_dzi():
    w, h = _slide.dimensions
    xml = _dzi_xml(w, h, TILE_SIZE, OVERLAP, "png")
    return Response(xml, media_type="application/xml")


@app.get("/api/overlay/annotations/tile/{level}/{col}_{row}.png")
def get_annotations_tile(level: int, col: int, row: int):
    if level < 0 or level >= _dz.level_count:
        raise HTTPException(status_code=404, detail="Invalid level")
    cols, rows = _dz.level_tiles[level]
    if col < 0 or row < 0 or col >= cols or row >= rows:
        raise HTTPException(status_code=404, detail="Invalid tile address")
    # Get tile dimensions from dz (respect limit_bounds edges)
    _, z_size = _dz._get_tile_info(level, (col, row))
    tile = Image.new("RGBA", z_size, (0, 0, 0, 0))
    _draw_annotations_tile(_polygons(), tile, level, col, row, TILE_SIZE, _l0_downsamples, _dz._l0_offset)
    buf = io.BytesIO()
    tile.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/api/overlay/tissue/dzi")
def get_tissue_dzi():
    w, h = _slide.dimensions
    xml = _dzi_xml(w, h, TILE_SIZE, OVERLAP, "png")
    return Response(xml, media_type="application/xml")


@app.get("/api/overlay/tissue/tile/{level}/{col}_{row}.png")
def get_tissue_tile(level: int, col: int, row: int):
    if level < 0 or level >= _dz.level_count:
        raise HTTPException(status_code=404, detail="Invalid level")
    cols, rows = _dz.level_tiles[level]
    if col < 0 or row < 0 or col >= cols or row >= rows:
        raise HTTPException(status_code=404, detail="Invalid tile address")
    _, z_size = _dz._get_tile_info(level, (col, row))
    tile = Image.new("RGBA", z_size, (0, 0, 0, 0))
    bbox = _polys_bbox(_polygons())
    _draw_tissue_tile(tile, level, col, row, TILE_SIZE, _l0_downsamples, _dz._l0_offset, bbox)
    buf = io.BytesIO()
    tile.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/api/overlay/heatmap/dzi")
def get_heatmap_dzi():
    w, h = _slide.dimensions
    xml = _dzi_xml(w, h, TILE_SIZE, OVERLAP, "png")
    return Response(xml, media_type="application/xml")


@app.get("/api/overlay/heatmap/tile/{level}/{col}_{row}.png")
def get_heatmap_tile(level: int, col: int, row: int, model: str = "mock-default", cell: str = "metastasis"):
    if level < 0 or level >= _dz.level_count:
        raise HTTPException(status_code=404, detail="Invalid level")
    cols, rows = _dz.level_tiles[level]
    if col < 0 or row < 0 or col >= cols or row >= rows:
        raise HTTPException(status_code=404, detail="Invalid tile address")
    _, z_size = _dz._get_tile_info(level, (col, row))
    tile = Image.new("RGBA", z_size, (0, 0, 0, 0))
    bbox = _polys_bbox(_polygons())
    seed = _model_seed(model)
    # Slightly vary offset per model to reduce overlap if multiple are toggled in future
    model_offset = (HEATMAP_OFFSET[0] + (seed % 1024), HEATMAP_OFFSET[1])
    _draw_heatmap_tile(
        tile,
        level,
        col,
        row,
        TILE_SIZE,
        _l0_downsamples,
        _dz._l0_offset,
        HEATMAP_CELL_SIZE,
        seed,
        palette=cell,
        bbox=bbox,
        field_offset=model_offset,
    )
    buf = io.BytesIO()
    tile.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


# Convenience root
@app.get("/api/info")
def info():
    return {
        "slide": str(SLIDE_PATH),
        "dimensions": _slide.dimensions,
        "levels": _dz.level_count,
        "tile_size": TILE_SIZE,
        "overlap": OVERLAP,
        "limit_bounds": LIMIT_BOUNDS,
        "annotations": len(_polygons()),
        "models": [
            {"id": "mock-default", "label": "Mock Default"},
            {"id": "mock-hi-sens", "label": "High Sensitivity"},
            {"id": "mock-hi-spec", "label": "High Specificity"},
        ],
        "cell_classes": [
            {"id": "metastasis", "label": "Metastasis (tumor)"},
            {"id": "epithelial", "label": "Epithelial"},
            {"id": "normal", "label": "Normal"},
        ],
    }
