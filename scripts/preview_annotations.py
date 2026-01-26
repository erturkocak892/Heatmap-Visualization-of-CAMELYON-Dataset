#!/usr/bin/env python3
"""
Quick visualization of a CAMELYON WSI with polygon annotations overlaid.

Example:
python scripts/preview_annotations.py \
  --slide data/camelyon17/training/center_0/patient_010_node_4.tif \
  --xml data/camelyon17/training/center_0/patient_010_node_4.xml \
  --output plots/patient_010_node_4_preview.png
"""

import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import openslide
except ImportError as exc:
    raise SystemExit(
        "openslide-python is required. Install with `pip install openslide-python` "
        "and ensure OpenSlide is installed (e.g., `brew install openslide`)."
    ) from exc


def load_polygons(xml_path: Path):
    """Return a list of polygons with labels from an ASAP XML file."""
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
        polys.append({"label": label, "coords": coords, "color": color})
    return polys


def union_bbox(polys):
    xs, ys = [], []
    for poly in polys:
        for x, y in poly["coords"]:
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def clamp_bbox(bbox, dims):
    """Clamp bbox to slide dimensions."""
    min_x, min_y, max_x, max_y = bbox
    w, h = dims
    return (
        max(0, min_x),
        max(0, min_y),
        min(w, max_x),
        min(h, max_y),
    )


def read_region(slide, bbox, level, pad):
    """Read a padded region at a given level, returning (PIL.Image, metadata)."""
    min_x, min_y, max_x, max_y = bbox
    ds = float(slide.level_downsamples[level])
    padded = clamp_bbox(
        (min_x - pad, min_y - pad, max_x + pad, max_y + pad),
        slide.dimensions,
    )
    min_x, min_y, max_x, max_y = padded
    width = int(math.ceil((max_x - min_x) / ds))
    height = int(math.ceil((max_y - min_y) / ds))
    if width <= 0 or height <= 0:
        raise ValueError("Computed region has non-positive size.")
    region = slide.read_region((int(min_x), int(min_y)), level, (width, height))
    return region.convert("RGB"), {
        "min_x": min_x,
        "min_y": min_y,
        "downsample": ds,
        "width": width,
        "height": height,
    }


def plot_overlay(image: Image.Image, polys, meta, output, show):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    ds = meta["downsample"]
    min_x, min_y = meta["min_x"], meta["min_y"]

    for poly in polys:
        color = poly.get("color") or ("red" if "tumor" in poly["label"].lower() else "cyan")
        scaled = [((x - min_x) / ds, (y - min_y) / ds) for x, y in poly["coords"]]
        patch = plt.Polygon(scaled, fill=False, edgecolor=color, linewidth=1, alpha=0.8)
        ax.add_patch(patch)

    ax.set_title(f"Annotations overlay (level {meta.get('level')})")
    ax.set_axis_off()

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved preview to {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Preview CAMELYON WSI annotations.")
    parser.add_argument("--slide", required=True, help="Path to .tif WSI.")
    parser.add_argument("--xml", required=True, help="Path to ASAP XML annotation.")
    parser.add_argument("--level", type=int, default=None, help="Slide level to read (default: lowest resolution).")
    parser.add_argument("--padding", type=int, default=2048, help="Padding in level-0 pixels around annotations.")
    parser.add_argument("--output", default="plots/annotation_preview.png", help="Output image path (PNG). Use '-' to skip saving.")
    parser.add_argument("--show", action="store_true", help="Show interactive window.")
    args = parser.parse_args()

    slide_path = Path(args.slide)
    xml_path = Path(args.xml)

    if not slide_path.exists():
        raise SystemExit(f"Slide not found: {slide_path}")
    if not xml_path.exists():
        raise SystemExit(f"XML not found: {xml_path}")

    polys = load_polygons(xml_path)
    if not polys:
        raise SystemExit("No polygons found in XML.")

    slide = openslide.OpenSlide(str(slide_path))
    level = args.level if args.level is not None else slide.level_count - 1
    
    if level < 0 or level >= slide.level_count:
        raise SystemExit(f"Invalid level {level}; slide has {slide.level_count} levels.")

    bbox = union_bbox(polys)
    if bbox is None:
        raise SystemExit("Could not compute bounding box for annotations.")

    image, meta = read_region(slide, bbox, level=level, pad=args.padding)
    meta["level"] = level

    out_path = None if args.output == "-" else args.output
    plot_overlay(image, polys, meta, output=out_path, show=args.show)


if __name__ == "__main__":
    main()
