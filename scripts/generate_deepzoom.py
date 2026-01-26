#!/usr/bin/env python3

import argparse
from pathlib import Path

from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator


def save_dzi(dz: DeepZoomGenerator, out_dir: Path, stem: str, fmt: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dzi_path = out_dir / f"{stem}.dzi"
    dzi_path.write_text(dz.get_dzi(fmt), encoding="utf-8")


def save_tiles(dz: DeepZoomGenerator, out_dir: Path, fmt: str) -> None:
    for level in range(dz.level_count):
        level_dir = out_dir / str(level)
        level_dir.mkdir(parents=True, exist_ok=True)
        cols, rows = dz.level_tiles[level]
        for col in range(cols):
            for row in range(rows):
                tile = dz.get_tile(level, (col, row))
                tile.save(level_dir / f"{col}_{row}.{fmt}")


def main():
    parser = argparse.ArgumentParser(description="Generate Deep Zoom tiles from a WSI.")
    parser.add_argument("--slide", required=True, help="Path to WSI (e.g., .tif)")
    parser.add_argument("--output", required=True, help="Output directory for DZI + tiles")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size")
    parser.add_argument("--overlap", type=int, default=0, help="Tile overlap")
    parser.add_argument("--format", choices=["jpeg", "png"], default="jpeg", help="Tile format")
    parser.add_argument("--limit-bounds", action="store_true", help="Limit to non-empty region")
    args = parser.parse_args()

    slide_path = Path(args.slide)
    out_dir = Path(args.output)
    stem = slide_path.stem

    slide = OpenSlide(str(slide_path))
    dz = DeepZoomGenerator(
        slide,x
        tile_size=args.tile_size,
        overlap=args.overlap,
        limit_bounds=args.limit_bounds,
    )

    dzi_out_dir = out_dir.parent if out_dir.name.endswith("_files") else out_dir
    tiles_dir = out_dir if out_dir.name.endswith("_files") else out_dir / f"{stem}_files"

    save_dzi(dz, dzi_out_dir, stem, args.format)
    save_tiles(dz, tiles_dir, args.format)
    print(f"Saved DZI to {dzi_out_dir / (stem + '.dzi')}")
    print(f"Saved tiles under {tiles_dir}")


if __name__ == "__main__":
    main()
