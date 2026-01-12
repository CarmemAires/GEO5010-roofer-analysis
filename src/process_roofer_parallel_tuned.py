#!/usr/bin/env python3
"""
process_roofer_with_cjio_parallel.py

Parallelized processing of LAZ tiles with Roofer and CJIO:

Workflow:
1. Reads a GeoPackage tile index (GPKG_PATH) and filters tiles with a valid typology.
2. For each tile:
   - Downloads the LAZ file from the given URL.
   - Creates tile-specific cropped building footprints from the master footprints GPKG.
   - Runs Roofer to generate 3D roof models.
   - Finds the produced .city.jsonl, copies it to DEST_DIR, and converts it to .city.json using CJIO.
3. Tasks run concurrently via ThreadPoolExecutor; the number of workers is configurable (MAX_WORKERS).

Notes:
- Roofer is CPU- and disk-intensive; choose MAX_WORKERS conservatively.
- Downloads are retried automatically; partial downloads are cleaned up on failure.
- CJIO conversion is robust to Windows file-locking issues by using temporary files and retries.
- Logs progress for downloads, Roofer execution, and CJIO conversion.

Requirements:
- Python ≥ 3.8
- geopandas, requests, tqdm
- Roofer executable and CJIO CLI accessible

Configuration:
- Set GPKG_PATH, DEST_DIR, ROOFER_EXE, FOOTPRINTS_GPKG, ROOFER_PARAMS, MAX_WORKERS as needed.
- The script assumes a column 'Typology' exists in the tile index to filter relevant tiles.

Author: Carmem E. F. Aires
Date: 2026-01-12
"""
from __future__ import annotations
import logging
import subprocess
import sys
from pathlib import Path
from shutil import copy2, which
from typing import Optional, Iterable, Union, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import uuid
import geopandas as gpd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# ---------- CONFIG ----------
GPKG_PATH = Path(r"data/BR17_SaoPaulo_TileIndex.gpkg") # Set the path to tile-index, with the 'Typology' attribute BR17_SaoPaulo_TileIndex_ChosenTypology
DEST_DIR = Path(r"data/output") # Set the path to directory to store outputs
ROOFER_EXE = Path(r"bin/roofer.exe") # Set the path to roofer.exe
FOOTPRINTS_GPKG = Path("data/footprints.gpkg") # Set the path to the footprints GPKG
OUTPUT_SUBDIRNAME = DEST_DIR / "intermediate_output"  

#  PARAMS
ROOFER_PARAMS = [
    "--complexity-factor", "0.741",
    "--plane-detect-min-points", "21",
    "--plane-detect-epsilon", "0.131",
    "--plane-detect-k", "20",
]

FIELD_URL = "URL"
FIELD_FILE_NAME = "file_name"
FIELD_TYPOLOGY = "Typology"

# Parallelism: number of concurrent tile workers (download + roofer + conversion)
# WARNING: roofer instances are resource heavy — choose conservatively.
MAX_WORKERS = 3

# ----------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_requests_session(retries: int = 3, backoff_factor: float = 0.3,
                            status_forcelist: Iterable[int] = (500, 502, 504)) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries,
                  backoff_factor=backoff_factor, status_forcelist=status_forcelist,
                  allowed_methods=frozenset(["GET"]))
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def download_stream(session: requests.Session, url: str, dest_path: Path,
                    chunk_size: int = 1024 * 1024) -> None:
    """Stream-download a file to disk (atomic-ish); overwrites if exists."""
    logging.info(f"Downloading {url} -> {dest_path}")
    tmp = dest_path.with_suffix(dest_path.suffix + ".partial")
    try:
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = r.headers.get("content-length")
            total_int = int(total) if total and total.isdigit() else None
            with open(tmp, "wb") as f, tqdm(total=total_int, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))
        tmp.replace(dest_path)
        logging.info(f"Downloaded {dest_path} ({dest_path.stat().st_size} bytes)")
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            logging.debug("Could not remove partial file", exc_info=True)
        raise


def sanitize_name(file_name_raw: Optional[Union[str, bytes]],
                  typology_raw: Optional[Union[str, bytes]]) -> Optional[str]:
    file_name = (file_name_raw or "").decode() if isinstance(file_name_raw, bytes) else (file_name_raw or "")
    typology = (typology_raw or "").decode() if isinstance(typology_raw, bytes) else (typology_raw or "")
    file_name = file_name.strip()
    typology = typology.strip()
    lowers = file_name.lower()
    for ext in (".laz", ".laz.gz", ".laz.zip", ".zip", ".gz"):
        if lowers.endswith(ext):
            file_name = file_name[: -len(ext)]
            break
    if typology == "" or typology.lower() in {"none", "null", "nan"}:
        return None
    bad = r'\/:*?"<>|'
    for ch in bad:
        file_name = file_name.replace(ch, "_")
        typology = typology.replace(ch, "_")
    typology = typology.replace(" ", "_")
    base = f"{file_name}_{typology}".strip("_")
    return base


def create_tile_cropped_footprints(buildings_gdf: gpd.GeoDataFrame, 
                                   tile_geom: Any,
                                   tile_crs: Any,
                                   file_name_base: str,
                                   typology: str,
                                   footprints_base_name: str,
                                   dest_dir: Path) -> Path:
    """
    Create a cropped footprints GPKG for a single tile.
    Uses sanitized naming: {file_name}_{Typology}_{footprints_base_name}.gpkg
    Returns the path to the created file.
    """
    # Sanitize the name for the cropped footprints file
    sanitized = sanitize_name(file_name_base, typology)
    if sanitized is None:
        raise ValueError("Cannot create cropped footprints with invalid typology")
    
    output_path = dest_dir / f"{sanitized}_{footprints_base_name}.gpkg"
    
    if output_path.exists():
        logging.info(f"Using existing cropped footprints: {output_path}")
        return output_path
    
    logging.info(f"Creating tile-specific cropped footprints: {output_path}")
    
    # Create a single-row GeoDataFrame for this tile
    tile_gdf = gpd.GeoDataFrame([{'geometry': tile_geom}], crs=tile_crs)
    
    # Ensure CRS match
    if buildings_gdf.crs != tile_gdf.crs:
        tile_gdf = tile_gdf.to_crs(buildings_gdf.crs)
    
    # Crop buildings to tile extent
    cropped = gpd.overlay(buildings_gdf, tile_gdf, how="intersection")
    
    # Explode multipart geometries
    try:
        cropped = cropped.explode(index_parts=False)
    except TypeError:
        cropped = cropped.explode()
    
    # Save to file
    cropped.to_file(output_path, driver="GPKG")
    logging.info(f"Cropped footprints written: {output_path} ({len(cropped)} features)")
    
    return output_path


def call_roofer_stream(roofer_exe: Path, laz_file: Path, footprints_gpkg: Path, out_dir: Path) -> None:
    """Run roofer and stream combined stdout/stderr to logging. Raises on non-zero exit."""
    cmd = [str(roofer_exe)] + ROOFER_PARAMS + [str(laz_file), str(footprints_gpkg), str(out_dir)]
    logging.info("Running roofer: " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True, bufsize=1)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            logging.info("[roofer] " + line.rstrip())
        ret = proc.wait()
        logging.info(f"roofer exit code: {ret}")
        if ret != 0:
            raise RuntimeError(f"roofer failed (exit code {ret})")
    finally:
        if proc.stdout:
            proc.stdout.close()


def find_city_jsonl_file(out_dir: Path, expected_basename: str) -> Optional[Path]:
    """Prefer files beginning with expected_basename, otherwise return newest *.city.jsonl."""
    candidates = sorted(out_dir.glob(f"{expected_basename}*.city.jsonl"))
    if candidates:
        return candidates[0]
    all_candidates = sorted(out_dir.glob("*.city.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return all_candidates[0] if all_candidates else None


def _find_cjio_executable() -> Optional[list]:
    """
    Return a command list that will run the cjio CLI, or None if not found.
    Prefers:
    1) 'cjio' found on PATH
    2) <venv>/Scripts/cjio.exe (Windows) or <venv>/bin/cjio (Unix)
    Does NOT try python -m cjio because some cjio packages don't expose __main__.
    """
    # plain 'cjio' on PATH
    if which("cjio"):
        return ["cjio", "stdin", "save", "stdout"]
    # try to locate the script next to the current python executable
    exe_dir = Path(sys.executable).parent
    candidates = [
        exe_dir / "cjio.exe",  # typical Windows console exe
        exe_dir / "cjio",  # Unix-style script
        exe_dir / "cjio-script.py",  # sometimes the entrypoint is a script

    ]
    for c in candidates:
        if c.exists():
            # If it's a python script (cjio-script.py), run it with sys.executable
            if c.suffix.lower() in {".py"}:
                return [sys.executable, str(c), "stdin", "save", "stdout"]
            return [str(c), "stdin", "save", "stdout"]
    return None


def convert_with_cjio_streaming(jsonl_path: Path, cityjson_path: Path,
                                max_replace_retries: int = 8) -> None:
    """
    Stream the jsonl file into cjio (using the cjio console script if available),
    write result to a unique temp file and then move it into place with retries
    to avoid Windows file-lock races.
    Raises RuntimeError if cjio cannot be found or if conversion fails.
    """
    cmd = _find_cjio_executable()
    if cmd is None:
        raise RuntimeError(
            "cjio CLI not found on PATH and no cjio script located next to the current Python executable. "
            "Install cjio (pip install --upgrade cjio) or make the 'cjio' script available."
        )
    # Make a unique temp output filename to avoid collisions
    uniq = uuid.uuid4().hex[:8]
    tmp_out = cityjson_path.with_name(cityjson_path.stem + f".{uniq}.city.json.partial")
    last_exc = None
    try:
        logging.info(f"Running cjio: {' '.join(cmd)} (writing to {tmp_out.name})")
        # Open files explicitly so they will be closed before we attempt the replace
        with open(jsonl_path, "rb") as fin, open(tmp_out, "wb") as fout:
            proc = subprocess.Popen(cmd, stdin=fin, stdout=fout, stderr=subprocess.PIPE)
            _, stderr = proc.communicate()
            stderr_text = (stderr.decode("utf-8", errors="replace") if stderr else "").strip()
            if proc.returncode != 0:
                logging.warning(f"cjio failed (exit {proc.returncode}): {stderr_text}")
                last_exc = RuntimeError(f"cjio exit {proc.returncode}: {stderr_text}")
                # ensure tmp_out removed if created
                try:
                    if tmp_out.exists():
                        tmp_out.unlink()
                except Exception:
                    logging.debug("Failed removing tmp cjio output", exc_info=True)
                raise last_exc

        # At this point the cjio process ended successfully and tmp_out should be complete.
        # Try to atomically replace the target; on Windows this can transiently fail if another
        # process briefly holds the target open (anti-virus, indexing, etc.). Retry a few times.
        for attempt in range(1, max_replace_retries + 1):
            try:
                tmp_out.replace(cityjson_path)
                logging.info(f"cjio conversion succeeded -> {cityjson_path}")
                return
            except PermissionError as pe:
                # brief backoff then retry
                wait = 0.05 * (2 ** (attempt - 1))
                logging.debug(f"replace attempt {attempt}/{max_replace_retries} failed with PermissionError; retrying in {wait:.3f}s")
                time.sleep(wait)
            except Exception as e:
                logging.exception("Unexpected error while replacing tmp cjio output into place")
                # remove tmp file and re-raise
                try:
                    if tmp_out.exists():
                        tmp_out.unlink()
                except Exception:
                    logging.debug("Failed to remove tmp file after unexpected replace error", exc_info=True)
                raise
        # If we exhausted retries:
        raise RuntimeError(f"Could not move temporary cjio output {tmp_out} -> {cityjson_path}: permission error after {max_replace_retries} attempts")
    finally:
        # Ensure we don't leave behind stray partial files on exceptions
        try:
            if tmp_out.exists():
                tmp_out.unlink()
        except Exception:
            logging.debug("Could not remove tmp cjio partial on cleanup", exc_info=True)


def convert_all_jsonl_in_dir(dest_dir: Path) -> None:
    logging.info(f"Scanning {dest_dir} for .city.jsonl")
    total = converted = skipped = failed = 0
    for j in sorted(dest_dir.glob("*.city.jsonl")):
        total += 1
        out = dest_dir / (j.name.replace(".city.jsonl", ".city.json"))
        if out.exists():
            skipped += 1
            logging.info(f"Skipping {j.name} -> {out.name} exists")
            continue
        try:
            convert_with_cjio_streaming(j, out)
            converted += 1
        except Exception:
            failed += 1
            logging.exception(f"Failed converting {j.name}")
    logging.info(f"convert_all: total={total}, converted={converted}, skipped={skipped}, failed={failed}")


def process_tile_task(tile_info: Dict[str, Any], 
                     session: requests.Session,
                     buildings_gdf: gpd.GeoDataFrame,
                     roofer_output_dir: Path,
                     dest_dir: Path) -> Tuple[int, str]:
    """
    Perform full processing for one tile.
    tile_info must contain: idx (int), url (str), file_name_base (str), typology (str),
                            geometry (tile geometry), crs (tile CRS)
    Returns (idx, 'OK' or error message)
    """
    idx = tile_info["idx"]
    url = tile_info["url"]
    file_name_base = tile_info["file_name_base"]
    typology = tile_info["typology"]
    tile_geom = tile_info["geometry"]
    tile_crs = tile_info["crs"]

    try:
        out_name = sanitize_name(file_name_base, typology)
        if out_name is None:
            msg = "invalid typology after sanitization"
            logging.info(f"Row {idx}: {msg}")
            return idx, msg

        final_jsonl = dest_dir / f"{out_name}.city.jsonl"
        final_cityjson = dest_dir / f"{out_name}.city.json"
        laz_path = dest_dir / f"{out_name}.laz"
        
        # # Create tile-specific cropped footprints path with sanitized name
        # cropped_fp = dest_dir / f"{out_name}_{tile_info['footprints_base_name']}.gpkg"

        # skip if final already exists (but still try to ensure .city.json exists)
        if final_jsonl.exists():
            logging.info(f"Row {idx}: {final_jsonl} exists; skipping roofer.")
            if not final_cityjson.exists():
                try:
                    convert_with_cjio_streaming(final_jsonl, final_cityjson)
                except Exception as e:
                    logging.exception(f"Row {idx}: failed to convert existing {final_jsonl}: {e}")
                    return idx, f"cjio convert failed for existing jsonl: {e}"
            return idx, "OK (already existed)"

        # download laz if missing
        if not laz_path.exists():
            try:
                download_stream(session, url, laz_path)
            except Exception as e:
                logging.exception(f"Row {idx}: download failed: {e}")
                return idx, f"download failed: {e}"
        else:
            logging.info(f"Row {idx}: {laz_path} already exists, skipping download")

        # Create tile-specific cropped footprints
        try:
            tile_cropped_fp = create_tile_cropped_footprints(
                buildings_gdf, tile_geom, tile_crs, file_name_base, typology, 
                tile_info["footprints_base_name"], dest_dir
            )
        except Exception as e:
            logging.exception(f"Row {idx}: failed to create cropped footprints: {e}")
            return idx, f"cropped footprints creation failed: {e}"

        # ensure roofer output dir present (roofer writes inside it)
        safe_mkdir(roofer_output_dir)

        # call roofer with tile-specific cropped footprints
        try:
            call_roofer_stream(ROOFER_EXE, laz_path, tile_cropped_fp, roofer_output_dir)
        except Exception as e:
            logging.exception(f"Row {idx}: roofer failed: {e}")
            return idx, f"roofer failed: {e}"

        # find produced jsonl
        city_jsonl_found = find_city_jsonl_file(roofer_output_dir, out_name)
        if not city_jsonl_found:
            logging.warning(f"Row {idx}: no .city.jsonl found for {out_name} in {roofer_output_dir}")
            # list files for debugging
            for p in sorted(roofer_output_dir.iterdir()):
                logging.debug(f"  roofer output: {p.name}")
            return idx, "no jsonl produced"

        # copy to final destination
        try:
            copy2(city_jsonl_found, final_jsonl)
            logging.info(f"Row {idx}: copied {city_jsonl_found.name} -> {final_jsonl.name}")
        except Exception as e:
            logging.exception(f"Row {idx}: copy failed: {e}")
            return idx, f"copy failed: {e}"

        # convert
        try:
            convert_with_cjio_streaming(final_jsonl, final_cityjson)
        except Exception as e:
            logging.exception(f"Row {idx}: cjio conversion failed: {e}")
            # still return OK but note conversion failure
            return idx, f"cjio conversion failed: {e}"

        logging.info(f"Row {idx}: processed successfully")
        return idx, "OK"

    except Exception as e:
        logging.exception(f"Row {idx}: unexpected error: {e}")
        return idx, f"unexpected error: {e}"


def main() -> None:
    start_time = time.time()

    if not GPKG_PATH.exists():
        logging.error(f"GeoPackage not found: {GPKG_PATH}")
        return
    if not ROOFER_EXE.exists():
        logging.error(f"roofer executable not found: {ROOFER_EXE}")
        return
    if not FOOTPRINTS_GPKG.exists():
        logging.error(f"Footprints file not found: {FOOTPRINTS_GPKG}")
        return

    safe_mkdir(DEST_DIR)
    roofer_output_dir = DEST_DIR / OUTPUT_SUBDIRNAME
    safe_mkdir(roofer_output_dir)

    session = build_requests_session()

    try:
        dg = gpd.read_file(GPKG_PATH)
    except Exception:
        logging.exception("Failed to read GeoPackage")
        return

    logging.info(f"Read {len(dg)} rows from {GPKG_PATH}")

    if FIELD_TYPOLOGY not in dg.columns:
        logging.error(f"Field '{FIELD_TYPOLOGY}' not in GPKG. Available fields: {list(dg.columns)}")
        return

    rows = dg[dg[FIELD_TYPOLOGY].notnull()]
    if rows.empty:
        logging.warning("No rows with non-null Typology found — exiting")
        return

    # Load buildings once (will be passed to each task)
    logging.info(f"Loading buildings from {FOOTPRINTS_GPKG}")
    buildings = gpd.read_file(FOOTPRINTS_GPKG)
    footprints_base_name = FOOTPRINTS_GPKG.stem

    # Extract tile info into plain dicts including geometry
    tile_infos = []
    for idx, r in rows.iterrows():
        url = r.get(FIELD_URL)
        file_name_base = str(r.get(FIELD_FILE_NAME) or f"row{idx}")
        typology = str(r.get(FIELD_TYPOLOGY) or "")
        if not url or not isinstance(url, str) or url.strip() == "":
            logging.warning(f"Row {idx} missing URL; skipping.")
            continue
        
        tile_infos.append({
            "idx": int(idx),
            "url": url.strip(),
            "file_name_base": file_name_base,
            "typology": typology,
            "geometry": r.geometry,
            "crs": rows.crs,
            "footprints_base_name": footprints_base_name
        })

    logging.info(f"Submitting {len(tile_infos)} tiles to a pool with max_workers={MAX_WORKERS}")

    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(process_tile_task, info, session, buildings, 
                     roofer_output_dir, DEST_DIR): info
            for info in tile_infos
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="tiles"):
            info = futures[future]
            try:
                idx, status = future.result()
                results[idx] = status
                logging.info(f"Tile {idx} finished: {status}")
            except Exception as e:
                logging.exception(f"Tile {info.get('idx')} failed with exception: {e}")
                results[info.get("idx")] = f"exception: {e}"

    # Final pass: convert any remaining .city.jsonl in DEST_DIR
    convert_all_jsonl_in_dir(DEST_DIR)

    elapsed = time.time() - start_time

    # Summary
    ok = sum(1 for s in results.values() if s == "OK" or s.startswith("OK"))
    failed = len(results) - ok
    logging.info(f"Processing done in {elapsed:.1f}s: total={len(results)}, ok={ok}, failed={failed}")
    for idx in sorted(results):
        logging.info(f"  - row {idx}: {results[idx]}")


if __name__ == "__main__":
    main()