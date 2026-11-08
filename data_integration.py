"""Step 1: Integrate global urban + land use projections, count pixels per category.

Handles:
- Global urban TIFs in EPSG:6933 (cylindrical equal area), 1km resolution
  Values: 0=no data/outside, 1=non-urban land, 2=urban (SSP-specific)
- Global land use NetCDFs in WGS84, 0.05deg resolution, 33 PFT variables
  Note: lat/lon may be transposed in some files — auto-detected and corrected
- Study area defined by historical LCZ files (any CRS)
- All data reprojected to the LCZ file's CRS and grid (the standard)
- Memory-efficient: processes one SSP/year at a time
- Optional validation maps for visual integration verification
"""

import logging
import re
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import reproject
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from config import PipelineConfig

log = logging.getLogger(__name__)


def detect_study_area(cfg: PipelineConfig) -> PipelineConfig:
    """Auto-detect study area name, bounds, CRS, and grid from historical LCZ files.

    The LCZ file's CRS, transform, and shape become the standard reference grid
    that all other data is reprojected to.
    """
    lcz_files = _discover_lcz_files(cfg.lcz_data_dir)
    if not lcz_files:
        raise FileNotFoundError(f"No LCZ TIF files found in {cfg.lcz_data_dir}")

    # Extract study area name from filenames
    # Handles: "Vancouver_LCZ1986.tif", "Vancouver_1986_filter_final.tif",
    #          "reprojected_Vancouver_1986_filter_final.tif"
    sample_name = lcz_files[0].stem
    # Try "Name_LCZ..." pattern first
    match = re.match(r"^(.+?)_(?:LCZ|lcz)", sample_name)
    if match:
        cfg.study_area_name = match.group(1)
    else:
        # Try "Name_YYYY..." pattern, skip common prefixes like "reprojected"
        match2 = re.match(r"^(?:reprojected_|reprojectied_)?(.+?)_\d{4}", sample_name)
        cfg.study_area_name = match2.group(1) if match2 else sample_name.split("_")[0]

    # Read spatial info from the first file — this becomes the reference grid
    with rasterio.open(lcz_files[0]) as src:
        cfg.study_area_crs = str(src.crs)
        cfg.study_area_transform = src.transform
        cfg.study_area_shape = (src.height, src.width)
        bounds = src.bounds

    # Convert bounds to WGS84 for cropping global datasets
    from pyproj import Transformer
    if cfg.study_area_crs and cfg.study_area_crs != "EPSG:4326":
        t = Transformer.from_crs(cfg.study_area_crs, "EPSG:4326", always_xy=True)
        min_lon, min_lat = t.transform(bounds.left, bounds.bottom)
        max_lon, max_lat = t.transform(bounds.right, bounds.top)
    else:
        min_lat, max_lat = bounds.bottom, bounds.top
        min_lon, max_lon = bounds.left, bounds.right

    cfg.study_area_bounds_wgs84 = {
        "min_lat": min_lat, "max_lat": max_lat,
        "min_lon": min_lon, "max_lon": max_lon,
    }

    # Auto-detect historical years from filenames
    detected_years = []
    for f in lcz_files:
        m = re.search(r"(\d{4})", f.stem)
        if m:
            detected_years.append(int(m.group(1)))
    if detected_years:
        cfg.historical_years = sorted(set(detected_years))

    # Set output dir based on study area name
    cfg.output_dir = Path(f"{cfg.study_area_name}_lcz_projection_results")

    log.info("Study area: %s", cfg.study_area_name)
    log.info("  CRS (standard): %s", cfg.study_area_crs)
    log.info("  Grid: %d x %d pixels", cfg.study_area_shape[0], cfg.study_area_shape[1])
    log.info("  Bounds (WGS84): lat=%.4f..%.4f, lon=%.4f..%.4f",
             min_lat, max_lat, min_lon, max_lon)
    log.info("  Historical years: %s", cfg.historical_years)
    return cfg


def _discover_lcz_files(lcz_dir: Path) -> list:
    """Find historical LCZ TIF files, excluding _uc variants."""
    if not lcz_dir.exists():
        return []
    all_tifs = sorted(lcz_dir.glob("**/*.tif"))
    return [f for f in all_tifs if "_uc" not in f.stem and not f.name.endswith(".aux.xml")]


class CategoryPixelCounter:
    """Integrates global urban and land use data, reprojected to the LCZ reference
    grid, and counts pixels per land use category per SSP/year."""

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        if not self.cfg.study_area_bounds_wgs84:
            raise ValueError("Run detect_study_area(config) before creating CategoryPixelCounter")
        self.bounds = self.cfg.study_area_bounds_wgs84
        # Reference grid from LCZ files
        self.ref_crs = self.cfg.study_area_crs
        self.ref_transform = self.cfg.study_area_transform
        self.ref_shape = self.cfg.study_area_shape  # (height, width)

    def run(self, save_validation: bool = True) -> pd.DataFrame:
        """Process each SSP/year independently for memory efficiency.

        Args:
            save_validation: If True, save validation maps for the first and last
                projection year to verify integration correctness.
        """
        rows = []
        # Determine which SSP/year combos get validation maps
        val_years = {self.cfg.projection_years[0], self.cfg.projection_years[-1]}

        for ssp in self.cfg.ssps:
            ssp_num = ssp.replace("SSP", "")
            for year in self.cfg.projection_years:
                log.info("Processing %s %d...", ssp, year)

                # Load urban data reprojected to LCZ grid (preserves values 0/1/2)
                urban_on_lcz_grid = self._load_urban_on_lcz_grid(ssp_num, year)

                # Load land use data reprojected to LCZ grid
                pft_on_lcz_grid = self._load_landuse_on_lcz_grid(ssp_num, year)

                if pft_on_lcz_grid is None:
                    log.warning("  No land use data for %s %d, skipping", ssp, year)
                    continue

                counts, cat_map = self._integrate_and_count(urban_on_lcz_grid, pft_on_lcz_grid)
                counts["SSP"] = ssp
                counts["Year"] = year
                rows.append(counts)
                log.info("  %s %d: %s", ssp, year,
                         {k: v for k, v in counts.items() if k not in ("SSP", "Year")})

                # Save validation maps for sample years
                if save_validation and year in val_years:
                    self._save_validation_map(
                        cat_map, urban_on_lcz_grid, pft_on_lcz_grid, ssp, year)

        df = pd.DataFrame(rows)
        if len(df) == 0:
            log.error("No data processed!")
            return df

        # Save SSP comparison summary
        if save_validation and len(df) > 0:
            self._save_ssp_comparison(df)

        cat_cols = list(self.cfg.category_pft_mapping.keys())
        ordered = ["SSP", "Year"] + cat_cols + ["Total"]
        return df[[c for c in ordered if c in df.columns]]

    # ---- Urban data: reproject from EPSG:6933 to LCZ grid ----

    def _load_urban_on_lcz_grid(self, ssp_num: str, year: int) -> np.ndarray | None:
        """Load global urban TIF and reproject directly to the LCZ reference grid.

        Input: EPSG:6933, 1km resolution, global
        Output: array matching LCZ grid (ref_crs, ref_transform, ref_shape)
        """
        urban_path = self.cfg.global_urban_dir / f"SSP{ssp_num}" / f"global_SSP{ssp_num}_{year}.tif"
        if not urban_path.exists():
            log.warning("  Urban file not found: %s", urban_path)
            return None

        try:
            dst = np.zeros(self.ref_shape, dtype=np.uint8)
            with rasterio.open(urban_path) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=self.ref_transform,
                    dst_crs=self.ref_crs,
                    resampling=Resampling.nearest,
                )
            return dst
        except Exception as e:
            log.error("  Failed to load urban data: %s", e)
            return None

    # ---- Land use data: crop WGS84 NetCDF then reproject to LCZ grid ----

    def _load_landuse_on_lcz_grid(self, ssp_num: str, year: int) -> dict | None:
        """Load global land use NetCDF, crop to study area, reproject each PFT
        to the LCZ reference grid.

        Input: WGS84, 0.05deg resolution, global, 33 PFT variables
        Output: dict {pft_num: array matching LCZ grid}
        """
        pattern = f"GCAM_Demeter_LU_ssp{ssp_num}_*_{year}.nc"
        matches = list(self.cfg.global_landuse_dir.glob(pattern))
        if not matches:
            log.warning("  Land use file not found: %s", pattern)
            return None
        filepath = matches[0]

        b = self.bounds
        buf = 0.1  # generous buffer in degrees for safe cropping

        try:
            ds = xr.open_dataset(filepath)

            # Find coordinate names
            lat_name = next((n for n in ["lat", "latitude", "y"] if n in ds.coords), None)
            lon_name = next((n for n in ["lon", "longitude", "x"] if n in ds.coords), None)
            if not lat_name or not lon_name:
                lat_name = next((c for c in ds.coords if "lat" in c.lower()), None)
                lon_name = next((c for c in ds.coords if "lon" in c.lower()), None)
            if not lat_name or not lon_name:
                log.error("  Cannot find lat/lon coords in %s", filepath.name)
                ds.close()
                return None

            lats = ds[lat_name].values
            lons = ds[lon_name].values

            # Check if study area center falls within coordinate bounds
            center_lat = (b["min_lat"] + b["max_lat"]) / 2
            center_lon = (b["min_lon"] + b["max_lon"]) / 2
            lat_range = (float(lats.min()), float(lats.max()))
            lon_range = (float(lons.min()), float(lons.max()))

            in_bounds = (lat_range[0] <= center_lat <= lat_range[1] and
                         lon_range[0] <= center_lon <= lon_range[1])

            if not in_bounds:
                # Check if lat/lon are transposed
                swapped = (lon_range[0] <= center_lat <= lon_range[1] and
                           lat_range[0] <= center_lon <= lat_range[1])
                if swapped:
                    log.warning("  Coordinates appear transposed in %s, swapping lat/lon",
                                filepath.name)
                    ds = ds.rename({lat_name: "_tmp_lon", lon_name: "_tmp_lat"})
                    ds = ds.rename({"_tmp_lat": lat_name, "_tmp_lon": lon_name})
                    lats = ds[lat_name].values
                    lons = ds[lon_name].values
                else:
                    log.warning("  Study area center (%.2f, %.2f) not in coordinate bounds "
                                "lat=(%.2f, %.2f) lon=(%.2f, %.2f)",
                                center_lat, center_lon,
                                lat_range[0], lat_range[1],
                                lon_range[0], lon_range[1])

            # Crop to study area (WGS84 bounds)
            lat_asc = lats[-1] > lats[0] if len(lats) > 1 else True
            lon_asc = lons[-1] > lons[0] if len(lons) > 1 else True
            lat_sl = slice(b["min_lat"] - buf, b["max_lat"] + buf) if lat_asc else slice(b["max_lat"] + buf, b["min_lat"] - buf)
            lon_sl = slice(b["min_lon"] - buf, b["max_lon"] + buf) if lon_asc else slice(b["max_lon"] + buf, b["min_lon"] - buf)

            cropped = ds.sel({lat_name: lat_sl, lon_name: lon_sl})
            c_lats = cropped[lat_name].values
            c_lons = cropped[lon_name].values

            if len(c_lats) == 0 or len(c_lons) == 0:
                log.error("  Empty crop for %s", filepath.name)
                ds.close()
                return None

            # Build a rasterio-compatible transform for the cropped WGS84 grid
            # Use pixel edges (not centers) for correct pixel size alignment.
            # NetCDF coordinates are pixel centers at 0.05° spacing.
            crop_shape = (len(c_lats), len(c_lons))
            half_dx = abs(float(c_lons[1] - c_lons[0])) / 2 if len(c_lons) > 1 else 0.025
            half_dy = abs(float(c_lats[1] - c_lats[0])) / 2 if len(c_lats) > 1 else 0.025
            crop_transform = from_bounds(
                float(c_lons.min()) - half_dx, float(c_lats.min()) - half_dy,
                float(c_lons.max()) + half_dx, float(c_lats.max()) + half_dy,
                crop_shape[1], crop_shape[0],
            )

            # Reproject each PFT variable to the LCZ reference grid
            pft_on_grid = {}
            for var in cropped.data_vars:
                pft_num = self._extract_pft_number(str(var))
                if pft_num is None:
                    continue

                arr = cropped[var].values.astype(np.float32)
                if arr.shape != crop_shape:
                    if arr.shape == (crop_shape[1], crop_shape[0]):
                        arr = arr.T
                    else:
                        continue

                dst = np.zeros(self.ref_shape, dtype=np.float32)
                reproject(
                    source=arr,
                    destination=dst,
                    src_transform=crop_transform,
                    src_crs="EPSG:4326",
                    dst_transform=self.ref_transform,
                    dst_crs=self.ref_crs,
                    resampling=Resampling.bilinear,
                )
                pft_on_grid[pft_num] = dst

            ds.close()
            return pft_on_grid if pft_on_grid else None

        except Exception as e:
            log.error("  Failed to load land use data: %s", e)
            return None

    def _extract_pft_number(self, var_name: str) -> int | None:
        """Extract PFT number from variable name like 'PFT01', 'PFT1'."""
        m = re.search(r'PFT[_]?(\d+)', var_name, re.IGNORECASE)
        return int(m.group(1)) if m else None

    # ---- Integration and counting ----

    def _integrate_and_count(self, urban_data: np.ndarray | None,
                              pft_on_grid: dict) -> tuple[dict, np.ndarray]:
        """Build dominant-PFT map on the LCZ grid, overlay urban, count per category.

        All inputs are already on the LCZ reference grid (same CRS, transform, shape).

        Urban TIF values: 0=no data, 1=non-urban land, 2=urban.
        Only value == 2 is treated as urban (matching the original integration script).

        Returns:
            (counts_dict, category_map): pixel counts per category, and the
            integer category map (0-6) for validation visualization.
        """
        h, w = self.ref_shape

        # Find dominant PFT per pixel (highest percentage wins)
        pft_map = np.full((h, w), -1, dtype=np.int32)  # -1 = no data
        max_vals = np.full((h, w), 0.0, dtype=np.float32)  # threshold: > 0

        pft_stats = {}
        for pft_num, arr in pft_on_grid.items():
            clean = np.nan_to_num(arr, nan=0.0)
            mask = clean > max_vals
            pft_map[mask] = pft_num
            max_vals[mask] = clean[mask]
            nz = int(np.sum(clean > 0))
            if nz > 0:
                pft_stats[pft_num] = nz

        n_with_data = int(np.sum(pft_map >= 0))
        log.info("  PFTs with data on grid: %s",
                 {f"PFT{k}": v for k, v in sorted(pft_stats.items())})
        log.info("  Pixels with land use data: %d / %d", n_with_data, h * w)

        # Urban overlay: ONLY value 2 = urban (value 1 = non-urban land)
        if urban_data is not None:
            urban_mask = urban_data == 2
            pft_map[urban_mask] = 31
            n_urban = int(np.sum(urban_mask))
            log.info("  Urban pixels (value==2): %d", n_urban)

        # Pixels with no data from either source are water (ocean/large water bodies
        # outside the land use model domain)
        nodata_mask = pft_map == -1
        n_nodata = int(np.sum(nodata_mask))
        if n_nodata > 0:
            pft_map[nodata_mask] = 0  # PFT0 = Water
            log.info("  No-data pixels classified as Water: %d", n_nodata)

        # Build category map: assign each pixel a category ID
        # 0=unclassified, 1=Urban, 2=Crops, 3=Grass/Shrub, 4=Forest, 5=Water, 6=Barren
        cat_map = np.zeros((h, w), dtype=np.int8)

        # Count pixels per category
        counts = {}
        total = 0
        for cat_idx, (cat_name, pft_list) in enumerate(self.cfg.category_pft_mapping.items(), 1):
            cat_mask = np.zeros((h, w), dtype=bool)
            for pft in pft_list:
                cat_mask |= (pft_map == pft)
            n = int(np.sum(cat_mask))
            counts[cat_name] = n
            total += n
            cat_map[cat_mask] = cat_idx

        counts["Total"] = total
        return counts, cat_map

    # ---- Validation maps ----

    def _save_validation_map(self, cat_map: np.ndarray, urban_data: np.ndarray | None,
                              pft_on_grid: dict, ssp: str, year: int):
        """Save validation outputs: GeoTIFFs + 3-panel PNG (urban, land use, integrated).

        Matches the original script's validation plot style with geographic extent.
        """
        val_dir = self.cfg.output_dir / "validation_maps"
        val_dir.mkdir(exist_ok=True, parents=True)
        name = self.cfg.study_area_name or "study_area"
        prefix = f"{name}_{ssp}_{year}"

        # Save category map as GeoTIFF
        tif_path = val_dir / f"{prefix}_category_map.tif"
        with rasterio.open(
            tif_path, "w", driver="GTiff",
            height=self.ref_shape[0], width=self.ref_shape[1],
            count=1, dtype="int8", crs=self.ref_crs,
            transform=self.ref_transform,
        ) as dst:
            dst.write(cat_map, 1)
        log.info("  Saved validation GeoTIFF: %s", tif_path.name)

        # Save urban overlay as GeoTIFF
        if urban_data is not None:
            urban_tif = val_dir / f"{prefix}_urban_overlay.tif"
            with rasterio.open(
                urban_tif, "w", driver="GTiff",
                height=self.ref_shape[0], width=self.ref_shape[1],
                count=1, dtype="uint8", crs=self.ref_crs,
                transform=self.ref_transform,
            ) as dst:
                dst.write(urban_data, 1)

        # Compute geographic extent in WGS84 for plotting (regardless of CRS)
        h, w = self.ref_shape
        b = self.bounds  # WGS84 bounds
        extent = [b["min_lon"], b["max_lon"], b["min_lat"], b["max_lat"]]

        # Build dominant PFT map for land use panel (before urban overlay)
        pft_map = np.zeros((h, w), dtype=np.int32)
        max_vals = np.full((h, w), -np.inf, dtype=np.float32)
        for pft_num, arr in pft_on_grid.items():
            clean = np.nan_to_num(arr, nan=0.0)
            mask = clean > max_vals
            pft_map[mask] = pft_num
            max_vals[mask] = clean[mask]

        # --- 3-panel PNG: Urban | Land Use (PFTs) | Integrated ---
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))

        # Panel 1: Urban data (0/1/2)
        ax1 = axes[0]
        urban_display = np.zeros((h, w), dtype=np.float32)
        if urban_data is not None:
            urban_display = urban_data.astype(np.float32)
        urban_cmap = mcolors.ListedColormap(["#F0F0F0", "#FFA500", "#FF0000"])
        urban_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], urban_cmap.N)
        ax1.imshow(urban_display, extent=extent, cmap=urban_cmap, norm=urban_norm,
                   interpolation="nearest", aspect="auto")
        ax1.set_title(f"Urban Data\n{ssp} {year}", fontweight="bold")
        u_patches = [
            Patch(facecolor="#F0F0F0", label=f"0: no data ({int(np.sum(urban_display == 0)):,})"),
            Patch(facecolor="#FFA500", label=f"1: non-urban ({int(np.sum(urban_display == 1)):,})"),
            Patch(facecolor="#FF0000", label=f"2: urban ({int(np.sum(urban_display == 2)):,})"),
        ]
        ax1.legend(handles=u_patches, loc="lower right", fontsize=7)

        # Panel 2: Land use dominant PFT (before urban overlay)
        ax2 = axes[1]
        # Use distinct colors per PFT for clarity
        pft_vals = sorted(set(int(v) for v in pft_map[pft_map > 0].flat))
        n_pfts = max(len(pft_vals), 1)
        # Create a color mapping: assign a distinct tab20 color to each PFT present
        pft_color_map = {0: "#F0F0F0"}  # background
        cm = plt.cm.tab20
        for idx, pft_num in enumerate(pft_vals):
            pft_color_map[pft_num] = cm(idx / max(n_pfts, 1))

        # Build RGBA image for PFT map
        pft_rgba = np.zeros((h, w, 4), dtype=np.float32)
        pft_rgba[:, :] = mcolors.to_rgba("#F0F0F0")  # background
        for pft_num, color in pft_color_map.items():
            if pft_num == 0:
                continue
            mask = pft_map == pft_num
            pft_rgba[mask] = mcolors.to_rgba(color)

        ax2.imshow(pft_rgba, extent=extent, interpolation="nearest", aspect="auto")
        ax2.set_title(f"Land Use (Dominant PFT)\n{ssp} {year}", fontweight="bold")
        # Show PFTs in legend sorted by count
        pft_counts = {}
        for v in pft_vals:
            pft_counts[v] = int(np.sum(pft_map == v))
        top_pfts = sorted(pft_counts.items(), key=lambda x: -x[1])[:12]
        pft_patches = [Patch(facecolor=pft_color_map.get(pn, "#808080"),
                             label=f"PFT{pn}: {cnt:,}")
                       for pn, cnt in top_pfts]
        ax2.legend(handles=pft_patches, loc="lower right", fontsize=7)

        # Panel 3: Integrated category map
        ax3 = axes[2]
        cat_names = list(self.cfg.category_pft_mapping.keys())
        cat_colors = {
            0: "#D3D3D3", 1: "#FF0000", 2: "#FFD700", 3: "#90EE90",
            4: "#006400", 5: "#4169E1", 6: "#DEB887",
        }
        cmap_cat = mcolors.ListedColormap([cat_colors[i] for i in range(7)])
        bounds_vals = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        norm = mcolors.BoundaryNorm(bounds_vals, cmap_cat.N)
        ax3.imshow(cat_map, extent=extent, cmap=cmap_cat, norm=norm,
                   interpolation="nearest", aspect="auto")
        ax3.set_title(f"Integrated Categories\n{ssp} {year}", fontweight="bold")

        legend_labels = ["Unclassified"] + cat_names
        legend_patches = []
        for i in range(len(legend_labels)):
            cnt = int(np.sum(cat_map == i))
            legend_patches.append(
                Patch(facecolor=cat_colors[i], label=f"{legend_labels[i]}: {cnt:,}"))
        ax3.legend(handles=legend_patches, loc="lower right", fontsize=7)

        for ax in axes:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        plt.tight_layout()
        png_path = val_dir / f"{prefix}_validation.png"
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        log.info("  Saved validation PNG: %s", png_path.name)

    def _save_ssp_comparison(self, df: pd.DataFrame):
        """Save a summary plot comparing category pixel counts across SSPs.

        Helps verify that SSPs actually produce different category counts,
        and highlights where the urban data's intensity values (1 vs 2) and
        land use differences drive SSP divergence.
        """
        val_dir = self.cfg.output_dir / "validation_maps"
        val_dir.mkdir(exist_ok=True, parents=True)
        name = self.cfg.study_area_name or "study_area"

        cat_cols = [c for c in self.cfg.category_pft_mapping.keys() if c in df.columns]
        ssp_colors = {
            "SSP1": "#2E8B57", "SSP2": "#4169E1", "SSP3": "#FF6347",
            "SSP4": "#FFD700", "SSP5": "#8B008B",
        }

        n = len(cat_cols)
        cols = min(3, n)
        nrows = (n + cols - 1) // cols
        fig, axes = plt.subplots(nrows, cols, figsize=(7 * cols, 5 * nrows), squeeze=False)

        for idx, cat in enumerate(cat_cols):
            ax = axes[idx // cols][idx % cols]
            for ssp in self.cfg.ssps:
                ssp_df = df[df["SSP"] == ssp].sort_values("Year")
                if ssp_df.empty or cat not in ssp_df.columns:
                    continue
                ax.plot(ssp_df["Year"], ssp_df[cat], marker="o", linewidth=2,
                        markersize=4, color=ssp_colors.get(ssp, "gray"), label=ssp)
            ax.set_title(f"{cat} (pixel count)", fontweight="bold")
            ax.set_xlabel("Year")
            ax.set_ylabel("Pixels")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        for idx in range(n, nrows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.suptitle(f"{name} - Category Pixel Counts by SSP (Integration Check)",
                     fontweight="bold", fontsize=14)
        plt.tight_layout()
        out = val_dir / f"{name}_ssp_comparison.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        log.info("SSP comparison plot saved: %s", out.name)

        # Also log the actual differences between SSPs
        last_year = df["Year"].max()
        last_year_df = df[df["Year"] == last_year]
        if len(last_year_df) > 1:
            log.info("=== SSP Comparison for year %d ===", last_year)
            for cat in cat_cols:
                if cat not in last_year_df.columns:
                    continue
                vals = last_year_df.set_index("SSP")[cat]
                diff = vals.max() - vals.min()
                pct = (diff / vals.mean() * 100) if vals.mean() > 0 else 0
                log.info("  %s: range=%d (%.1f%% spread) %s",
                         cat, diff, pct,
                         dict(vals))
