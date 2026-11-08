"""Central configuration for the LCZ Projection pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class PipelineConfig:
    # --- Input paths (global datasets) ---
    global_urban_dir: Path = Path(r"E:\LCZ Projection\Global_Urban")
    global_landuse_dir: Path = Path(r"E:\LCZ Projection\Global_LU")
    lcz_data_dir: Path = Path("LCZ_Data")  # Directory with historical LCZ TIFs
    output_dir: Path = Path("lcz_projection_results")

    # --- Scenario dimensions ---
    ssps: List[str] = field(default_factory=lambda: ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"])
    projection_years: List[int] = field(default_factory=lambda: list(range(2030, 2101, 10)))
    historical_years: List[int] = field(default_factory=lambda: [1986, 1991, 1996, 2001, 2006, 2011, 2016])

    # --- Auto-detected at runtime (from LCZ files) ---
    study_area_name: Optional[str] = None       # e.g. "Vancouver", derived from LCZ filenames
    study_area_bounds_wgs84: Optional[dict] = None  # {min_lat, max_lat, min_lon, max_lon}
    study_area_crs: Optional[str] = None         # CRS of the LCZ files
    study_area_transform: Optional[object] = None
    study_area_shape: Optional[tuple] = None

    # --- File naming patterns (auto-detected, can be overridden) ---
    # LCZ files: discovered automatically by scanning lcz_data_dir for *.tif
    # Urban files: global_urban_dir/SSP{n}/global_SSP{n}_{year}.tif
    # LandUse files: global_landuse_dir/GCAM_Demeter_LU_ssp{n}_*_{year}.nc

    # --- PFT-to-category grouping (from GCAM-Demeter land use data) ---
    category_pft_mapping: Dict[str, List[int]] = field(default_factory=lambda: {
        "Urban Land": [31],
        "Crops": list(range(15, 31)),
        "Grass/Shrub": [9, 10, 11, 12, 13, 14],
        "Forest": list(range(1, 9)),
        "Water": [0],
        "Barren": [32],
    })

    # --- Category-to-LCZ mapping (which LCZ types compose each category) ---
    category_lcz_mapping: Dict[str, List[int]] = field(default_factory=lambda: {
        "Urban Land": list(range(1, 11)),   # LCZ 1-10
        "Forest": [11, 12],                 # LCZ A, B
        "Grass/Shrub": [13],                # LCZ C
        "Crops": [14],                      # LCZ D
        "Barren": [15, 16],                 # LCZ E, F
        "Water": [17],                      # LCZ G
    })

    # --- LCZ descriptions ---
    lcz_descriptions: Dict[int, str] = field(default_factory=lambda: {
        1: "Compact high-rise", 2: "Compact midrise", 3: "Compact low-rise",
        4: "Open high-rise", 5: "Open midrise", 6: "Open low-rise",
        7: "Lightweight low-rise", 8: "Large low-rise", 9: "Sparsely built",
        10: "Heavy industry", 11: "Dense trees (A)", 12: "Scattered trees (B)",
        13: "Bush/scrub (C)", 14: "Low plants (D)", 15: "Bare rock/paved (E)",
        16: "Bare soil/sand (F)", 17: "Water (G)",
    })

    # --- Projection parameters ---
    min_pixel_threshold: int = 15
    auto_bounds_percentile: Tuple[float, float] = (5.0, 95.0)
    auto_bounds_buffer: float = 0.10
    auto_bounds_min_spread: float = 0.02
    ensemble_seed: int = 42
