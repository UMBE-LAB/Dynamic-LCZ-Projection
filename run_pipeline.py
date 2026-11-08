"""LCZ Projection Pipeline - Orchestrator.

Projects future LCZ pixel counts under SSP scenarios for any study area.
The study area is auto-detected from historical LCZ files.

Usage:
    python run_pipeline.py --lcz-dir path/to/LCZ_files
    python run_pipeline.py --lcz-dir Vancouver_LCZs --output-dir results
    python run_pipeline.py --skip-transition
"""

import argparse
import logging
from pathlib import Path

from config import PipelineConfig
from data_integration import detect_study_area, CategoryPixelCounter
from lcz_composition import LCZCompositionProjector
from lcz_projector import LCZPixelProjector
from transition_matrix import TransitionAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lcz_projection")


def run(config: PipelineConfig = None, skip_transition: bool = False,
        save_validation: bool = True):
    """Execute the full pipeline."""
    if config is None:
        config = PipelineConfig()

    # Step 0: Auto-detect study area from LCZ files (skip if already done)
    if not config.study_area_name:
        log.info("=" * 60)
        log.info("Step 0: Detecting study area from LCZ files")
        log.info("=" * 60)
        detect_study_area(config)
    config.output_dir.mkdir(exist_ok=True, parents=True)

    name = config.study_area_name

    # Step 1: Integrate urban + land use data -> category pixel counts
    log.info("=" * 60)
    log.info("Step 1: Computing category pixel counts for %s", name)
    log.info("=" * 60)
    counter = CategoryPixelCounter(config)
    category_pixels = counter.run(save_validation=save_validation)
    cat_csv = config.output_dir / f"{name}_category_pixel_counts.csv"
    category_pixels.to_csv(cat_csv, index=False)
    log.info("Saved %s (%d rows)", cat_csv.name, len(category_pixels))

    # Step 2: Analyze historical LCZ composition + project forward
    log.info("=" * 60)
    log.info("Step 2: Projecting LCZ compositions from historical data")
    log.info("=" * 60)
    compositor = LCZCompositionProjector(config)
    lcz_fractions = compositor.run()
    frac_csv = config.output_dir / f"{name}_lcz_composition_fractions.csv"
    lcz_fractions.to_csv(frac_csv, index=False)
    log.info("Saved %s (%d rows)", frac_csv.name, len(lcz_fractions))
    compositor.create_plots(lcz_fractions, config.output_dir)

    # Step 3: Multiply -> final LCZ pixel projections
    log.info("=" * 60)
    log.info("Step 3: Computing final LCZ pixel projections")
    log.info("=" * 60)
    projector = LCZPixelProjector(config)
    final = projector.run(category_pixels, lcz_fractions)
    projector.save_results(final, config.output_dir)
    projector.create_plots(final, config.output_dir)
    log.info("Final projections: %d rows across %d SSPs",
             len(final), final["SSP"].nunique())

    # Step 4 (optional): Transition probability matrix
    if not skip_transition:
        log.info("=" * 60)
        log.info("Step 4: Computing LCZ transition probabilities (optional)")
        log.info("=" * 60)
        try:
            analyzer = TransitionAnalyzer(config)
            matrix = analyzer.run()
            analyzer.save_results(matrix, config.output_dir)
        except Exception as e:
            log.warning("Transition analysis skipped: %s", e)

    log.info("=" * 60)
    log.info("Pipeline complete for %s. Results in: %s", name, config.output_dir)
    log.info("=" * 60)
    return final


def main():
    parser = argparse.ArgumentParser(
        description="LCZ Projection Pipeline - project future LCZ pixel counts")
    parser.add_argument("--lcz-dir", type=Path, default=None,
                        help="Directory with historical LCZ TIF files (defines study area)")
    parser.add_argument("--global-urban-dir", type=Path, default=None,
                        help="Directory with global urban projection TIFs")
    parser.add_argument("--global-landuse-dir", type=Path, default=None,
                        help="Directory with global land use NetCDF files")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: auto from study area name)")
    parser.add_argument("--skip-transition", action="store_true",
                        help="Skip the optional transition matrix analysis")
    parser.add_argument("--no-validation", action="store_true",
                        help="Skip saving validation maps")
    args = parser.parse_args()

    config = PipelineConfig()
    if args.lcz_dir:
        config.lcz_data_dir = args.lcz_dir
    if args.global_urban_dir:
        config.global_urban_dir = args.global_urban_dir
    if args.global_landuse_dir:
        config.global_landuse_dir = args.global_landuse_dir

    # Detect study area first so output_dir can be overridden
    detect_study_area(config)
    if args.output_dir:
        config.output_dir = args.output_dir

    run(config, skip_transition=args.skip_transition,
        save_validation=not args.no_validation)


if __name__ == "__main__":
    main()
