"""Step 2: Analyze historical LCZ composition and project to future years."""

import logging
import re
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

from config import PipelineConfig
from data_integration import _discover_lcz_files

log = logging.getLogger(__name__)


class LCZCompositionProjector:
    """Analyzes LCZ composition within each land category from historical data,
    then projects compositions forward using an ensemble of methods.

    Output: DataFrame with columns [Category, LCZ_Type, Year, Fraction, Data_Type]
    """

    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def run(self) -> pd.DataFrame:
        """Run historical analysis + projection. Returns composition fractions."""
        # Load historical LCZ maps
        lcz_maps = self._load_historical_maps()
        if not lcz_maps:
            raise FileNotFoundError("No historical LCZ maps loaded")

        # Compute historical compositions per category
        historical = self._compute_historical_compositions(lcz_maps)

        # Filter LCZ types by pixel threshold
        valid_lcz = self._filter_valid_lcz(lcz_maps)
        log.info("Valid LCZ types (>=%d pixels): %s", self.cfg.min_pixel_threshold, sorted(valid_lcz))

        # Build and project compositions for each category
        rows = []
        for cat_name, lcz_types in self.cfg.category_lcz_mapping.items():
            cat_lcz = [t for t in lcz_types if t in valid_lcz]
            if not cat_lcz:
                log.warning("No valid LCZ types in %s", cat_name)
                continue

            # Historical rows
            for year in self.cfg.historical_years:
                for lcz_id in cat_lcz:
                    frac = historical.get((cat_name, lcz_id, year), 0.0)
                    rows.append({"Category": cat_name, "LCZ_Type": lcz_id,
                                 "Year": year, "Fraction": frac, "Data_Type": "historical"})

            # Project
            projected = self._project_category(cat_name, cat_lcz, historical)
            for year in self.cfg.projection_years:
                for lcz_id in cat_lcz:
                    frac = projected.get((lcz_id, year), 0.0)
                    rows.append({"Category": cat_name, "LCZ_Type": lcz_id,
                                 "Year": year, "Fraction": frac, "Data_Type": "projected"})

        return pd.DataFrame(rows)

    def create_plots(self, df: pd.DataFrame, output_dir: Path):
        """Create composition plots: historical (solid) + projected (dotted) per category."""
        categories = df["Category"].unique()
        n = len(categories)
        cols = min(3, n)
        nrows = (n + cols - 1) // cols
        fig, axes = plt.subplots(nrows, cols, figsize=(7 * cols, 5 * nrows), squeeze=False)

        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for idx, cat in enumerate(categories):
            ax = axes[idx // cols][idx % cols]
            cat_df = df[df["Category"] == cat]
            lcz_types = sorted(cat_df["LCZ_Type"].unique())

            for i, lcz_id in enumerate(lcz_types):
                color = colors[i % len(colors)]
                lcz_label = self.cfg.lcz_descriptions.get(lcz_id, f"LCZ {lcz_id}")

                hist = cat_df[(cat_df["LCZ_Type"] == lcz_id) & (cat_df["Data_Type"] == "historical")]
                hist = hist.sort_values("Year")
                ax.plot(hist["Year"], hist["Fraction"] * 100, color=color,
                        linewidth=2, marker="o", markersize=4, label=lcz_label)

                proj = cat_df[(cat_df["LCZ_Type"] == lcz_id) & (cat_df["Data_Type"] == "projected")]
                proj = proj.sort_values("Year")
                if not proj.empty and not hist.empty:
                    # Bridge from last historical to first projection
                    bridge_y = [hist["Year"].iloc[-1], proj["Year"].iloc[0]]
                    bridge_f = [hist["Fraction"].iloc[-1] * 100, proj["Fraction"].iloc[0] * 100]
                    ax.plot(bridge_y, bridge_f, color=color, linewidth=2, linestyle=":")
                ax.plot(proj["Year"], proj["Fraction"] * 100, color=color,
                        linewidth=2, linestyle=":")

            ax.set_title(cat, fontweight="bold")
            ax.set_xlabel("Year")
            ax.set_ylabel("% within category")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=self.cfg.historical_years[-1], color="gray", linestyle="--", alpha=0.5)

        # Hide unused axes
        for idx in range(n, nrows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        title = f"{self.cfg.study_area_name or ''} LCZ Composition (Historical + Projected)".strip()
        fig.suptitle(title, fontweight="bold", fontsize=14)
        plt.tight_layout()
        name = self.cfg.study_area_name or "study_area"
        out = output_dir / f"{name}_lcz_composition_plots.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info("Composition plot saved: %s", out.name)

    # ---- Historical data loading ----

    def _load_historical_maps(self) -> dict:
        """Load historical LCZ rasters. Auto-discovers files by year."""
        lcz_files = _discover_lcz_files(self.cfg.lcz_data_dir)
        maps = {}
        for fp in lcz_files:
            m = re.search(r"(\d{4})", fp.stem)
            if not m:
                continue
            year = int(m.group(1))
            if year not in self.cfg.historical_years:
                continue
            with rasterio.open(fp) as src:
                maps[year] = src.read(1)
            log.info("Loaded %s (%s)", fp.name, maps[year].shape)
        return maps

    def _filter_valid_lcz(self, lcz_maps: dict) -> set:
        """Return set of LCZ type IDs that have >= min_pixel_threshold across all years."""
        pixel_counts = {}
        for data in lcz_maps.values():
            vals, counts = np.unique(data[data > 0], return_counts=True)
            for v, c in zip(vals, counts):
                pixel_counts[int(v)] = max(pixel_counts.get(int(v), 0), int(c))
        return {k for k, v in pixel_counts.items() if v >= self.cfg.min_pixel_threshold}

    def _compute_historical_compositions(self, lcz_maps: dict) -> dict:
        """Compute LCZ fractions within each category for each year.

        Returns dict keyed by (category_name, lcz_id, year) -> fraction.
        """
        result = {}
        for year, data in lcz_maps.items():
            vals, counts = np.unique(data[data > 0], return_counts=True)
            count_map = dict(zip(vals.astype(int), counts.astype(int)))

            for cat_name, lcz_types in self.cfg.category_lcz_mapping.items():
                cat_total = sum(count_map.get(t, 0) for t in lcz_types)
                for lcz_id in lcz_types:
                    frac = count_map.get(lcz_id, 0) / cat_total if cat_total > 0 else 0.0
                    result[(cat_name, lcz_id, year)] = frac
        return result

    # ---- Projection ----

    def _project_category(self, cat_name: str, lcz_types: list, historical: dict) -> dict:
        """Project LCZ compositions for one category. Returns {(lcz_id, year): fraction}."""
        years = sorted(self.cfg.historical_years)
        n_lcz = len(lcz_types)

        # Single-LCZ categories are trivially 100%
        if n_lcz == 1:
            lcz_id = lcz_types[0]
            return {(lcz_id, y): 1.0 for y in self.cfg.projection_years}

        # Build historical composition matrix: [n_years x n_lcz]
        comp = np.zeros((len(years), n_lcz))
        for t, year in enumerate(years):
            for i, lcz_id in enumerate(lcz_types):
                comp[t, i] = historical.get((cat_name, lcz_id, year), 0.0)
            # Normalize row
            row_sum = comp[t, :].sum()
            if row_sum > 0:
                comp[t, :] /= row_sum

        # Compute auto-bounds for each LCZ from historical data
        auto_bounds = {}
        for i, lcz_id in enumerate(lcz_types):
            auto_bounds[lcz_id] = self._compute_auto_bounds(comp[:, i])

        log.info("  %s auto-bounds: %s", cat_name,
                 {self.cfg.lcz_descriptions.get(k, f"LCZ {k}"): (f"{v[0]:.3f}", f"{v[1]:.3f}")
                  for k, v in auto_bounds.items()})

        # Project each LCZ type independently
        projected = np.zeros((len(self.cfg.projection_years), n_lcz))
        for i, lcz_id in enumerate(lcz_types):
            series = comp[:, i]
            bounds = auto_bounds[lcz_id]
            projected[:, i] = self._ensemble_project(series, years, bounds)

        # Normalize compositions to sum to 1 per time step
        for t in range(len(self.cfg.projection_years)):
            total = projected[t, :].sum()
            if total > 0:
                projected[t, :] /= total

        # Ensure perfect continuity: first projection year matches last historical
        projected[0, :] = comp[-1, :]

        # Re-normalize after continuity fix
        for t in range(len(self.cfg.projection_years)):
            total = projected[t, :].sum()
            if total > 0:
                projected[t, :] /= total

        # Build output dict
        result = {}
        for t, year in enumerate(self.cfg.projection_years):
            for i, lcz_id in enumerate(lcz_types):
                result[(lcz_id, year)] = float(projected[t, i])
        return result

    def _compute_auto_bounds(self, series: np.ndarray) -> tuple:
        """Derive realistic bounds from historical composition ratios.

        Uses percentile-based range with a buffer for extrapolation.
        """
        lo_pct, hi_pct = self.cfg.auto_bounds_percentile
        lo = float(np.percentile(series, lo_pct))
        hi = float(np.percentile(series, hi_pct))
        spread = hi - lo
        buf = max(spread * self.cfg.auto_bounds_buffer, self.cfg.auto_bounds_min_spread)
        return (max(0.0, lo - buf), min(1.0, hi + buf))

    def _ensemble_project(self, series: np.ndarray, years: list, bounds: tuple) -> np.ndarray:
        """Combine three projection methods into an ensemble."""
        p1 = self._trend_momentum(series, bounds)
        p2 = self._volatility_walk(series, bounds)
        p3 = self._pattern_projection(series, bounds)

        # Adaptive weights based on series characteristics
        weights = self._adaptive_weights(series)
        ensemble = weights[0] * p1 + weights[1] * p2 + weights[2] * p3
        return np.clip(ensemble, bounds[0], bounds[1])

    def _trend_momentum(self, series: np.ndarray, bounds: tuple) -> np.ndarray:
        """Project using trend with momentum and adaptive damping."""
        n_proj = len(self.cfg.projection_years)
        if len(series) < 3:
            return np.full(n_proj, series[-1])

        overall_trend = (series[-1] - series[0]) / len(series)
        recent_trend = (series[-1] - series[-3]) / 2
        combined = 0.7 * recent_trend + 0.3 * overall_trend

        projected = []
        val = series[-1]
        for i in range(n_proj):
            dist_upper = bounds[1] - val
            dist_lower = val - bounds[0]
            damping = max(0.5, min(dist_upper, dist_lower) * 2)
            val = val + combined * (i + 1) * damping / (i + 2)
            val = np.clip(val, bounds[0], bounds[1])
            projected.append(val)
        return np.array(projected)

    def _volatility_walk(self, series: np.ndarray, bounds: tuple) -> np.ndarray:
        """Project using drift + volatility random walk."""
        n_proj = len(self.cfg.projection_years)
        if len(series) < 2:
            return np.full(n_proj, series[-1])

        changes = np.diff(series)
        drift = np.mean(changes)
        vol = max(np.std(series), 0.005)

        np.random.seed(self.cfg.ensemble_seed)
        projected = []
        val = series[-1]
        hist_mean = np.mean(series)
        for i in range(n_proj):
            shock = np.random.normal(0, vol)
            reversion = 0.05 * (hist_mean - val)
            val = val + drift * (0.95 ** i) + shock + reversion
            val = np.clip(val, bounds[0], bounds[1])
            projected.append(val)
        return np.array(projected)

    def _pattern_projection(self, series: np.ndarray, bounds: tuple) -> np.ndarray:
        """Project based on pattern continuation / gentle trend stabilization."""
        n_proj = len(self.cfg.projection_years)
        if len(series) < 2:
            return np.full(n_proj, series[-1])

        trend = series[-1] - series[-2] if len(series) >= 2 else 0
        projected = []
        val = series[-1]
        for i in range(n_proj):
            val = val + trend * 0.3 * (i + 1) / (i + 2)  # Diminishing trend
            val = np.clip(val, bounds[0], bounds[1])
            projected.append(val)
        return np.array(projected)

    def _adaptive_weights(self, series: np.ndarray) -> list:
        """Adaptive ensemble weights based on series characteristics."""
        if len(series) < 3:
            return [0.6, 0.3, 0.1]
        vol = np.std(series)
        trend = abs((series[-1] - series[0]) / len(series))
        if trend > 0.05:
            return [0.7, 0.2, 0.1]
        elif vol > 0.1:
            return [0.3, 0.5, 0.2]
        return [0.4, 0.3, 0.3]
