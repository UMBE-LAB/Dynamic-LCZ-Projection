"""Step 3: Multiply category pixels by LCZ fractions to get final pixel projections."""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from datetime import datetime

from config import PipelineConfig

log = logging.getLogger(__name__)


class LCZPixelProjector:
    """Combines category pixel counts with LCZ composition fractions
    to produce final LCZ pixel count projections per SSP/year."""

    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def run(self, category_pixels: pd.DataFrame, lcz_fractions: pd.DataFrame) -> pd.DataFrame:
        """Compute LCZ pixel counts = category_pixels * lcz_fraction.

        Parameters:
            category_pixels: from CategoryPixelCounter (columns: SSP, Year, <categories>, Total)
            lcz_fractions:   from LCZCompositionProjector (columns: Category, LCZ_Type, Year, Fraction, Data_Type)

        Returns DataFrame with columns:
            SSP, Year, LCZ_Type, LCZ_Description, Category, Pixel_Count, Pct_of_Total
        """
        # Use only projected fractions (not historical)
        proj_fracs = lcz_fractions[lcz_fractions["Data_Type"] == "projected"]

        rows = []
        for _, px_row in category_pixels.iterrows():
            ssp = px_row["SSP"]
            year = px_row["Year"]
            total_pixels = 0
            year_rows = []

            for cat_name, lcz_types in self.cfg.category_lcz_mapping.items():
                if cat_name not in px_row.index:
                    continue
                cat_pixels = px_row[cat_name]

                for lcz_id in lcz_types:
                    match = proj_fracs[
                        (proj_fracs["Category"] == cat_name) &
                        (proj_fracs["LCZ_Type"] == lcz_id) &
                        (proj_fracs["Year"] == year)
                    ]
                    frac = float(match["Fraction"].iloc[0]) if len(match) > 0 else 0.0
                    pixel_count = int(round(cat_pixels * frac))
                    total_pixels += pixel_count
                    year_rows.append({
                        "SSP": ssp, "Year": year,
                        "LCZ_Type": lcz_id,
                        "LCZ_Description": self.cfg.lcz_descriptions.get(lcz_id, f"LCZ {lcz_id}"),
                        "Category": cat_name,
                        "Pixel_Count": pixel_count,
                    })

            for r in year_rows:
                r["Pct_of_Total"] = round(r["Pixel_Count"] / total_pixels * 100, 2) if total_pixels > 0 else 0.0
            rows.extend(year_rows)

        return pd.DataFrame(rows)

    def save_results(self, df: pd.DataFrame, output_dir: Path):
        """Save final projections as CSV and multi-sheet Excel."""
        output_dir.mkdir(exist_ok=True, parents=True)

        name = self.cfg.study_area_name or "study_area"

        # CSV
        csv_path = output_dir / f"{name}_lcz_pixel_projections.csv"
        df.to_csv(csv_path, index=False)
        log.info("Saved %s", csv_path.name)

        # Excel — use LCZ_1, LCZ_2, ... as column headers
        xlsx_path = output_dir / f"{name}_lcz_pixel_projections.xlsx"
        df = df.copy()
        df["LCZ_Label"] = df["LCZ_Type"].apply(lambda x: f"LCZ_{x}")

        # Column order: LCZ_1, LCZ_2, ..., LCZ_17 (numeric sort)
        lcz_col_order = [f"LCZ_{i}" for i in sorted(df["LCZ_Type"].unique())]

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            # Per-SSP pivot sheets
            for ssp in self.cfg.ssps:
                ssp_df = df[df["SSP"] == ssp]
                if ssp_df.empty:
                    continue
                pivot = ssp_df.pivot_table(
                    index="Year", columns="LCZ_Label",
                    values="Pixel_Count", aggfunc="sum"
                ).fillna(0).astype(int)
                pivot = pivot[[c for c in lcz_col_order if c in pivot.columns]]
                pivot.to_excel(writer, sheet_name=ssp)

            # Summary: all SSPs side by side
            summary = df.pivot_table(
                index=["Year", "SSP"], columns="LCZ_Label",
                values="Pixel_Count", aggfunc="sum"
            ).fillna(0).astype(int).reset_index()
            ordered_cols = ["Year", "SSP"] + [c for c in lcz_col_order if c in summary.columns]
            summary = summary[ordered_cols]
            summary.to_excel(writer, sheet_name="Summary", index=False)

            # Metadata
            meta = pd.DataFrame([
                ["Pipeline", "LCZ Projection Pipeline"],
                ["Study Area", self.cfg.study_area_name or "Unknown"],
                ["Study Area CRS", self.cfg.study_area_crs or "Unknown"],
                ["Date", datetime.now().strftime("%Y-%m-%d %H:%M")],
                ["SSPs", ", ".join(self.cfg.ssps)],
                ["Projection Years", f"{self.cfg.projection_years[0]}-{self.cfg.projection_years[-1]}"],
                ["Historical Years", f"{self.cfg.historical_years[0]}-{self.cfg.historical_years[-1]}"],
                ["Min Pixel Threshold", str(self.cfg.min_pixel_threshold)],
                ["Auto-bounds Percentile", str(self.cfg.auto_bounds_percentile)],
                ["Auto-bounds Buffer", str(self.cfg.auto_bounds_buffer)],
            ], columns=["Parameter", "Value"])
            meta.to_excel(writer, sheet_name="Metadata", index=False)

        log.info("Saved %s", xlsx_path.name)

    def create_plots(self, df: pd.DataFrame, output_dir: Path):
        """Create projection plots: individual LCZ pixel counts per SSP."""
        ssp_colors = {
            "SSP1": "#2E8B57", "SSP2": "#4169E1", "SSP3": "#FF6347",
            "SSP4": "#FFD700", "SSP5": "#8B008B",
        }
        name = self.cfg.study_area_name or "study_area"

        # --- Plot 1: Individual LCZ types, one subplot per LCZ, lines per SSP ---
        lcz_types = sorted(df["LCZ_Type"].unique())
        n = len(lcz_types)
        cols = min(4, n)
        nrows = (n + cols - 1) // cols
        fig, axes = plt.subplots(nrows, cols, figsize=(6 * cols, 4.5 * nrows), squeeze=False)

        for idx, lcz_id in enumerate(lcz_types):
            ax = axes[idx // cols][idx % cols]
            lcz_df = df[df["LCZ_Type"] == lcz_id]
            agg = lcz_df.groupby(["SSP", "Year"])["Pixel_Count"].sum().reset_index()

            for ssp in self.cfg.ssps:
                ssp_data = agg[agg["SSP"] == ssp].sort_values("Year")
                if ssp_data.empty:
                    continue
                ax.plot(ssp_data["Year"], ssp_data["Pixel_Count"],
                        marker="o", linewidth=2, markersize=3,
                        color=ssp_colors.get(ssp, "gray"), label=ssp)

            desc = self.cfg.lcz_descriptions.get(lcz_id, f"LCZ {lcz_id}")
            ax.set_title(f"LCZ {lcz_id}: {desc}", fontweight="bold", fontsize=10)
            ax.set_xlabel("Year")
            ax.set_ylabel("Pixel Count")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        for idx in range(n, nrows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.suptitle(f"{name} - Projected LCZ Pixel Counts by Type and SSP",
                     fontweight="bold", fontsize=14)
        plt.tight_layout()
        out = output_dir / f"{name}_lcz_projection_plots.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info("LCZ projection plot saved: %s", out.name)

        # --- Plot 2: Category-level summary (one subplot per category) ---
        self._create_category_summary_plot(df, output_dir, ssp_colors)

        # --- Plot 3: Per-SSP heatmaps ---
        self._create_heatmaps(df, output_dir, ssp_colors)

    def _create_category_summary_plot(self, df: pd.DataFrame, output_dir: Path, ssp_colors: dict):
        """Category-level summary: total pixels per category across SSPs."""
        name = self.cfg.study_area_name or "study_area"
        categories = sorted(df["Category"].unique())
        n = len(categories)
        cols = min(3, n)
        nrows = (n + cols - 1) // cols
        fig, axes = plt.subplots(nrows, cols, figsize=(7 * cols, 5 * nrows), squeeze=False)

        for idx, cat in enumerate(categories):
            ax = axes[idx // cols][idx % cols]
            cat_df = df[df["Category"] == cat]
            agg = cat_df.groupby(["SSP", "Year"])["Pixel_Count"].sum().reset_index()

            for ssp in self.cfg.ssps:
                ssp_data = agg[agg["SSP"] == ssp].sort_values("Year")
                if ssp_data.empty:
                    continue
                ax.plot(ssp_data["Year"], ssp_data["Pixel_Count"],
                        marker="o", linewidth=2, markersize=4,
                        color=ssp_colors.get(ssp, "gray"), label=ssp)

            ax.set_title(cat, fontweight="bold")
            ax.set_xlabel("Year")
            ax.set_ylabel("Pixel Count")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        for idx in range(n, nrows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.suptitle(f"{name} - Category Pixel Totals by SSP",
                     fontweight="bold", fontsize=14)
        plt.tight_layout()
        out = output_dir / f"{name}_category_summary_plots.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info("Category summary plot saved: %s", out.name)

    def _create_heatmaps(self, df: pd.DataFrame, output_dir: Path, ssp_colors: dict):
        """Create heatmap: Year x LCZ_Type pixel counts, one per SSP."""
        ssps_with_data = [s for s in self.cfg.ssps if s in df["SSP"].values]
        n = len(ssps_with_data)
        if n == 0:
            return

        fig, axes = plt.subplots(1, n, figsize=(6 * n, 8), squeeze=False)
        for i, ssp in enumerate(ssps_with_data):
            ax = axes[0][i]
            ssp_df = df[df["SSP"] == ssp]
            pivot = ssp_df.pivot_table(
                index="LCZ_Description", columns="Year",
                values="Pixel_Count", aggfunc="sum"
            ).fillna(0)

            im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)
            ax.set_title(ssp, fontweight="bold")
            fig.colorbar(im, ax=ax, shrink=0.6, label="Pixels")

        name = self.cfg.study_area_name or "study_area"
        fig.suptitle(f"{name} - LCZ Pixel Count Heatmaps by SSP", fontweight="bold", fontsize=14)
        plt.tight_layout()
        out = output_dir / f"{name}_lcz_projection_heatmaps.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info("Heatmap saved: %s", out.name)
