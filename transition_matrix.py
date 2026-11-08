"""Optional Step 4: Markov transition probability matrix from historical LCZ maps."""

import logging
import re
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path

from config import PipelineConfig
from data_integration import _discover_lcz_files

log = logging.getLogger(__name__)


class TransitionAnalyzer:
    """Computes pixel-by-pixel LCZ transition probabilities from historical data."""

    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def run(self) -> pd.DataFrame:
        """Load historical maps, compute transitions, return probability matrix as DataFrame."""
        maps = self._load_maps()
        if len(maps) < 2:
            raise ValueError("Need at least 2 historical maps for transitions")

        transitions = self._count_transitions(maps)
        matrix = self._build_probability_matrix(transitions)
        return matrix

    def save_results(self, matrix: pd.DataFrame, output_dir: Path):
        """Save transition matrix to Excel and heatmap plot."""
        output_dir.mkdir(exist_ok=True, parents=True)

        name = self.cfg.study_area_name or "study_area"

        # Excel
        xlsx = output_dir / f"{name}_lcz_transition_matrix.xlsx"
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            matrix.to_excel(writer, sheet_name="Transition_Probabilities")

            # Persistence summary
            diag = np.diag(matrix.values)
            persistence = pd.DataFrame({
                "LCZ_Class": matrix.index,
                "Persistence": diag,
            }).sort_values("Persistence", ascending=False)
            persistence.to_excel(writer, sheet_name="Persistence", index=False)

            # Top transitions (off-diagonal > 1%)
            off_diag = []
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    if i != j and matrix.iloc[i, j] > 0.01:
                        off_diag.append({
                            "From": matrix.index[i],
                            "To": matrix.columns[j],
                            "Probability": matrix.iloc[i, j],
                        })
            if off_diag:
                pd.DataFrame(off_diag).sort_values("Probability", ascending=False).to_excel(
                    writer, sheet_name="Top_Transitions", index=False)

        log.info("Saved %s", xlsx.name)

        # Heatmap
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        sns.heatmap(matrix, annot=True, fmt=".3f", cmap="viridis", ax=axes[0],
                    cbar_kws={"label": "Probability"})
        axes[0].set_title(f"{name} - LCZ Transition Probabilities", fontweight="bold")
        axes[0].set_xlabel("To")
        axes[0].set_ylabel("From")

        labels = [lbl.replace("LCZ ", "") for lbl in matrix.index]
        bars = axes[1].bar(range(len(diag)), diag, color="skyblue", alpha=0.7)
        axes[1].set_title("Persistence Probabilities", fontweight="bold")
        axes[1].set_xticks(range(len(diag)))
        axes[1].set_xticklabels(labels, rotation=45)
        axes[1].set_ylim(0, 1.0)
        axes[1].set_ylabel("P(stay same)")
        axes[1].grid(True, alpha=0.3)
        for bar, val in zip(bars, diag):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        out = output_dir / f"{name}_lcz_transition_heatmap.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved %s", out.name)

    # ---- Internal ----

    def _load_maps(self) -> dict:
        """Auto-discover and load historical LCZ maps."""
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
        log.info("Loaded %d historical maps for transition analysis", len(maps))
        return maps

    def _count_transitions(self, maps: dict) -> dict:
        """Count pixel transitions between consecutive years."""
        years = sorted(maps.keys())

        # Find valid LCZ classes (>= threshold)
        pixel_counts = defaultdict(int)
        for data in maps.values():
            vals, counts = np.unique(data[data > 0], return_counts=True)
            for v, c in zip(vals, counts):
                pixel_counts[int(v)] += int(c)
        valid = {k for k, v in pixel_counts.items() if v >= self.cfg.min_pixel_threshold}

        transitions = defaultdict(lambda: defaultdict(int))
        for i in range(len(years) - 1):
            d1, d2 = maps[years[i]], maps[years[i + 1]]
            mask = (d1 > 0) & (d2 > 0)
            for f, t in zip(d1[mask].flat, d2[mask].flat):
                fi, ti = int(f), int(t)
                if fi in valid and ti in valid:
                    transitions[fi][ti] += 1

        return transitions

    def _build_probability_matrix(self, transitions: dict) -> pd.DataFrame:
        """Convert transition counts to probability matrix."""
        classes = sorted(set(transitions.keys()) | {t for d in transitions.values() for t in d})
        n = len(classes)
        mat = np.zeros((n, n))
        idx = {c: i for i, c in enumerate(classes)}

        for from_c in classes:
            total = sum(transitions[from_c].values())
            if total > 0:
                for to_c in classes:
                    mat[idx[from_c], idx[to_c]] = transitions[from_c][to_c] / total

        labels = [f"LCZ {c}" for c in classes]
        return pd.DataFrame(mat, index=labels, columns=labels)
