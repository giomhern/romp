"""
Simulated Contiguous Rain Area (CRA) verification demo.

This script is intentionally disconnected from the main ROMP workflow. It uses
idealized rainfall fields with known errors to exercise the core CRA ideas from
Ebert and Gallus (2009): best-fit displacement, volume error, and pattern error.

Run from the repo root:
    python demo/cra/demo_simulated_cra.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from momp.metrics.cra import CraResult, cra_decomposition, shift_field


RAIN_LEVELS = [1.0, 12.7, 25.4]
OBS_COLOR = "#202020"
FCST_COLOR = "#1f77b4"
SHIFTED_COLOR = "#d62728"


def make_paper_ellipse_field(
    shape: tuple[int, int],
    *,
    center: tuple[float, float],
    dimensions: tuple[float, float],
    intensity: float = 12.7,
    core_dimensions: tuple[float, float] = (20.0, 80.0),
    core_intensity: float = 25.4,
    core_offset_x: float = 10.0,
) -> np.ndarray:
    """
    Create the idealized geometric rain field from Ebert and Gallus (2009).

    The paper's observed field is a north-south ellipse of dimension
    50 x 200 grid points at 12.7 mm h-1 with an embedded 20 x 80 grid-point
    heavy-rain ellipse at 25.4 mm h-1, centered 10 points east of the parent.
    Dimensions are passed as (east-west width, north-south height).
    """
    y, x = np.indices(shape, dtype=float)
    cy, cx = center
    width, height = dimensions
    core_width, core_height = core_dimensions

    outer = ((x - cx) / (width / 2.0)) ** 2 + ((y - cy) / (height / 2.0)) ** 2 <= 1.0
    core_cx = cx + core_offset_x
    core = ((x - core_cx) / (core_width / 2.0)) ** 2 + (
        (y - cy) / (core_height / 2.0)
    ) ** 2 <= 1.0

    field = np.zeros(shape, dtype=float)
    field[outer] = intensity
    field[core] = core_intensity
    return field


def make_cases(shape: tuple[int, int]) -> dict[str, tuple[np.ndarray, np.ndarray, int, int]]:
    """Return the paper's idealized geometric cases with known forecast errors."""
    obs = make_paper_ellipse_field(
        shape,
        center=(170, 150),
        dimensions=(50, 200),
    )

    cases = {
        "geom001_shift_50_east": (obs, shift_field(obs, 0, 50), 50, 0),
        "geom003_stretched_200x200": (
            obs,
            make_paper_ellipse_field(
                shape,
                center=(170, 275),
                dimensions=(200, 200),
                core_dimensions=(80, 80),
            ),
            125,
            0,
        ),
        "geom004_wrong_aspect_200x50": (
            obs,
            make_paper_ellipse_field(
                shape,
                center=(170, 275),
                dimensions=(200, 50),
                core_dimensions=(80, 20),
            ),
            125,
            0,
        ),
    }

    return cases


def masked_centroid(field: np.ndarray, threshold: float) -> tuple[float, float] | None:
    """Return x, y centroid of rain above threshold in array/display coordinates."""
    mask = np.asarray(field) >= threshold
    if not np.any(mask):
        return None

    y, x = np.nonzero(mask)
    return float(x.mean()), float(y.mean())


def add_rain_contours(
    ax: plt.Axes,
    data: np.ndarray,
    *,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.1,
    alpha: float = 1.0,
) -> None:
    """Draw stable rain contours for the idealized two-intensity fields."""
    levels = [level for level in RAIN_LEVELS if np.nanmin(data) < level <= np.nanmax(data)]
    if not levels:
        return

    ax.contour(
        data,
        levels=levels,
        colors=color,
        linestyles=linestyle,
        linewidths=linewidth,
        alpha=alpha,
        origin="lower",
    )


def plot_case(
    case: str,
    obs: np.ndarray,
    fcst: np.ndarray,
    shifted: np.ndarray,
    cra_mask: np.ndarray,
    result: CraResult,
    *,
    threshold: float,
    output_dir: Path,
) -> None:
    """Save a compact visual diagnostic for one CRA case."""
    vmax = max(float(obs.max()), float(fcst.max()), float(shifted.max()))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)

    panels = [
        ("Observed rain", obs),
        ("Forecast and correction", fcst),
        ("Best-fit overlay", shifted),
    ]
    rain_im = None

    for ax, (title, data) in zip(axes, panels):
        rain_im = ax.imshow(data, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        add_rain_contours(ax, data, color="white", linewidth=0.9)

        if title == "Forecast and correction":
            add_rain_contours(ax, obs, color=OBS_COLOR, linewidth=1.2, alpha=0.85)
            add_rain_contours(ax, fcst, color=FCST_COLOR, linestyle="--", linewidth=1.4)

            centroid = masked_centroid(fcst, threshold)
            if centroid is not None:
                x0, y0 = centroid
                ax.annotate(
                    "",
                    xy=(x0 + result.corrective_shift_dx, y0 + result.corrective_shift_dy),
                    xytext=(x0, y0),
                    arrowprops={"arrowstyle": "->", "lw": 2.2, "color": SHIFTED_COLOR},
                )

        if title == "Best-fit overlay":
            add_rain_contours(ax, obs, color=OBS_COLOR, linewidth=1.2)
            add_rain_contours(ax, shifted, color=SHIFTED_COLOR, linewidth=1.4)

        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[1].text(
        0.03,
        0.97,
        f"corrective shift: dx={result.corrective_shift_dx}, dy={result.corrective_shift_dy}",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 3},
    )

    if rain_im is not None:
        fig.colorbar(rain_im, ax=axes[:3], shrink=0.82, label="rain rate (mm h-1)")

    fig.suptitle(
        (
            f"{case}: known error dx={result.imposed_forecast_dx:g}, dy={result.imposed_forecast_dy:g}; "
            f"diagnosed dx={result.diagnosed_forecast_error_dx}, dy={result.diagnosed_forecast_error_dy}; "
            f"error split D/V/P={result.pct_displacement:.1f}/{result.pct_volume:.1f}/{result.pct_pattern:.1f}%"
        ),
        fontsize=12,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{case}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    threshold = 1.0
    max_shift = 180
    dx_values = range(-180, 21)
    dy_values = range(0, 1)
    shape = (340, 520)
    cases = make_cases(shape)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for case, (obs, fcst, imposed_forecast_dx, imposed_forecast_dy) in cases.items():
        result, shifted, cra_mask = cra_decomposition(
            case,
            obs,
            fcst,
            imposed_forecast_dx=imposed_forecast_dx,
            imposed_forecast_dy=imposed_forecast_dy,
            threshold=threshold,
            max_shift=max_shift,
            dx_values=dx_values,
            dy_values=dy_values,
        )
        rows.append(result.__dict__)
        plot_case(
            case,
            obs,
            fcst,
            shifted,
            cra_mask,
            result,
            threshold=threshold,
            output_dir=OUTPUT_DIR,
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "cra_simulated_summary.csv", index=False)

    display_cols = [
        "case",
        "imposed_forecast_dx",
        "imposed_forecast_dy",
        "corrective_shift_dx",
        "corrective_shift_dy",
        "diagnosed_forecast_error_dx",
        "diagnosed_forecast_error_dy",
        "pct_displacement",
        "pct_volume",
        "pct_pattern",
        "spatial_corr_shifted",
    ]
    print(summary[display_cols].round(3).to_string(index=False))
    print(f"\nSaved figures and CSV summary to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
