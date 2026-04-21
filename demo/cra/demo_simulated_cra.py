"""
Simulated Contiguous Rain Area (CRA) verification demo.

This script is intentionally disconnected from the main ROMP workflow. It uses
idealized rainfall fields with known errors to exercise the core CRA ideas from
Ebert and Gallus (2009): best-fit displacement, volume error, and pattern error.

Run from the repo root:
    python demo/cra/demo_simulated_cra.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass(frozen=True)
class CraResult:
    case: str
    imposed_forecast_dx: int
    imposed_forecast_dy: int
    corrective_shift_dx: int
    corrective_shift_dy: int
    diagnosed_forecast_error_dx: int
    diagnosed_forecast_error_dy: int
    n_obs_objects: int
    n_fcst_objects: int
    mse_total: float
    mse_shifted: float
    mse_displacement: float
    mse_volume: float
    mse_pattern: float
    pct_displacement: float
    pct_volume: float
    pct_pattern: float
    mean_obs: float
    mean_fcst_shifted: float
    peak_obs: float
    peak_fcst_shifted: float
    spatial_corr_original: float
    spatial_corr_shifted: float


def make_ellipse_field(
    shape: tuple[int, int],
    *,
    center: tuple[float, float],
    radii: tuple[float, float],
    amplitude: float,
    core_amplitude: float | None = None,
    core_radii: tuple[float, float] | None = None,
    angle_degrees: float = 0.0,
) -> np.ndarray:
    """Create a smooth-ish synthetic rain object with an optional heavy core."""
    y, x = np.indices(shape, dtype=float)
    cy, cx = center
    ry, rx = radii
    theta = np.deg2rad(angle_degrees)

    x0 = x - cx
    y0 = y - cy
    x_rot = x0 * np.cos(theta) + y0 * np.sin(theta)
    y_rot = -x0 * np.sin(theta) + y0 * np.cos(theta)

    ellipse = (x_rot / rx) ** 2 + (y_rot / ry) ** 2 <= 1.0
    field = np.zeros(shape, dtype=float)

    # Taper the object slightly so the pattern has more structure than a block.
    radial = np.sqrt((x_rot / rx) ** 2 + (y_rot / ry) ** 2)
    field[ellipse] = amplitude * (1.15 - 0.35 * radial[ellipse])

    if core_amplitude is not None and core_radii is not None:
        cry, crx = core_radii
        core = (x_rot / crx) ** 2 + (y_rot / cry) ** 2 <= 1.0
        core_radial = np.sqrt((x_rot / crx) ** 2 + (y_rot / cry) ** 2)
        field[core] += core_amplitude * (1.1 - 0.25 * core_radial[core])

    return field


def shift_field(field: np.ndarray, dy: int, dx: int, fill: float = 0.0) -> np.ndarray:
    """Translate a 2-D field by integer grid cells without wraparound."""
    shifted = np.full_like(field, fill, dtype=float)
    ny, nx = field.shape

    src_y0 = max(0, -dy)
    src_y1 = min(ny, ny - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(nx, nx - dx)

    dst_y0 = max(0, dy)
    dst_y1 = min(ny, ny + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(nx, nx + dx)

    if src_y0 < src_y1 and src_x0 < src_x1:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = field[src_y0:src_y1, src_x0:src_x1]

    return shifted


def finite_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Spatial correlation over a mask; returns nan for constant fields."""
    aa = a[mask].ravel()
    bb = b[mask].ravel()
    valid = np.isfinite(aa) & np.isfinite(bb)

    if valid.sum() < 2:
        return np.nan

    aa = aa[valid]
    bb = bb[valid]
    if np.allclose(aa, aa[0]) or np.allclose(bb, bb[0]):
        return np.nan

    return float(np.corrcoef(aa, bb)[0, 1])


def masked_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    diff = a[mask] - b[mask]
    return float(np.nanmean(diff**2))


def object_count(field: np.ndarray, threshold: float) -> int:
    _, count = ndimage.label(field >= threshold)
    return int(count)


def best_shift_by_mse(
    obs: np.ndarray,
    fcst: np.ndarray,
    *,
    threshold: float,
    max_shift: int,
) -> tuple[int, int, np.ndarray, np.ndarray, float]:
    """
    Find the integer forecast translation that minimizes MSE over the CRA mask.

    The CRA mask is relaxed for this demo: it is the union of observed rain,
    original forecast rain, and shifted forecast rain above threshold. This lets
    the synthetic non-overlap case be matched, mirroring the paper's recommended
    fix to the strict overlap requirement.
    """
    obs_mask = obs >= threshold
    fcst_mask = fcst >= threshold
    best: tuple[int, int, np.ndarray, np.ndarray, float] | None = None

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = shift_field(fcst, dy, dx)
            shifted_mask = shifted >= threshold
            cra_mask = obs_mask | fcst_mask | shifted_mask
            mse = masked_mse(shifted, obs, cra_mask)

            if best is None or mse < best[-1]:
                best = (dy, dx, shifted, cra_mask, mse)

    if best is None:
        raise RuntimeError("No valid shift found")

    return best


def cra_decomposition(
    case: str,
    obs: np.ndarray,
    fcst: np.ndarray,
    *,
    imposed_forecast_dx: int,
    imposed_forecast_dy: int,
    threshold: float = 1.0,
    max_shift: int = 80,
) -> tuple[CraResult, np.ndarray, np.ndarray]:
    """Compute a simple CRA-style MSE decomposition for one synthetic case."""
    dy, dx, shifted, cra_mask, mse_shifted = best_shift_by_mse(
        obs,
        fcst,
        threshold=threshold,
        max_shift=max_shift,
    )

    mse_total = masked_mse(fcst, obs, cra_mask)
    mean_obs = float(np.nanmean(obs[cra_mask]))
    mean_fcst_shifted = float(np.nanmean(shifted[cra_mask]))

    mse_displacement = max(mse_total - mse_shifted, 0.0)
    mse_volume = (mean_fcst_shifted - mean_obs) ** 2
    mse_pattern = max(mse_shifted - mse_volume, 0.0)

    if mse_total > 0:
        pct_displacement = 100.0 * mse_displacement / mse_total
        pct_volume = 100.0 * mse_volume / mse_total
        pct_pattern = 100.0 * mse_pattern / mse_total
    else:
        pct_displacement = pct_volume = pct_pattern = np.nan

    original_corr = finite_corr(fcst, obs, cra_mask)
    shifted_corr = finite_corr(shifted, obs, cra_mask)

    result = CraResult(
        case=case,
        imposed_forecast_dx=imposed_forecast_dx,
        imposed_forecast_dy=imposed_forecast_dy,
        corrective_shift_dx=dx,
        corrective_shift_dy=dy,
        diagnosed_forecast_error_dx=-dx,
        diagnosed_forecast_error_dy=-dy,
        n_obs_objects=object_count(obs, threshold),
        n_fcst_objects=object_count(fcst, threshold),
        mse_total=mse_total,
        mse_shifted=mse_shifted,
        mse_displacement=mse_displacement,
        mse_volume=mse_volume,
        mse_pattern=mse_pattern,
        pct_displacement=pct_displacement,
        pct_volume=pct_volume,
        pct_pattern=pct_pattern,
        mean_obs=mean_obs,
        mean_fcst_shifted=mean_fcst_shifted,
        peak_obs=float(np.nanmax(obs[cra_mask])),
        peak_fcst_shifted=float(np.nanmax(shifted[cra_mask])),
        spatial_corr_original=original_corr,
        spatial_corr_shifted=shifted_corr,
    )

    return result, shifted, cra_mask


def make_cases(shape: tuple[int, int]) -> dict[str, tuple[np.ndarray, np.ndarray, int, int]]:
    """Return synthetic observed/forecast pairs and known forecast shifts."""
    obs = make_ellipse_field(
        shape,
        center=(70, 75),
        radii=(28, 17),
        amplitude=7.0,
        core_amplitude=10.0,
        core_radii=(10, 6),
        angle_degrees=-12,
    )

    cases = {
        "shift_only": (obs, shift_field(obs, 10, 20), 20, 10),
        "shift_and_amplify": (obs, 1.5 * shift_field(obs, 10, 20), 20, 10),
        "shift_and_stretch": (
            obs,
            make_ellipse_field(
                shape,
                center=(80, 95),
                radii=(19, 31),
                amplitude=7.0,
                core_amplitude=9.0,
                core_radii=(6, 14),
                angle_degrees=8,
            ),
            20,
            10,
        ),
        "nonoverlap_shift_only": (obs, shift_field(obs, 0, 58), 58, 0),
    }

    return cases


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
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)

    panels = [
        ("Observed", obs),
        ("Forecast", fcst),
        ("Shifted forecast", shifted),
        ("CRA mask", cra_mask.astype(float)),
    ]

    for ax, (title, data) in zip(axes, panels):
        if title == "CRA mask":
            im = ax.imshow(data, origin="lower", cmap="Greys", vmin=0, vmax=1)
        else:
            im = ax.imshow(data, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
            ax.contour(data >= threshold, levels=[0.5], colors="white", linewidths=0.9)

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[1].annotate(
        "",
        xy=(0.52, 0.56),
        xytext=(
            0.52 - result.corrective_shift_dx / 130,
            0.56 - result.corrective_shift_dy / 130,
        ),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "crimson"},
    )
    axes[1].text(
        0.03,
        0.97,
        f"correction: dx={result.corrective_shift_dx}, dy={result.corrective_shift_dy}",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 3},
    )

    fig.colorbar(im, ax=axes[:3], shrink=0.82, label="rain rate / accumulation")
    fig.suptitle(
        (
            f"{case}: displacement={result.pct_displacement:.1f}%, "
            f"volume={result.pct_volume:.1f}%, pattern={result.pct_pattern:.1f}%"
        ),
        fontsize=12,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{case}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    threshold = 1.0
    max_shift = 80
    shape = (140, 160)
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
