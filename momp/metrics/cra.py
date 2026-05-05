"""Contiguous Rain Area (CRA) verification helpers.

The functions here implement a compact, reusable subset of the CRA ideas used
in the demo scripts: integer-grid best-fit displacement, a relaxed CRA mask, and
the original MSE-style decomposition into displacement, volume, and pattern
terms. They are intentionally small so they can be used by isolated demos before
being wired into the broader ROMP workflow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class CraResult:
    case: str
    imposed_forecast_dx: float
    imposed_forecast_dy: float
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
    """Spatial correlation over a mask; returns NaN for constant fields."""
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
    """Mean squared error over a boolean mask."""
    valid_mask = mask & np.isfinite(a) & np.isfinite(b)
    if not np.any(valid_mask):
        return np.nan

    diff = a[valid_mask] - b[valid_mask]
    return float(np.nanmean(diff**2))


def object_count(field: np.ndarray, threshold: float) -> int:
    """Count contiguous objects above a threshold."""
    _, count = ndimage.label(np.asarray(field) >= threshold)
    return int(count)


def best_shift_by_mse(
    obs: np.ndarray,
    fcst: np.ndarray,
    *,
    threshold: float,
    max_shift: int,
    verification_mask: np.ndarray | None = None,
    dy_values: range | None = None,
    dx_values: range | None = None,
) -> tuple[int, int, np.ndarray, np.ndarray, float]:
    """
    Find the integer forecast translation that minimizes MSE over a CRA mask.

    The mask is relaxed: it is the union of observed rain, original forecast
    rain, and shifted forecast rain above threshold. This permits matching
    nearby but non-overlapping features, which is one of the practical fixes
    discussed by Ebert and Gallus (2009).
    """
    obs = np.asarray(obs, dtype=float)
    fcst = np.asarray(fcst, dtype=float)

    if obs.shape != fcst.shape:
        raise ValueError(f"obs and fcst must have the same shape, got {obs.shape} and {fcst.shape}")
    if verification_mask is not None and verification_mask.shape != obs.shape:
        raise ValueError(
            f"verification_mask must have shape {obs.shape}, got {verification_mask.shape}"
        )

    obs_mask = obs >= threshold
    fcst_mask = fcst >= threshold
    best: tuple[int, int, np.ndarray, np.ndarray, float] | None = None

    if dy_values is None:
        dy_values = range(-max_shift, max_shift + 1)
    if dx_values is None:
        dx_values = range(-max_shift, max_shift + 1)

    for dy in dy_values:
        for dx in dx_values:
            shifted = shift_field(fcst, dy, dx)
            shifted_mask = shifted >= threshold
            cra_mask = obs_mask | fcst_mask | shifted_mask
            score_mask = cra_mask if verification_mask is None else cra_mask & verification_mask
            mse = masked_mse(shifted, obs, score_mask)

            if np.isnan(mse):
                continue

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
    imposed_forecast_dx: float = np.nan,
    imposed_forecast_dy: float = np.nan,
    threshold: float = 1.0,
    max_shift: int = 80,
    verification_mask: np.ndarray | None = None,
    dy_values: range | None = None,
    dx_values: range | None = None,
) -> tuple[CraResult, np.ndarray, np.ndarray]:
    """Compute a CRA-style MSE decomposition for one forecast-observation pair."""
    obs = np.asarray(obs, dtype=float)
    fcst = np.asarray(fcst, dtype=float)
    dy, dx, shifted, cra_mask, mse_shifted = best_shift_by_mse(
        obs,
        fcst,
        threshold=threshold,
        max_shift=max_shift,
        verification_mask=verification_mask,
        dy_values=dy_values,
        dx_values=dx_values,
    )

    original_cra_mask = (obs >= threshold) | (fcst >= threshold)
    original_score_mask = (
        original_cra_mask if verification_mask is None else original_cra_mask & verification_mask
    )
    shifted_score_mask = cra_mask if verification_mask is None else cra_mask & verification_mask
    mse_total = masked_mse(fcst, obs, original_score_mask)
    valid_mask = shifted_score_mask & np.isfinite(obs) & np.isfinite(shifted)
    mean_obs = float(np.nanmean(obs[valid_mask]))
    mean_fcst_shifted = float(np.nanmean(shifted[valid_mask]))

    mse_displacement = max(mse_total - mse_shifted, 0.0)
    mse_volume = (mean_fcst_shifted - mean_obs) ** 2
    mse_pattern = max(mse_shifted - mse_volume, 0.0)

    if mse_total > 0:
        pct_displacement = 100.0 * mse_displacement / mse_total
        pct_volume = 100.0 * mse_volume / mse_total
        pct_pattern = 100.0 * mse_pattern / mse_total
    else:
        pct_displacement = pct_volume = pct_pattern = np.nan

    original_corr = finite_corr(fcst, obs, valid_mask)
    shifted_corr = finite_corr(shifted, obs, valid_mask)

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
        peak_obs=float(np.nanmax(obs[valid_mask])),
        peak_fcst_shifted=float(np.nanmax(shifted[valid_mask])),
        spatial_corr_original=original_corr,
        spatial_corr_shifted=shifted_corr,
    )

    return result, shifted, cra_mask
