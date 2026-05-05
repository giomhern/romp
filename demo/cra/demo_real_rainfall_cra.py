"""
Real-data CRA rainfall verification demo using ROMP sample or root data.

This stays disconnected from the main ROMP driver. It demonstrates how the
reusable CRA functions in `momp.metrics.cra` can be applied to actual forecast
and observed rainfall fields:

1. Load model forecast files plus the matching IMD observation file.
2. Accumulate forecast rainfall over lead days 1-15 for one common initialization.
3. Accumulate observed rainfall over the matching valid dates.
4. Run CRA-style displacement/error decomposition.
5. Save a CSV summary and diagnostic figures per model.

Run from the repo root:
    python demo/cra/demo_real_rainfall_cra.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = Path(__file__).resolve().parents[1]
ROOT_DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))
PLOT_BACKGROUND = "#fafafa"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import xarray as xr
import geopandas as gpd

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from momp.metrics.cra import CraResult, cra_decomposition
from momp.utils.standard import dim_fmt, dim_fmt_model


DEMO_MODEL_CONFIGS = {
    "aifs": {"label": "AIFS", "ensemble": False, "variable": "tp"},
    "ifs": {"label": "IFS", "ensemble": True, "variable": "tp"},
    "ngcm": {"label": "NGCM", "ensemble": True, "variable": "tp"},
}

ROOT_MODEL_CONFIGS = {
    "aifs_2p0": {
        "label": "AIFS 2p0",
        "path": "model_forecast_data/aifs/tp_2p0_lsm/{year}.nc",
        "variable": "tp",
    },
    "aifs_4p0": {
        "label": "AIFS 4p0",
        "path": "model_forecast_data/aifs/tp_4p0_lsm/{year}.nc",
        "variable": "tp",
    },
    "fuxi_s2s_2p0": {
        "label": "FuXi-S2S 2p0",
        "path": "model_forecast_data/fuxi_s2s/tp_2p0/{year}.nc",
        "variable": "tp",
    },
    "fuxi_s2s_4p0": {
        "label": "FuXi-S2S 4p0",
        "path": "model_forecast_data/fuxi_s2s/tp_4p0/{year}.nc",
        "variable": "tp",
    },
    "ifs_s2s_2p0": {
        "label": "IFS-S2S 2p0",
        "path": "model_forecast_data/IFS-S2S/tp_2p0/{year}.nc",
        "variable": "tp",
    },
    "ifs_s2s_4p0": {
        "label": "IFS-S2S 4p0",
        "path": "model_forecast_data/IFS-S2S/tp_4p0/{year}.nc",
        "variable": "tp",
    },
    "ngcm51_2p0": {
        "label": "NGCM51 2p0",
        "path": "model_forecast_data/ngcm51/twice_weekly_0z/tp_2p0/{year}.nc",
        "variable": "tp",
    },
    "ngcm51_4p0": {
        "label": "NGCM51 4p0",
        "path": "model_forecast_data/ngcm51/twice_weekly_0z/tp_4p0/{year}.nc",
        "variable": "tp",
    },
    "gencast52_2p0": {
        "label": "GenCast52 2p0",
        "path": "model_forecast_data/gencast52/tp_lsm_2p0/{year}.nc",
        "variable": "tp",
    },
    "gencast52_4p0": {
        "label": "GenCast52 4p0",
        "path": "model_forecast_data/gencast52/tp_lsm_4p0/{year}.nc",
        "variable": "tp",
    },
}

DEFAULT_ROOT_MODELS = ["aifs_2p0", "ifs_s2s_2p0", "ngcm51_2p0"]

DEMO_SHAPEFILE_PATH = DEMO_DIR / "data" / "shpfile" / "india_shapefile.shp"
ROOT_SHAPEFILE_PATH = ROOT_DATA_DIR / "ind_map_shpfile" / "india_shapefile.shp"


def open_netcdf(path: Path) -> xr.Dataset:
    """Open a NetCDF file with xarray and provide a clearer backend error."""
    try:
        return xr.open_dataset(path)
    except ValueError as exc:
        msg = str(exc)
        if "IO backends" in msg or "guess_engine" in msg:
            raise RuntimeError(
                "xarray could not open this NetCDF file because a required "
                "backend is missing. Install the project dependencies with "
                "`pip install -e .` from the repo root, or install `netcdf4` "
                "in the active environment."
            ) from exc
        raise


def standardize_model_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Standardize model dimensions and normalize optional ensemble dimensions."""
    dataset = dim_fmt_model(dataset)
    coord_names = list(dataset.coords)

    if "member" not in coord_names:
        for candidate in ("number", "sample"):
            if candidate in dataset.dims or candidate in dataset.coords:
                dataset = dataset.rename({candidate: "member"})
                break

    return dataset


def open_model_dataset(path: Path) -> xr.Dataset:
    """Open a model file and standardize dimensions."""
    return standardize_model_dataset(open_netcdf(path))


def model_configs(data_source: str) -> dict[str, dict[str, object]]:
    """Return model registry for the selected data source."""
    return ROOT_MODEL_CONFIGS if data_source == "root" else DEMO_MODEL_CONFIGS


def resolve_model_path(model_key: str, *, year: int, data_source: str) -> Path:
    """Resolve a model file path from the selected data tree."""
    configs = model_configs(data_source)
    model = configs[model_key]
    if data_source == "root":
        return ROOT_DATA_DIR / str(model["path"]).format(year=year)
    return DEMO_DIR / "data" / model_key / f"{year}.nc"


def resolve_obs_path(*, year: int, data_source: str, obs_resolution: str) -> Path:
    """Resolve the observation file path for the selected data tree."""
    if data_source == "demo":
        return DEMO_DIR / "data" / "obs" / f"{year}.nc"

    obs_dir = ROOT_DATA_DIR / "imd_rainfall_data" / obs_resolution
    candidates = [obs_dir / f"{year}.nc", obs_dir / f"data_{year}.nc"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No IMD observation file found for {year} at {obs_dir}. "
        f"Tried: {', '.join(str(path) for path in candidates)}"
    )


def available_init_times(model_key: str, *, year: int, data_source: str) -> pd.DatetimeIndex:
    """Return initialization times available for one bundled model."""
    path = resolve_model_path(model_key, year=year, data_source=data_source)
    dataset = open_model_dataset(path)
    init_times = pd.DatetimeIndex(pd.to_datetime(dataset.init_time.values))
    dataset.close()
    return init_times


def select_common_init_time(
    model_keys: list[str],
    *,
    year: int,
    init_index: int,
    data_source: str,
) -> pd.Timestamp:
    """Select one initialization time common to all requested model files."""
    common = available_init_times(model_keys[0], year=year, data_source=data_source)
    for model_key in model_keys[1:]:
        common = common.intersection(available_init_times(model_key, year=year, data_source=data_source))

    if len(common) == 0:
        raise ValueError(f"No common initialization dates found for {model_keys} in {year}.")
    if init_index < 0 or init_index >= len(common):
        raise IndexError(f"init_index={init_index} outside common-date range 0-{len(common) - 1}.")

    return pd.Timestamp(common.sort_values()[init_index])


def load_boundary(data_source: str) -> gpd.GeoDataFrame | None:
    """Load the bundled India boundary if geopandas can read it."""
    shapefile_path = ROOT_SHAPEFILE_PATH if data_source == "root" else DEMO_SHAPEFILE_PATH
    if not shapefile_path.exists():
        return None

    boundary = gpd.read_file(shapefile_path)
    if boundary.crs is not None and boundary.crs.to_epsg() != 4326:
        boundary = boundary.to_crs("EPSG:4326")
    return boundary


def grid_mask_from_boundary(
    lat: np.ndarray,
    lon: np.ndarray,
    boundary: gpd.GeoDataFrame | None,
) -> np.ndarray | None:
    """Return a boolean mask for grid-cell centers inside the plotted boundary."""
    if boundary is None or boundary.empty:
        return None

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    if hasattr(boundary.geometry, "union_all"):
        geometry = boundary.geometry.union_all()
    else:
        geometry = boundary.geometry.unary_union

    return shapely.contains_xy(geometry, lon_grid, lat_grid)


def mse_change_note(original_mse: float, shifted_mse: float) -> str:
    """Describe whether the shifted forecast improved relative to the original."""
    mse_delta = original_mse - shifted_mse
    if np.isfinite(original_mse) and original_mse > 0 and np.isfinite(shifted_mse):
        mse_change_pct = 100.0 * mse_delta / original_mse
        if mse_delta > 0:
            return f"improved {mse_change_pct:.1f}%"
        if mse_delta < 0:
            return f"worse {abs(mse_change_pct):.1f}%"
        return "no MSE change"
    return "MSE change n/a"


def load_forecast_accumulation(
    path: Path,
    *,
    variable: str,
    init_time: pd.Timestamp,
    lead_start: int,
    lead_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Timestamp, str]:
    """Load one accumulated forecast rainfall field from a bundled model sample."""
    dataset = open_model_dataset(path)
    rainfall = dataset[variable]

    member_note = "deterministic"
    if "member" in rainfall.dims:
        rainfall = rainfall.mean(dim="member", skipna=True)
        member_note = "ensemble_mean"

    available = pd.DatetimeIndex(pd.to_datetime(dataset.init_time.values))
    if init_time not in available:
        raise ValueError(f"{init_time:%Y-%m-%d} is not available in {path}.")

    fcst_accum = (
        rainfall.sel(init_time=init_time)
        .sel(step=slice(lead_start, lead_end))
        .sum(dim="step", skipna=True)
    )

    return (
        fcst_accum.values.astype(float),
        fcst_accum.lat.values.astype(float),
        fcst_accum.lon.values.astype(float),
        init_time,
        member_note,
    )


def load_observed_accumulation(
    path: Path,
    *,
    variable: str,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load observed rainfall accumulated over the matching valid dates."""
    dataset = dim_fmt(open_netcdf(path))
    rainfall = dataset[variable].sel(time=slice(valid_start, valid_end))
    obs_accum = rainfall.sum(dim="time", skipna=True)

    return obs_accum.values.astype(float), obs_accum.lat.values.astype(float), obs_accum.lon.values.astype(float)


def plot_map_panel(
    ax: plt.Axes,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    data: np.ndarray,
    *,
    threshold: float,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    mask_panel: bool = False,
    boundary: gpd.GeoDataFrame | None = None,
    draw_threshold_contour: bool = True,
) -> object:
    """Draw one map panel and its threshold contour."""
    if mask_panel:
        im = ax.pcolormesh(lon_grid, lat_grid, data, cmap="Greys", shading="nearest", vmin=0, vmax=1)
    else:
        im = ax.pcolormesh(lon_grid, lat_grid, data, cmap=cmap, shading="nearest", vmin=vmin, vmax=vmax)

    if draw_threshold_contour and np.nanmax(data) >= threshold and np.nanmin(data) < threshold:
        ax.contour(lon_grid, lat_grid, data, levels=[threshold], colors="#7f1d1d", linewidths=0.9)

    if boundary is not None:
        boundary.boundary.plot(ax=ax, color="black", linewidth=1.1)

    ax.set_facecolor(PLOT_BACKGROUND)
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return im


def plot_real_case(
    obs: np.ndarray,
    fcst: np.ndarray,
    shifted: np.ndarray,
    cra_mask: np.ndarray,
    result: CraResult,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    threshold: float,
    title: str,
    output_path: Path,
    boundary: gpd.GeoDataFrame | None,
    display_mse_original: float,
    display_mse_shifted: float,
    display_mse_label: str,
    display_mask: np.ndarray | None = None,
    draw_threshold_contours: bool = True,
) -> None:
    """Save a diagnostic plot for a real-data CRA case."""
    if display_mask is not None:
        obs = np.where(display_mask, obs, np.nan)
        fcst = np.where(display_mask, fcst, np.nan)
        shifted = np.where(display_mask, shifted, np.nan)

    finite_rain = np.concatenate(
        [
            obs[np.isfinite(obs)].ravel(),
            fcst[np.isfinite(fcst)].ravel(),
            shifted[np.isfinite(shifted)].ravel(),
        ]
    )
    vmax = float(np.nanpercentile(finite_rain, 98)) if finite_rain.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax([np.nanmax(obs), np.nanmax(fcst), np.nanmax(shifted), 1.0]))
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    mse_note = mse_change_note(display_mse_original, display_mse_shifted)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    fig.patch.set_facecolor(PLOT_BACKGROUND)
    panels = [
        ("Observed accumulation", obs),
        ("Forecast accumulation", fcst),
        ("Shifted forecast", shifted),
    ]
    rain_im = None

    for ax, (panel_title, data) in zip(axes, panels):
        im = plot_map_panel(
            ax,
            lon_grid,
            lat_grid,
            data,
            threshold=threshold,
            title=panel_title,
            cmap="YlGnBu",
            vmin=0,
            vmax=vmax,
            boundary=boundary,
            draw_threshold_contour=draw_threshold_contours,
        )
        rain_im = im

    annotation_style = {
        "transform": None,
        "va": "top",
        "ha": "left",
        "color": "black",
        "fontsize": 9,
        "bbox": {"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 3},
    }
    axes[1].text(
        0.03,
        0.97,
        f"{display_mse_label}\nforecast MSE: {display_mse_original:.2f}",
        **{**annotation_style, "transform": axes[1].transAxes},
    )
    axes[2].text(
        0.03,
        0.97,
        f"{display_mse_label}\nshifted MSE: {display_mse_shifted:.2f}\n{mse_note}",
        **{**annotation_style, "transform": axes[2].transAxes},
    )

    if rain_im is not None:
        fig.colorbar(rain_im, ax=axes[:3], shrink=0.82, label="rainfall accumulation (mm)")

    fig.suptitle(
        (
            f"{title}\n"
            f"Error split: displacement {result.pct_displacement:.1f}% | "
            f"volume {result.pct_volume:.1f}% | pattern {result.pct_pattern:.1f}%"
        ),
        fontsize=12,
        linespacing=1.35,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_model_case(
    model_key: str,
    *,
    data_source: str,
    obs_resolution: str,
    year: int,
    init_index: int,
    init_time: pd.Timestamp,
    lead_start: int,
    lead_end: int,
    threshold: float,
    max_shift: int,
    output_dir: Path,
    boundary: gpd.GeoDataFrame | None,
) -> dict[str, object]:
    """Run one real-data CRA case for one bundled model."""
    model = model_configs(data_source)[model_key]
    fcst_path = resolve_model_path(model_key, year=year, data_source=data_source)
    obs_path = resolve_obs_path(year=year, data_source=data_source, obs_resolution=obs_resolution)

    fcst_accum, lat, lon, init_time, member_note = load_forecast_accumulation(
        fcst_path,
        variable=model["variable"],
        init_time=init_time,
        lead_start=lead_start,
        lead_end=lead_end,
    )
    valid_start = init_time + pd.Timedelta(days=lead_start)
    valid_end = init_time + pd.Timedelta(days=lead_end)
    obs_accum, obs_lat, obs_lon = load_observed_accumulation(
        obs_path,
        variable="RAINFALL",
        valid_start=valid_start,
        valid_end=valid_end,
    )

    if not np.allclose(lat, obs_lat) or not np.allclose(lon, obs_lon):
        raise ValueError(
            "Forecast and observed grids do not match. Choose the observation "
            "resolution that matches the model, for example `--obs-resolution 2p0` "
            "with `*_2p0` model keys or `--obs-resolution 4p0` with `*_4p0` model keys."
        )

    india_mask = grid_mask_from_boundary(lat, lon, boundary)
    if india_mask is None:
        verification_mask = None
        display_mse_label = "CRA objective"
    else:
        verification_mask = india_mask
        display_mse_label = "India CRA objective"

    case = f"{model['label']}_{year}_init{init_time:%Y%m%d}_lead{lead_start}-{lead_end}"
    result, shifted, cra_mask = cra_decomposition(
        case,
        obs_accum,
        fcst_accum,
        threshold=threshold,
        max_shift=max_shift,
        verification_mask=verification_mask,
    )
    display_mse_original = result.mse_total
    display_mse_shifted = result.mse_shifted

    row = {
        "model": model["label"],
        "model_key": model_key,
        "data_source": data_source,
        "obs_resolution": obs_resolution,
        "forecast_file": str(fcst_path.relative_to(REPO_ROOT)),
        "obs_file": str(obs_path.relative_to(REPO_ROOT)),
        "member_aggregation": member_note,
        "year": year,
        "init_index": init_index,
        "init_time": init_time.strftime("%Y-%m-%d"),
        "valid_start": valid_start.strftime("%Y-%m-%d"),
        "valid_end": valid_end.strftime("%Y-%m-%d"),
        "lead_start": lead_start,
        "lead_end": lead_end,
        "threshold": threshold,
        "max_shift": max_shift,
        "display_mse_region": display_mse_label,
        "display_mse_original": display_mse_original,
        "display_mse_shifted": display_mse_shifted,
        **asdict(result),
    }

    plot_path = output_dir / f"cra_real_rainfall_{model_key}_{year}.png"
    india_plot_path = output_dir / f"cra_real_rainfall_{model_key}_{year}_india_only_no_outlines.png"
    plot_real_case(
        obs_accum,
        fcst_accum,
        shifted,
        cra_mask,
        result,
        lat=lat,
        lon=lon,
        threshold=threshold,
        title=str(model["label"]),
        output_path=plot_path,
        boundary=boundary,
        display_mse_original=display_mse_original,
        display_mse_shifted=display_mse_shifted,
        display_mse_label=display_mse_label,
    )
    plot_real_case(
        obs_accum,
        fcst_accum,
        shifted,
        cra_mask,
        result,
        lat=lat,
        lon=lon,
        threshold=threshold,
        title=f"{model['label']} India-only",
        output_path=india_plot_path,
        boundary=boundary,
        display_mse_original=display_mse_original,
        display_mse_shifted=display_mse_shifted,
        display_mse_label=display_mse_label,
        display_mask=india_mask,
        draw_threshold_contours=False,
    )
    row["figure"] = str(plot_path.relative_to(REPO_ROOT))
    row["figure_india_only_no_outlines"] = str(india_plot_path.relative_to(REPO_ROOT))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-source",
        choices=["root", "demo"],
        default="root",
        help="Use root ./data files or the tiny bundled demo/data files.",
    )
    parser.add_argument("--year", type=int, default=2015, help="Forecast and observation year to use.")
    parser.add_argument(
        "--obs-resolution",
        choices=["0p25", "1p0", "2p0", "4p0"],
        default="2p0",
        help="IMD observation resolution under root data/imd_rainfall_data. Use 2p0 with *_2p0 models.",
    )
    parser.add_argument(
        "--init-index",
        type=int,
        default=10,
        help="Index into dates common to all selected model files, used when --init-date is omitted.",
    )
    parser.add_argument(
        "--init-date",
        type=str,
        default=None,
        help="Initialization date to use, for example 2015-06-06. Overrides --init-index.",
    )
    parser.add_argument("--lead-start", type=int, default=1, help="First lead day included in the accumulation.")
    parser.add_argument("--lead-end", type=int, default=15, help="Last lead day included in the accumulation.")
    parser.add_argument("--threshold", type=float, default=20.0, help="Accumulation threshold used to define CRAs.")
    parser.add_argument("--max-shift", type=int, default=3, help="Maximum grid-cell shift searched in each direction.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Model keys to process. Root defaults: "
            f"{', '.join(DEFAULT_ROOT_MODELS)}. Demo defaults: {', '.join(DEMO_MODEL_CONFIGS)}."
        ),
    )
    args = parser.parse_args()
    configs = model_configs(args.data_source)
    if args.models is None:
        args.models = DEFAULT_ROOT_MODELS if args.data_source == "root" else list(DEMO_MODEL_CONFIGS)

    unknown = sorted(set(args.models) - set(configs))
    if unknown:
        parser.error(
            f"Unknown model key(s) for --data-source {args.data_source}: {', '.join(unknown)}. "
            f"Available: {', '.join(configs)}"
        )

    return args


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    init_time = (
        pd.Timestamp(args.init_date)
        if args.init_date
        else select_common_init_time(
            args.models,
            year=args.year,
            init_index=args.init_index,
            data_source=args.data_source,
        )
    )
    boundary = load_boundary(args.data_source)

    rows = [
        run_model_case(
            model_key,
            data_source=args.data_source,
            obs_resolution=args.obs_resolution,
            year=args.year,
            init_index=args.init_index,
            init_time=init_time,
            lead_start=args.lead_start,
            lead_end=args.lead_end,
            threshold=args.threshold,
            max_shift=args.max_shift,
            output_dir=OUTPUT_DIR,
            boundary=boundary,
        )
        for model_key in args.models
    ]

    summary = pd.DataFrame(rows)
    summary_path = OUTPUT_DIR / "cra_real_rainfall_summary.csv"
    summary.to_csv(summary_path, index=False)

    display_cols = [
        "model",
        "data_source",
        "obs_resolution",
        "member_aggregation",
        "init_time",
        "valid_start",
        "valid_end",
        "threshold",
        "corrective_shift_dx",
        "corrective_shift_dy",
        "display_mse_region",
        "display_mse_original",
        "display_mse_shifted",
        "diagnosed_forecast_error_dx",
        "diagnosed_forecast_error_dy",
        "pct_displacement",
        "pct_volume",
        "pct_pattern",
        "spatial_corr_shifted",
    ]
    print(summary[display_cols].round(3).to_string(index=False))
    print(f"\nSaved real-data CRA summary to: {summary_path}")
    print("Saved real-data CRA figures:")
    for figure in summary["figure"]:
        print(f"  {REPO_ROOT / figure}")
    print("Saved India-only CRA figures without object outlines:")
    for figure in summary["figure_india_only_no_outlines"]:
        print(f"  {REPO_ROOT / figure}")


if __name__ == "__main__":
    main()
