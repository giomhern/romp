"""Run CRA rainfall verification from the installed MOMP workflow."""

from __future__ import annotations

from dataclasses import asdict
from itertools import product
import os
from pathlib import Path
import tempfile

import geopandas as gpd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "momp_mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import xarray as xr

from momp.lib.control import iter_list, make_case
from momp.lib.convention import Case
from momp.lib.loader import get_cfg, get_setting
from momp.metrics.cra import CraResult, cra_decomposition
from momp.utils.standard import dim_fmt, dim_fmt_model


PLOT_BACKGROUND = "#fafafa"


def _first(value):
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


def _format_year_pattern(pattern: str, year: int) -> str:
    if "{year" in pattern:
        return pattern.format(year=year)
    return pattern.format(year)


def _open_netcdf(path: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(path)
    except ValueError as exc:
        msg = str(exc)
        if "IO backends" in msg or "guess_engine" in msg:
            raise RuntimeError(
                "xarray could not open this NetCDF file because a required backend is missing. "
                "Install the project dependencies with `pip install -e .` or install `netcdf4`."
            ) from exc
        raise


def _standardize_model_dataset(dataset: xr.Dataset) -> xr.Dataset:
    dataset = dim_fmt_model(dataset)
    if "member" not in dataset.coords:
        for candidate in ("number", "sample"):
            if candidate in dataset.dims or candidate in dataset.coords:
                dataset = dataset.rename({candidate: "member"})
                break
    return dataset


def _model_configs(cfg) -> dict[str, dict[str, object]]:
    configs = {}
    for model, model_dir, variable, pattern, unit_cvt in zip(
        cfg.model_list,
        cfg.model_dir_list,
        cfg.model_var_list,
        cfg.file_pattern_list,
        cfg.unit_cvt_list,
    ):
        configs[model] = {
            "label": model,
            "model_dir": Path(model_dir),
            "variable": variable,
            "file_pattern": pattern,
            "unit_cvt": unit_cvt,
        }
    return configs


def _model_path(model_cfg: dict[str, object], year: int) -> Path:
    return Path(model_cfg["model_dir"]) / _format_year_pattern(str(model_cfg["file_pattern"]), year)


def _obs_path(setting, year: int) -> Path:
    pattern = _first(setting.obs_file_pattern)
    return Path(setting.obs_dir) / _format_year_pattern(str(pattern), year)


def _available_init_times(model_cfg: dict[str, object], year: int) -> pd.DatetimeIndex:
    dataset = _standardize_model_dataset(_open_netcdf(_model_path(model_cfg, year)))
    try:
        return pd.DatetimeIndex(pd.to_datetime(dataset.init_time.values))
    finally:
        dataset.close()


def _select_common_init_time(
    model_configs: dict[str, dict[str, object]],
    *,
    year: int,
    init_index: int,
) -> pd.Timestamp:
    common = None
    for model_cfg in model_configs.values():
        init_times = _available_init_times(model_cfg, year)
        common = init_times if common is None else common.intersection(init_times)

    if common is None or len(common) == 0:
        raise ValueError(f"No common initialization dates found for configured CRA models in {year}.")
    if init_index < 0 or init_index >= len(common):
        raise IndexError(f"cra_init_index={init_index} outside common-date range 0-{len(common) - 1}.")

    return pd.Timestamp(common.sort_values()[init_index])


def _load_boundary(shpfile_dir) -> gpd.GeoDataFrame | None:
    if shpfile_dir in (None, "None"):
        return None

    path = Path(shpfile_dir)
    if path.is_dir():
        shpfiles = sorted(path.glob("*.shp"))
        if not shpfiles:
            return None
        path = shpfiles[0]
    if not path.exists():
        return None

    boundary = gpd.read_file(path)
    if boundary.crs is not None and boundary.crs.to_epsg() != 4326:
        boundary = boundary.to_crs("EPSG:4326")
    return boundary


def _grid_mask_from_boundary(
    lat: np.ndarray,
    lon: np.ndarray,
    boundary: gpd.GeoDataFrame | None,
) -> np.ndarray | None:
    if boundary is None or boundary.empty:
        return None

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    geometry = boundary.geometry.union_all() if hasattr(boundary.geometry, "union_all") else boundary.geometry.unary_union
    return shapely.contains_xy(geometry, lon_grid, lat_grid)


def _load_forecast_accumulation(
    path: Path,
    *,
    variable: str,
    unit_cvt,
    init_time: pd.Timestamp,
    lead_start: int,
    lead_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    dataset = _standardize_model_dataset(_open_netcdf(path))
    try:
        rainfall = dataset[variable]
        member_note = "deterministic"
        if "member" in rainfall.dims:
            rainfall = rainfall.mean(dim="member", skipna=True)
            member_note = "ensemble_mean"
        if unit_cvt is not None:
            rainfall = rainfall * unit_cvt

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
            member_note,
        )
    finally:
        dataset.close()


def _load_observed_accumulation(
    path: Path,
    *,
    variable: str,
    unit_cvt,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = dim_fmt(_open_netcdf(path))
    try:
        rainfall = dataset[variable].sel(time=slice(valid_start, valid_end))
        if unit_cvt is not None:
            rainfall = rainfall * unit_cvt
        obs_accum = rainfall.sum(dim="time", skipna=True)
        return obs_accum.values.astype(float), obs_accum.lat.values.astype(float), obs_accum.lon.values.astype(float)
    finally:
        dataset.close()


def _mse_change_note(original_mse: float, shifted_mse: float) -> str:
    mse_delta = original_mse - shifted_mse
    if np.isfinite(original_mse) and original_mse > 0 and np.isfinite(shifted_mse):
        mse_change_pct = 100.0 * mse_delta / original_mse
        if mse_delta > 0:
            return f"improved {mse_change_pct:.1f}%"
        if mse_delta < 0:
            return f"worse {abs(mse_change_pct):.1f}%"
        return "no MSE change"
    return "MSE change n/a"


def _plot_map_panel(
    ax: plt.Axes,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    data: np.ndarray,
    *,
    threshold: float,
    title: str,
    vmin: float,
    vmax: float,
    boundary: gpd.GeoDataFrame | None = None,
) -> object:
    im = ax.pcolormesh(lon_grid, lat_grid, data, cmap="YlGnBu", shading="nearest", vmin=vmin, vmax=vmax)
    if np.nanmax(data) >= threshold and np.nanmin(data) < threshold:
        ax.contour(lon_grid, lat_grid, data, levels=[threshold], colors="#7f1d1d", linewidths=0.9)
    if boundary is not None:
        boundary.boundary.plot(ax=ax, color="black", linewidth=1.1)
    ax.set_facecolor(PLOT_BACKGROUND)
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return im


def _plot_real_case(
    obs: np.ndarray,
    fcst: np.ndarray,
    shifted: np.ndarray,
    result: CraResult,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    threshold: float,
    title: str,
    output_path: Path,
    boundary: gpd.GeoDataFrame | None,
    display_mse_label: str,
) -> None:
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
    mse_note = _mse_change_note(result.mse_total, result.mse_shifted)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    fig.patch.set_facecolor(PLOT_BACKGROUND)
    rain_im = None
    for ax, panel_title, data in zip(
        axes,
        ("Observed accumulation", "Forecast accumulation", "Shifted forecast"),
        (obs, fcst, shifted),
    ):
        rain_im = _plot_map_panel(
            ax,
            lon_grid,
            lat_grid,
            data,
            threshold=threshold,
            title=panel_title,
            vmin=0,
            vmax=vmax,
            boundary=boundary,
        )

    text_style = {
        "va": "top",
        "ha": "left",
        "color": "black",
        "fontsize": 9,
        "bbox": {"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 3},
    }
    axes[1].text(0.03, 0.97, f"{display_mse_label}\nforecast MSE: {result.mse_total:.2f}", transform=axes[1].transAxes, **text_style)
    axes[2].text(0.03, 0.97, f"{display_mse_label}\nshifted MSE: {result.mse_shifted:.2f}\n{mse_note}", transform=axes[2].transAxes, **text_style)

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


def _run_case(case: Case, cfg, setting, model_cfg: dict[str, object], init_time: pd.Timestamp, year: int, boundary) -> dict[str, object]:
    lead_start, lead_end = case.verification_window
    fcst_path = _model_path(model_cfg, year)
    obs_path = _obs_path(setting, year)

    fcst_accum, lat, lon, member_note = _load_forecast_accumulation(
        fcst_path,
        variable=case.model_var,
        unit_cvt=case.unit_cvt,
        init_time=init_time,
        lead_start=lead_start,
        lead_end=lead_end,
    )
    valid_start = init_time + pd.Timedelta(days=lead_start)
    valid_end = init_time + pd.Timedelta(days=lead_end)
    obs_accum, obs_lat, obs_lon = _load_observed_accumulation(
        obs_path,
        variable=case.obs_var,
        unit_cvt=setting.obs_unit_cvt,
        valid_start=valid_start,
        valid_end=valid_end,
    )

    if not np.allclose(lat, obs_lat) or not np.allclose(lon, obs_lon):
        raise ValueError(
            f"Forecast grid in {fcst_path} does not match observation grid in {obs_path}. "
            "Use matching model and observation resolutions."
        )

    region_mask = _grid_mask_from_boundary(lat, lon, boundary)
    display_mse_label = f"{case.region} CRA objective" if region_mask is not None else "CRA objective"
    cra_case = f"{case.model}_{year}_init{init_time:%Y%m%d}_lead{lead_start}-{lead_end}"
    result, shifted, _ = cra_decomposition(
        cra_case,
        obs_accum,
        fcst_accum,
        threshold=cfg.cra_threshold,
        max_shift=cfg.cra_max_shift,
        verification_mask=region_mask,
    )

    figure_path = Path(setting.dir_fig) / f"cra_real_rainfall_{case.case_name}_{year}.png"
    if cfg.cra_save_fig:
        _plot_real_case(
            obs_accum,
            fcst_accum,
            shifted,
            result,
            lat=lat,
            lon=lon,
            threshold=cfg.cra_threshold,
            title=str(case.model),
            output_path=figure_path,
            boundary=boundary,
            display_mse_label=display_mse_label,
        )

    return {
        "model": case.model,
        "year": year,
        "init_time": init_time.strftime("%Y-%m-%d"),
        "valid_start": valid_start.strftime("%Y-%m-%d"),
        "valid_end": valid_end.strftime("%Y-%m-%d"),
        "verification_window": f"{lead_start}-{lead_end}",
        "threshold": cfg.cra_threshold,
        "max_shift": cfg.cra_max_shift,
        "forecast_file": str(fcst_path),
        "obs_file": str(obs_path),
        "member_aggregation": member_note,
        "display_mse_region": display_mse_label,
        **asdict(result),
        "figure": str(figure_path) if cfg.cra_save_fig else "",
    }


def run_cra_workflow(cfg=None, setting=None) -> pd.DataFrame:
    """Run CRA verification for configured MOMP models and windows."""
    cfg = cfg or get_cfg()
    setting = setting or get_setting()
    model_configs = _model_configs(cfg)
    boundary = _load_boundary(setting.shpfile_dir)
    layout_pool = iter_list(vars(cfg))
    rows = []

    for year in case_years(cfg):
        init_time = (
            pd.Timestamp(cfg.cra_init_date)
            if cfg.cra_init_date not in (None, "None")
            else _select_common_init_time(model_configs, year=year, init_index=cfg.cra_init_index)
        )

        for combi in product(*layout_pool):
            case = make_case(Case, combi, vars(cfg))
            print(f"{'=' * 50}")
            print(
                f"processing {case.model} CRA rainfall verification for "
                f"window {case.verification_window}, init {init_time:%Y-%m-%d}, year {year}"
            )
            rows.append(_run_case(case, cfg, setting, model_configs[case.model], init_time, year, boundary))

    summary = pd.DataFrame(rows)
    output_path = Path(setting.dir_out) / "cra_real_rainfall_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    display_cols = [
        "model",
        "year",
        "init_time",
        "verification_window",
        "corrective_shift_dx",
        "corrective_shift_dy",
        "mse_total",
        "mse_shifted",
        "pct_displacement",
        "pct_volume",
        "pct_pattern",
        "spatial_corr_shifted",
    ]
    print(summary[display_cols].round(3).to_string(index=False))
    print(f"\nSaved CRA summary to: {output_path}")
    if cfg.cra_save_fig:
        print(f"Saved CRA figures under: {setting.dir_fig}")
    return summary


def case_years(cfg) -> tuple[int, ...]:
    if cfg.years not in (None, "All"):
        return tuple(cfg.years)
    return tuple(range(cfg.start_date[0], cfg.end_date[0] + 1))


if __name__ == "__main__":
    run_cra_workflow()
