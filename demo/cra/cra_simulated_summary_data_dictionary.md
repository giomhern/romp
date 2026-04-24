# CRA Simulated Summary Data Dictionary

The demo writes `output/cra_simulated_summary.csv`. Each row is one synthetic
forecast-verification case.

| Column | Meaning |
| --- | --- |
| `case` | Name of the synthetic experiment. |
| `imposed_forecast_dx` | Known east-west forecast displacement applied when creating the synthetic forecast. Positive values mean the forecast object was moved to the right/east relative to the observation. |
| `imposed_forecast_dy` | Known north-south forecast displacement applied when creating the synthetic forecast. Positive values mean the forecast object was moved upward/northward in array coordinates used by the plot. |
| `corrective_shift_dx` | Best-fit x shift applied to the forecast to align it with the observation. This has the opposite sign of the diagnosed forecast location error. |
| `corrective_shift_dy` | Best-fit y shift applied to the forecast to align it with the observation. This has the opposite sign of the diagnosed forecast location error. |
| `diagnosed_forecast_error_dx` | Diagnosed forecast x-location error, computed as `-corrective_shift_dx`. This is directly comparable to `imposed_forecast_dx`. |
| `diagnosed_forecast_error_dy` | Diagnosed forecast y-location error, computed as `-corrective_shift_dy`. This is directly comparable to `imposed_forecast_dy`. |
| `n_obs_objects` | Number of contiguous observed rain objects above the CRA threshold. |
| `n_fcst_objects` | Number of contiguous forecast rain objects above the CRA threshold before shifting. |
| `mse_total` | Mean squared error between the original forecast and observation over the relaxed CRA mask. |
| `mse_shifted` | Mean squared error after applying the best-fit corrective shift to the forecast. |
| `mse_displacement` | Error attributed to displacement, calculated as `mse_total - mse_shifted` and floored at zero. |
| `mse_volume` | Error attributed to mean rainfall bias after shifting, calculated as `(mean_fcst_shifted - mean_obs)^2`. |
| `mse_pattern` | Residual error after displacement and volume are accounted for, calculated as `mse_shifted - mse_volume` and floored at zero. |
| `pct_displacement` | Percent of `mse_total` attributed to displacement error. |
| `pct_volume` | Percent of `mse_total` attributed to volume error. |
| `pct_pattern` | Percent of `mse_total` attributed to pattern error. |
| `mean_obs` | Mean observed rainfall over the relaxed CRA mask. |
| `mean_fcst_shifted` | Mean shifted-forecast rainfall over the relaxed CRA mask. |
| `peak_obs` | Maximum observed rainfall inside the relaxed CRA mask. |
| `peak_fcst_shifted` | Maximum shifted-forecast rainfall inside the relaxed CRA mask. |
| `spatial_corr_original` | Spatial correlation between the original forecast and observation over the relaxed CRA mask. |
| `spatial_corr_shifted` | Spatial correlation between the shifted forecast and observation over the relaxed CRA mask. |

## Sign Convention

The imposed forecast displacement describes where the forecast was placed
relative to the observation. The corrective shift describes how the forecast must
be moved back to match the observation.

For example, if the forecast was created by shifting the observed rain object
`20` grid cells right and `10` grid cells up, then the best corrective shift is
approximately `dx=-20`, `dy=-10`. The diagnosed forecast error is therefore
reported as `dx=20`, `dy=10`.

## Error Components

The demo follows the original CRA-style MSE decomposition:

```text
mse_total = mse_displacement + mse_volume + mse_pattern
```

In this simplified sandbox, the CRA mask is relaxed to include the union of:

- observed rain above threshold,
- original forecast rain above threshold,
- shifted forecast rain above threshold.

That relaxation lets the demo handle the non-overlapping synthetic case, which is
one of the important weaknesses discussed in the paper.
