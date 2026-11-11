"""
Helper functions for MITgcm diagnostics.
"""

import numpy as np
import xarray as xr

def add_month_coord(ds: xr.Dataset, time_dim="T") -> xr.Dataset:
    """Assign month numbers (1–12) to a dataset with monthly time steps."""
    nt = ds.sizes[time_dim]
    return ds.assign_coords(month=(time_dim, (np.arange(nt) % 12) + 1))

def monthly_climatology(ds: xr.Dataset, time_dim="T") -> xr.Dataset:
    """Compute the 12-month climatology by averaging each month across years."""
    ds = add_month_coord(ds, time_dim=time_dim)
    return ds.groupby("month").mean(time_dim)

def seasonal_from_monthly_clim(ds_clim: xr.Dataset, season: str) -> xr.Dataset:
    """Compute DJF or JJA mean from a 12-month climatology."""
    season = season.upper()

    if season == "DJF":
        months = [12, 1, 2]
    elif season == "JJA":
        months = [6, 7, 8]
    else:
        raise ValueError("season must be DJF or JJA")

    return ds_clim.sel(month=months).mean("month")

def global_mean(field3d: xr.DataArray, cvol: xr.DataArray) -> xr.DataArray:
    """Compute the global volume-weighted mean over valid ocean cells."""
    valid_vol = cvol.where(field3d.notnull())
    return (field3d * valid_vol).sum() / valid_vol.sum()
