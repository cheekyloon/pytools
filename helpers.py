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

def global_mean(field, weights):
    """Compute a weighted global mean over valid cells."""
    valid_weights = weights.where(field.notnull())
    return (field * valid_weights).sum() / valid_weights.sum()

def ice_edge_lat_SH_zonalmean(iceA: xr.DataArray, threshold=0.15):
    """
    Southern Hemisphere sea-ice edge from zonal-mean ice concentration.
    Returns the northernmost latitude (max YC, still <= 0) where zonal mean >= threshold.
    """
    # Zonal mean
    ice_zm = iceA.mean("XC", skipna=True).compute()

    # Southern Hemisphere only
    ice_zm = ice_zm.where(ice_zm["YC"] <= 0, drop=True)

    # Keep only points above the edge threshold
    ok = ice_zm.where(ice_zm >= threshold, drop=True)

    if ok.size == 0:
        return np.nan

    return float(ok["YC"].max().item())

def compute_diagnostics(theta, SST, ice, grid, vol, mask, threshold=0.15):
    """
    Compute global ocean temperature and Antarctic sea-ice diagnostics.

    Parameters
    ----------
    theta : xr.DataArray
        3D ocean potential temperature [°C].
    SST : xr.DataArray
        Sea surface temperature [°C].
    ice : xr.DataArray
        Sea-ice concentration [0–1].
    grid : xr.Dataset
        MITgcm grid containing cell areas (rA) and latitudes (YC).
    vol : xr.DataArray
        Ocean cell volumes [m³].
    mask : xr.DataArray
        Surface ocean mask.
    threshold : float, optional
        Minimum zonal-mean sea-ice concentration used to define
        the Antarctic sea-ice edge (default: 0.15).

    Returns
    -------
    dict
        Global volume-weighted mean ocean temperature [°C].
        Global area-weighted mean SST [°C].
        Antarctic sea-ice area [10^6 km²].
        Northernmost Antarctic sea-ice edge latitude [°].
    """

    # Ocean cell area [m²]
    area = grid.rA.where(mask > 0, 0)

    # Global mean ocean temperature [°C]
    mean_ocean_temp = global_mean(theta, vol).compute().item()

    # Global mean SST [°C]
    mean_SST = global_mean(SST, area).compute().item()

    # Antarctic sea-ice area [million km²]
    antarctic = grid.YC < 0

    ice_area = ((ice.fillna(0) * area.where(antarctic, 0))
        .sum(("YC", "XC")).compute().item() / 1e12)

    # Antarctic sea-ice edge [°]
    ice_edge = ice_edge_lat_SH_zonalmean(ice, threshold)

    return {
        "Mean ocean temperature [°C]": mean_ocean_temp,
        "Mean SST [°C]": mean_SST,
        "Antarctic sea-ice area [10^6 km²]": ice_area,
        "Antarctic sea-ice edge [°]": ice_edge,
    }


