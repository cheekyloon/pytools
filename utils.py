#!/usr/bin/env python

import numpy as np
import sys

def write_to_binary(data, fileout, precision='double'):
    """
    Write a NumPy array to a binary file in big-endian format,
    compatible with MITgcm.

    Note:
    numpy.tofile() does not allow explicit control over byte order
    and precision (float vs double), so it is not recommended.
    Instead, use the function below.

    Parameters
    ----------
    data : np.ndarray
        Input array to write.
    fileout : str
        Output binary file path.
    precision : str, optional
        'single' (32-bit float) or 'double' (64-bit float).
    """

    # Open file in binary write mode
    fid = open(fileout, "wb")

    # Flatten array (MITgcm expects Fortran-style contiguous layout externally)
    flatdata = data.flatten()

    # Convert to requested precision and enforce big-endian byte order
    if precision == 'single':
        if sys.byteorder == 'little':
            tmp = flatdata.astype(np.dtype('f')).byteswap(True).tobytes()
        else:
            tmp = flatdata.astype(np.dtype('f')).tobytes()

    elif precision == 'double':
        if sys.byteorder == 'little':
            tmp = flatdata.astype(np.dtype('d')).byteswap(True).tobytes()
        else:
            tmp = flatdata.astype(np.dtype('d')).tobytes()

    # Write binary data to file
    fid.write(tmp)
    fid.close()

    return None

def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    :param:
    ...data: numpy array of any dimension
    invalid: a binary array of same shape as 'data'. 
             True cells set where data value should be replaced.
             If None (default), use: invalid  = np.isnan(data)

    :return:
    ...data: the filled array. 
    """

    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]

def join(dirpath, fname):
    """Safe path join."""
    return os.path.join(dirpath, fname)

def resolve_nc(dirF, fname, legacy_name=None):
    path = join(dirF, fname)
    if os.path.exists(path):
        return path
    if legacy_name is not None:
        path2 = join(dirF, legacy_name)
        if os.path.exists(path2):
            return path2
    return path  

def pick_time(da, ind, time_dim="T"):
    """
    Select time index if dimension 'T' exists,
    otherwise return DataArray unchanged.
    """
    if (da is None) or (ind is None):
        return da
    return da.isel({time_dim: ind}) if time_dim in da.dims else da

def open_nc(path: str, strange_axes=None, grid=None) -> xr.Dataset:
    """
    Open MITgcm NetCDF file with backward compatibility.

    - First tries mitgcm_tools.open_ncfile (expects MNC-style metadata like diag_levels).
    - If that fails due to missing diag_levels (Dryad snapshot), falls back to xr.open_dataset
      and renames common dims to match grid conventions (XC/YC/XG/YG/ZC/ZL).
    """
    strange_axes = strange_axes or {}

    try:
        return mitgcm_tools.open_ncfile(path, strange_axes=strange_axes, grid=grid)

    except AttributeError as e:
        if "diag_levels" not in str(e):
            raise  # real bug, not Dryad

        ds = xr.open_dataset(path, decode_times=False)

        # --- rename dims to match grid if possible
        if grid is not None:
            rename = {}
            if "X" in ds.dims and "XC" in grid.dims: rename["X"] = "XC"
            if "Y" in ds.dims and "YC" in grid.dims: rename["Y"] = "YC"
            if "Xp1" in ds.dims and "XG" in grid.dims: rename["Xp1"] = "XG"
            if "Yp1" in ds.dims and "YG" in grid.dims: rename["Yp1"] = "YG"
            if "Zmd000029" in ds.dims and "ZC" in grid.dims: rename["Zmd000029"] = "ZC"
            if "Zld000029" in ds.dims and "ZL" in grid.dims: rename["Zld000029"] = "ZL"
            if "Zmd000001" in ds.dims and "ZC" in grid.dims: rename["Zmd000001"] = "ZC"
            if "Zd000001"  in ds.dims and "ZL" in grid.dims: rename["Zd000001"]  = "ZL"

            ds = ds.rename(rename)

            # attach coords from grid when matching dims exist
            for dim in ["XC", "YC", "XG", "YG", "ZC", "ZL"]:
                if dim in ds.dims and dim in grid.coords:
                    ds = ds.assign_coords({dim: grid[dim]})

        return ds

def label_axes(axarray,ignore=None,label_columns=False):
    import string      as st
    m=0

    if label_columns:
        axarray=np.transpose(axarray)

    if isinstance(axarray, dict):
        for n, ax in enumerate(axarray):
            if ignore is not None and n in ignore:
                m+=1
                continue
            else:
                axarray[ax].text(-0.1, 1.1, st.ascii_uppercase[n-m], transform=axarray[ax].transAxes,
                        size=14, weight='bold')
    else:
        for n, ax in enumerate(axarray.flat):
            if ignore is not None and n in ignore:
                m+=1
                continue
            else:
                ax.text(-0.1, 1.1, st.ascii_uppercase[n-m], transform=ax.transAxes,
                        size=14, weight='bold')

