#!/usr/bin/env python 

### import modules
#import xgcm
import mitgcm_tools
import numpy             as np
import xarray            as xr
import MITgcmutils.jmd95 as jmd95
from .utils   import resolve_nc
from .utils   import open_nc
from .utils   import pick_time
from .utils   import join

def load_ice_lat(
    dirF: str,
    indT: int | None = -1,
    *,
    ilon: int = 34,
    grid_file: str = "grid.nc",
    ice_file: str = "iceDiag.nc",
):
    """
    Extract sea-ice edge latitudes in the Southern Hemisphere
    and North Atlantic.

    Compatible with:
      - Dryad-style files: grid.nc, iceDiag.nc
      - MNC-style files: grid.glob.nc, iceDiag.glob.nc

    Parameters
    ----------
    dirF : str
        Directory containing grid and ice diagnostics.
    indT : int or None
        Time index if dimension 'T' exists. Use None for snapshots.
    ilon : int
        Longitude index separating Atlantic from Indo-Pacific.
    grid_file : str
        Preferred grid filename.
    ice_file : str
        Preferred ice diagnostics filename.

    Returns
    -------
    lats : float
        Northernmost sea-ice latitude in the Southern Hemisphere.
    latn : float
        Southernmost sea-ice latitude in the North Atlantic.
    """

    grid_path = resolve_nc(dirF, grid_file, "grid.glob.nc")
    grid, xgrid = mitgcm_tools.loadgrid(grid_path, basin_masks=False)

    ice_path = resolve_nc(dirF, ice_file, "iceDiag.glob.nc")
    icediag = open_nc(
        ice_path,
        strange_axes={"Zmd000001": "ZC", "Zd000001": "ZL"},
        grid=grid,
    )
    icediag.close()

    seaice = pick_time(icediag.SIarea, indT)

    # Keep only actual sea ice
    seaice = seaice.where(seaice > 0)

    # Southern Hemisphere ice edge
    zai_sh = seaice.mean("XC", skipna=True).compute()
    zai_sh = zai_sh.where(zai_sh.YC < 0, drop=True)

    if zai_sh.notnull().any():
        lats = float(zai_sh.where(zai_sh > 0, drop=True).YC.max())
    else:
        lats = np.nan

    # North Atlantic ice edge
    zai_natl = seaice.isel(XC=slice(0, ilon)).mean("XC", skipna=True).compute()
    zai_natl = zai_natl.where(zai_natl.YC > 0, drop=True)

    if zai_natl.notnull().any():
        latn = float(zai_natl.where(zai_natl > 0, drop=True).YC.min())
    else:
        latn = np.nan

    return lats, latn

def gen_BL(indT, rho0, Pref, dir_exp):
    """
    compute the buoyancy loss 

    :param:
    ......indT: time index for which the buoyancy loss is computed 
    ......rho0: reference density 
    ......Pref: reference pressure to compute the thermal expansion and haline contraction coeeficients 
    ...dir_exp: directory for experiment 

    :return:
    BL (m2/s3): buoyancy loss from the ocean (shape (Y,X))
    """

    ### load grid
    grid, xgrid = mitgcm_tools.loadgrid(dir_exp + 'grid.glob.nc', basin_masks=False)
    grid.close()
    ### load arrays
    ocediag     = mitgcm_tools.open_ncfile(dir_exp + 'oceDiag.glob.nc',\
      strange_axes={'Zmd000029':'ZC','Zld000029':'ZL'},grid=grid)
    ocediag.close()
    icediag     = mitgcm_tools.open_ncfile(dir_exp + 'iceDiag.glob.nc',\
      strange_axes={'Zmd000001':'ZC','Zd000001':'ZL'},grid=grid)
    icediag.close()

    ### load salinity and theta
    S       = ocediag.SALT.isel(ZC=0, T=indT)
    T       = ocediag.THETA.isel(ZC=0, T=indT)
    # calculates conservative temperature of seawater 
    # from potential temperature (whose reference sea pressure is zero dbar)
    # http://www.teos-10.org/pubs/gsw/html/gsw_CT_from_pt.html
    CT      = gsw.CT_from_pt(S,T)
    # Calculates the thermal expansion coefficient
    # and the saline contraction coefficient of seawater from
    # Absolute Salinity and Conservative Temperature.
    # http://www.teos-10.org/pubs/gsw/html/gsw_alpha.html
    # http://www.teos-10.org/pubs/gsw/html/gsw_beta.html
    alpha   = gsw.alpha(S,CT,Pref)
    beta    = gsw.beta(S,CT,Pref)

    ### load SIqnet and SIempmr
    SIqnet  = icediag.SIqnet.isel(T=indT)
    SIempmr = icediag.SIempmr.isel(T=indT)
    # Calculates the buoyancy loss 
    g       = 9.81
    Cp      = 3994
    BL      = grid.HFacC.isel(ZC=0) * (g / rho0) * (alpha * SIqnet / Cp + beta * SIempmr * S)

    return BL

def gen_PsiBT(dirF, indT):
    """
    compute barotropic streamfunction

    :param:
    ......dirF: work directory
    ......indT: time index

    :return:
    PsiBT (PW): barotropic streamfunction (shape (T,Yp1,X))
    """

    # load grid
    grid, xgrid = mitgcm_tools.loadgrid(dirF+'grid.glob.nc', basin_masks=False)
    grid.close()
    # load ocean variables
    ocediag   = mitgcm_tools.open_ncfile(dirF+'oceDiag.glob.nc',\
          strange_axes={'Zmd000029':'ZC','Zld000029':'ZL'},grid=grid)
    ocediag.close()
    # compute uBT 
    uBT       = (ocediag.UVELMASS.isel(T=indT) * grid.HFacW * grid.drF).sum('ZC')/ 1e6
    # compute barotropic streamfunction
    psiBT     = (-uBT * grid.dyG).cumsum('YC') 

    return psiBT

def gen_vel(
    dirF: str,
    indT: int | None = -1,
    *,
    grid_file: str = "grid.nc",
    oce_file: str = "oceDiag.nc",
    ):
    """
    Compute eddy and residual velocities.

    Compatible with:
      - standard MNC outputs (e.g., grid.glob.nc, oceDiag.glob.nc with diag_levels and T)
      - Dryad-style snapshots (e.g., grid.nc, oceDiag.nc without diag_levels and possibly without T)

    Parameters
    ----------
    dirF : str
        Directory containing the grid and ocean diagnostics.
    indT : int or None
        Time index (used only if dimension 'T' exists). Use None for snapshots.
    grid_file : str
        Grid filename (Dryad-style). Default "grid.nc".
    oce_file : str
        Ocean diagnostics filename (Dryad-style). Default "oceDiag.nc".

    Returns
    -------
    vgm : xarray.DataArray
        Eddy meridional velocity from GM parameterization (m/s),
        typically on (ZC, YG, XC) or equivalent staggered grid.
    vres : xarray.DataArray
        Residual meridional velocity v_res = v + v_eddy (m/s),
        same grid as VVELMASS.
    """

    # -------------------------
    # Load grid 
    # -------------------------
    grid_path = resolve_nc(dirF, grid_file, "grid.glob.nc")
    grid, xgrid = mitgcm_tools.loadgrid(grid_path, basin_masks=False)

    # -------------------------
    # Load ocean diagnostics
    # -------------------------
    oce_path = resolve_nc(dirF, oce_file, "oceDiag.glob.nc")
    ocediag = open_nc(
        oce_path,
        strange_axes={"Zmd000029": "ZC", "Zld000029": "ZL"},
        grid=grid,
    )
    ocediag.close()

    # -------------------------
    # Select time safely (works whether 'T' exists or not)
    # -------------------------
    psiY = pick_time(ocediag.GM_PsiY, indT)
    vvel = pick_time(ocediag.VVELMASS, indT)

    # -------------------------
    # Compute bolus velocity (GM) and residual velocity
    # -------------------------
    # xgrid.diff(..., 'Z') returns a vertical difference on the xgcm grid.
    # grid.drF is the vertical thickness (ZC-like), compatible with the diff output.
    vgm = grid.HFacS * xgrid.diff(psiY, "Z", boundary="fill") / grid.drF

    # Residual velocity
    vres = vgm + vvel

    return vgm, vres

def interp_v_to_rho_grid(VELO_da, grid, xgrid, hfacc, method="transport_south_face"):
    """
    Interpolate meridional velocity or transport from the native
    V-grid (YG / HFacS) onto the density grid (YC / HFacC).

    Parameters
    ----------
    VELO_da : xarray.DataArray
        Meridional velocity on the V-grid (typically vres, vgm, or VVELMASS).

    grid : xarray.Dataset
        MITgcm grid dataset.

    xgrid : xgcm.Grid
        xgcm grid object used for staggered-grid interpolation.

    hfacc : np.ndarray
        Cell-center ocean mask (HFacC-like) used to mask the final
        interpolated field on the YC grid.

    method : str
        Method used to map the velocity/transport from YG to YC.

        Options are:

        - "wall_zero":
            Closed V-faces are treated as V=0 before interpolation.
            This is the closest to the native MITgcm no-normal-flow
            wall interpretation.

        - "valid_only":
            Interpolate using only open ocean V-faces. Closed faces
            do not contribute to the interpolation, which reduces
            artificial damping near topography.

        - "transport":
            Interpolate volume transport (V * dx * dz) instead of
            velocity. This is more conservative for overturning
            circulation diagnostics.

        - "transport_north_face":
            Use the northern V-face transport directly for each YC
            grid cell without interpolation. This preserves transport
            amplitudes near boundaries and avoids smoothing by closed
            faces.

        - "transport_south_face":
            Use the southern V-face transport directly for each YC
            grid cell without interpolation.

    Returns
    -------
    FIELD : np.ndarray
        Interpolated velocity or transport field on the YC grid.

    is_transport : bool
        True if FIELD already includes dx*dz transport factors,
        False if velocity still needs to be multiplied by grid metrics.
    """

    mask_v = (grid.HFacS > 0).astype(float)

    if method == "wall_zero":
        # Closed faces remain zero: wall/no-normal-flow interpretation
        V_yc = xgrid.interp(VELO_da.where(mask_v > 0, 0.0), axis="Y")

        VELO = V_yc.values
        VELO[hfacc == 0] = np.nan

        return VELO, False  # False = still need dxF*drF later

    elif method == "valid_only":
        # Ignore closed faces in the interpolation
        num = xgrid.interp(VELO_da.where(mask_v > 0, 0.0), axis="Y")
        den = xgrid.interp(mask_v, axis="Y")

        V_yc = num / den.where(den > 0)

        VELO = V_yc.values
        VELO[hfacc == 0] = np.nan

        return VELO, False  # still need dxF*drF later

    elif method == "transport":
        # Interpolate transport directly
        transport_yg = VELO_da * grid.dxG * grid.drF

        num = xgrid.interp(transport_yg.where(mask_v > 0, 0.0), axis="Y")
        den = xgrid.interp(mask_v, axis="Y")

        transport_yc = num / den.where(den > 0)

        TRANSPORT = transport_yc.values
        TRANSPORT[hfacc == 0] = np.nan

        return TRANSPORT, True  # True = already includes dx*drF

    elif method == "transport_north_face":
        transport_yg = (VELO_da * grid.dxG * grid.drF).where(grid.HFacS > 0)

        FIELD = transport_yg.values[:, 1:, :]
        FIELD[hfacc == 0] = np.nan

        return FIELD, True

    elif method == "transport_south_face":
        transport_yg = (VELO_da * grid.dxG * grid.drF).where(grid.HFacS > 0)

        FIELD = transport_yg.values[:, :-1, :]
        FIELD[hfacc == 0] = np.nan
   
        return FIELD, True

    else:
        raise ValueError("method must be 'wall_zero', 'valid_only', or 'transport'")


def gen_potdens(
    dirF: str,
    indT: int | None = -1,
    Pref: float = 2000,
    *,
    grid_file: str = "grid.nc",
    oce_file: str = "oceDiag.nc",
):
    """
    Compute potential density anomaly (sigma = rho - 1000) using MITgcm jmd95.

    Compatible with:
      - standard MNC outputs (grid.glob.nc, oceDiag.glob.nc, with T + diag_levels)
      - Dryad-style snapshots (grid.nc, oceDiag.nc, without diag_levels and possibly without T)

    Parameters
    ----------
    dirF : str
        Directory containing grid and ocean diagnostics.
    indT : int or None
        Time index used only if dimension 'T' exists. Ignored for snapshot files without 'T'.
        Default = -1 (last record when time exists).
    Pref : float
        Reference pressure in dbar (default 2000).
    grid_file : str
        Dryad-style grid filename (default "grid.nc"). Falls back to "grid.glob.nc".
    oce_file : str
        Dryad-style oceDiag filename (default "oceDiag.nc"). Falls back to "oceDiag.glob.nc".

    Returns
    -------
    sigma : xarray.DataArray
        Potential density anomaly (kg/m3) = rho(Pref) - 1000, masked over topography.
        Typically on (ZC, YC, XC).
    """

    # -------------------------
    # Load grid 
    # -------------------------
    grid_path = resolve_nc(dirF, grid_file, "grid.glob.nc")
    grid, xgrid = mitgcm_tools.loadgrid(grid_path, basin_masks=False)

    # -------------------------
    # Load ocean diagnostics 
    # -------------------------
    oce_path = resolve_nc(dirF, oce_file, "oceDiag.glob.nc")
    ocediag = open_nc(
        oce_path,
        strange_axes={"Zmd000029": "ZC", "Zld000029": "ZL"},
        grid=grid,
    )
    ocediag.close()

    # -------------------------
    # Select time safely (works with or without 'T')
    # -------------------------
    S = pick_time(ocediag.SALT, indT)
    T = pick_time(ocediag.THETA, indT)

    # -------------------------
    # Compute sigma = rho - 1000
    # -------------------------
    # jmd95 returns a numpy array; wrap back into a DataArray using S as a template
    rho = jmd95.densjmd95(S.values, T.values, Pref)
    sigma = xr.DataArray(
        rho - 1000.0,
        dims=S.dims,
        coords=S.coords,
        name="sigma",
        attrs={
            "description": f"Potential density anomaly at {Pref:g} dbar",
            "units": "kg/m3",
            "reference_pressure_dbar": Pref,
        },
    )

    return sigma.where(grid.HFacC>0)

def make_sigma_bins(sigma, nsig: int = 80, a: float = 1.5):
    """
    Build density (sigma) classes from a sigma field, dense -> light.

    Parameters
    ----------
    sigma : array-like
        Potential density anomaly field (rho - 1000), can be np.ndarray or xarray.DataArray.
    nsig : int
        Number of bins (default 80).
    a : float
        Exponent controlling bin spacing (default 1.5).

    Returns
    -------
    dsig : np.ndarray (nsig,)
        Density classes, ordered from dense to light.
    minsig, maxsig : float
        Min/max used to scale the bins.
    """
    # Works for np arrays and xarray objects
    minsig = np.nanmin(np.asarray(sigma))
    maxsig = np.nanmax(np.asarray(sigma))

    sdflog = (np.logspace(-1, 1, nsig) / 10.0) ** a
    sdf = sdflog - sdflog[-1]
    dsig = ((sdf / sdf[0]) * (maxsig - minsig) + minsig)  

    return dsig, minsig, maxsig

def enforce_monotonic_z(zcol, fill_depth):
    """
    Enforce z[k+1] >= z[k] for a 1D vertical coordinate profile (dense -> light),
    where z is negative (e.g. -4000 bottom, -10 surface).

    Parameters
    ----------
    zcol : (nsig,) array
    fill_depth : float
        Depth used when nz==0 (typically deepest level, e.g. min(zc)).

    Returns
    -------
    z : (nsig,) array, monotonic non-decreasing.
    """
    z = zcol.copy()

    # Treat fill_depth as "missing" for monotonic fixing, but keep it if unavoidable
    mask_valid = np.isfinite(z) & (z != fill_depth)

    if not np.any(mask_valid):
        # all missing -> keep as is
        return z

    # Work on a copy where missing are set to NaN, fix only valid segments
    z_work = z.copy().astype(float)
    z_work[~mask_valid] = np.nan

    # Enforce non-decreasing where both finite
    for k in range(1, len(z_work)):
        if np.isfinite(z_work[k]) and np.isfinite(z_work[k-1]):
            if z_work[k] < z_work[k-1]:
                z_work[k] = z_work[k-1]

    # Put back: keep fixed valid values, keep fill_depth for missing
    z[mask_valid] = z_work[mask_valid]
    return z

def gen_rocz(dirF, indT, ilon):
    """
    TWO BASIN CASE
    compute residual overturning circulation in depth space
    Global, Atlantic and Indo-Pacific basins

    :param:
    ..........dirF: work directory
    ..........indT: time index
    ..........ilon: index of longitude that separates the 2 bassins 

    :return:
    .....rocz (Sv): global residual streamfunction (shape (T,Zi,Yp1,X), !on cell interface!)
    .rocz_atl (Sv): ATL residual streamfunction (shape (T,Zi,Yp1,X), !on cell interface!)
    rocz_ipac (Sv): Indo-PAC residual streamfunction (shape (T,Zi,Yp1,X), !on cell interface!)
    """

    # load grid
    grid, xgrid = mitgcm_tools.loadgrid(dirF+'grid.glob.nc', basin_masks=False)
    grid.close()
    # load ocean variables
    ocediag   = mitgcm_tools.open_ncfile(dirF+'oceDiag.glob.nc',\
          strange_axes={'Zmd000029':'ZC','Zld000029':'ZL'},grid=grid)
    ocediag.close()
    # get residual velocity
    vgm, vres = gen_vel(dirF, indT)
    # take zonal sum of v*dx*dz
    vdxdz     = (vres * grid.dxG * grid.drF * grid.HFacS).sum('XC')
    vdxdzA    = (vres * grid.dxG * grid.drF * grid.HFacS).isel(XC=slice(0,ilon)).sum('XC')
    vdxdzIPAC = (vres * grid.dxG * grid.drF * grid.HFacS).isel(XC=slice(ilon,None)).sum('XC')
    # compute bottom to surface integral for v
    rocz      = - vdxdz[::-1, :].cumsum('ZC')    [::-1, :] / 1e6 
    roczA     = - vdxdzA[::-1, :].cumsum('ZC')   [::-1, :] / 1e6 
    roczIPAC  = - vdxdzIPAC[::-1, :].cumsum('ZC')[::-1, :] / 1e6 
 
    return rocz, roczA, roczIPAC

def gen_rocsig2B(
    dirF,
    indT: int | None = -1,
    ilon: int = 34,
    Pref: float = 2000,
    nsig: int = 80,
    a: float = 1.5,
    flag_roc: int = 0,
    *,
    grid_file="grid.nc",
    oce_file="oceDiag.nc",
    ):
    """
    TWO BASIN CASE
    Compute residual overturning circulation in density space
    for the Global, Atlantic and Indo-Pacific basins.

    Compatible with:
      - standard MNC outputs (grid.glob.nc, oceDiag.glob.nc; may include T)
      - Dryad snapshots (grid.nc, oceDiag.nc; may omit T and diag_levels)

    Parameters
    ----------
    dirF : str
        Working directory (must contain grid + oceDiag; for Dryad, usually PB folder).
    indT : int or None
        Time index used only if dimension 'T' exists. Ignored for snapshot files without 'T'.
        Default = -1 (last record when time exists).
    ilon : int
        Longitude index separating Atlantic from Indo-Pacific (default=34 for 128x80 grid).
    Pref : float
        Reference pressure in dbar (default 2000).
    nsig : int
        Number of density classes (default 80).
    a : float
        Exponent to build density classes (default 1.5).
    flag_roc : int
        0 -> use residual velocity vres (default)
        1 -> use Eulerian velocity v (VVELMASS)
        2 -> use GM bolus velocity vgm
    grid_file, oce_file : str
        Preferred Dryad filenames; function will fall back to standard *.glob.nc automatically.

    Returns
    -------
    mocsig, mocsigA, mocsigIPAC, zsig, zsigA, zsigIPAC : np.ndarray
    """

    # -------------------------
    # Load grid (nc first, fallback to glob.nc via resolve_nc)
    # -------------------------
    grid_path = resolve_nc(dirF, grid_file, "grid.glob.nc")
    grid, xgrid = mitgcm_tools.loadgrid(grid_path, basin_masks=False)
    grid.close()

    # Pull grid metrics as numpy arrays
    zc  = grid.RC.values 

    # -------------------------
    # Potential density (sigma) at Pref
    # -------------------------
    sigma_da = gen_potdens(dirF, indT, Pref, grid_file=grid_file, oce_file=oce_file)
    sigma = sigma_da.values

    ny = sigma.shape[1]
    nx = sigma.shape[2]

    # -------------------------
    # Density classes (dense -> light)
    # -------------------------
    dsig, minsig, maxsig = make_sigma_bins(sigma_da.where(grid.HFacC>0).values, nsig=nsig, a=a)
    
    # -------------------------
    # Velocities (GM + residual)
    # -------------------------
    vgm_da, vres_da = gen_vel(dirF, indT, grid_file=grid_file, oce_file=oce_file)

    if flag_roc == 0:
        VELO_da = vres_da
    elif flag_roc == 2:
        VELO_da = vgm_da
    elif flag_roc == 1:
        # Load Eulerian meridional velocity from oceDiag
        oce_path = resolve_nc(dirF, oce_file, "oceDiag.glob.nc")
        ocediag = open_nc(
            oce_path,
            strange_axes={"Zmd000029": "ZC", "Zld000029": "ZL"},
            grid=grid,
        )
        ocediag.close()
        VELO_da = pick_time(ocediag.VVELMASS, indT)
    else:
        raise ValueError("flag_roc must be 0 (vres), 1 (v), or 2 (vgm).")

    ## Compute the transport in each cell
    transport       = (VELO_da * grid.dxG * grid.drF).where(grid.HFacS > 0)
    ## Convert to numpy for your loops and get the south face
    transport_south = transport.values[:, :-1, :]

    # -------------------------
    # Allocate output arrays
    # -------------------------
    mocsig     = np.zeros((nsig, ny))
    mocsigA    = np.zeros((nsig, ny))
    mocsigIPAC = np.zeros((nsig, ny))

    zsig       = np.zeros((nsig, ny))
    zsigA      = np.zeros((nsig, ny))
    zsigIPAC   = np.zeros((nsig, ny))

    # Deepest level used to fill missing sigma classes
    fill_depth = np.nanmin(zc)

    # -------------------------
    # Main loops (kept as-is)
    # -------------------------
    for k in range(nsig):
        for j in range(ny):
            mocrho     = 0.0
            mocrhoIPAC = 0.0
            zrho       = 0.0
            zrhoIPAC   = 0.0
            nz         = 0
            nzIPAC     = 0

            for i in range(nx):
                # Interpolate density bins to depths at this (j,i)
                zdsig = np.interp(dsig, sigma[:, j, i], zc)

                ind = np.where(sigma[:, j, i] >= dsig[k])[0]
                if ind.size != 0:
                    contrib = np.nansum(transport_south[ind, j, i])
                    mocrho += contrib

                    zmax = zdsig[k]
                    zrho += zmax
                    nz += 1

                    if i > ilon - 1:
                        mocrhoIPAC += contrib
                        zrhoIPAC += zmax
                        nzIPAC += 1

                if i == ilon - 1:
                    if nz == 0:
                        zsigA[k, j] = fill_depth 
                    else:
                        zsigA[k, j] = zrho / nz
                        mocsigA[k, j] = -mocrho / 1e6

            # Global
            if nz == 0:
                zsig[k, j] = fill_depth
            else:
                zsig[k, j] = zrho / nz
                mocsig[k, j] = -mocrho / 1e6

            # IPAC
            if nzIPAC == 0:
                zsigIPAC[k, j] = fill_depth
            else:
                zsigIPAC[k, j] = zrhoIPAC / nzIPAC
                mocsigIPAC[k, j] = -mocrhoIPAC / 1e6

    # -------------------------
    # Fix depth inversions 
    # -------------------------
    for j in range(ny):
        zsig[:, j]     = enforce_monotonic_z(zsig[:, j], fill_depth)
        zsigA[:, j]    = enforce_monotonic_z(zsigA[:, j], fill_depth)
        zsigIPAC[:, j] = enforce_monotonic_z(zsigIPAC[:, j], fill_depth)

    return mocsig, mocsigA, mocsigIPAC, zsig, zsigA, zsigIPAC

def gen_rocsig2B_SO(
    dirF,
    indT: int | None = -1,
    ilon: int = 34,
    Pref: float = 2000,
    nsig: int = 80,
    a: float = 1.5,
    flag_roc: int = 0,
    latSO: float = -51,
    *,
    grid_file="grid.nc",
    oce_file="oceDiag.nc",
    surf_file="surfDiag.nc",
    ):
    """
    TWO BASIN CASE
    Compute residual overturning circulation in density space
    in the Southern Ocean for the Global basin.

    Compatible with:
      - standard MNC outputs (grid.glob.nc, oceDiag.glob.nc; may include T)
      - Dryad snapshots (grid.nc, oceDiag.nc; may omit T and diag_levels)

    Parameters
    ----------
    dirF : str
        Working directory (must contain grid + oceDiag; for Dryad, usually PB folder).
    indT : int or None
        Time index used only if dimension 'T' exists. Ignored for snapshot files without 'T'.
        Default = -1 (last record when time exists).
    ilon : int
        Longitude index separating Atlantic from Indo-Pacific (default=34 for 128x80 grid).
    Pref : float
        Reference pressure in dbar (default 2000).
    nsig : int
        Number of density classes (default 80).
    a : float
        Exponent to build density classes (default 1.5).
    flag_roc : int
        0 -> use residual velocity vres (default)
        1 -> use Eulerian velocity v (VVELMASS)
        2 -> use GM bolus velocity vgm
    latSO : float
        Northern boundary latitude of the Southern Ocean channel (default -51).
    grid_file, oce_file, surf_file : str
        Preferred Dryad filenames; function will fall back to standard *.glob.nc automatically.

    Returns
    -------
    mocsig : np.ndarray
    """

    # -------------------------
    # Load grid (nc first, fallback to glob.nc via resolve_nc)
    # -------------------------
    grid_path = resolve_nc(dirF, grid_file, "grid.glob.nc")
    grid, xgrid = mitgcm_tools.loadgrid(grid_path, basin_masks=False)
    grid.close()

    # Pull grid metrics as numpy arrays
    YC = grid.YC.values
    zc  = grid.RC.values
    hfacc = grid.HFacC.values.copy()

    # -------------------------
    # Potential density (sigma) at Pref
    # -------------------------
    sigma = gen_potdens(dirF, indT, Pref, grid_file=grid_file, oce_file=oce_file)

    ny = sigma.sizes["YC"]
    nx = sigma.sizes["XC"]

    # --- load MLD
    surf_path = resolve_nc(dirF, surf_file, "surfDiag.glob.nc")
    surfdiag  = open_nc(
        surf_path,
        strange_axes={"Zmd000001": "ZC", "Zd000001": "ZL"},
        grid=grid
    )
    MLDc = pick_time(surfdiag.MXLDEPTH, indT)
    surfdiag.close()

    # Convert to numpy before loops
    MLDc_np = MLDc.values
    sigma_np = sigma.values.copy()

    j_idx_c = np.where(YC <= latSO)[0]

    # Mask density below the MLD on YC
    for i in range(nx):
        for j in j_idx_c:
            izc = int(np.abs(zc + MLDc_np[j, i]).argmin()) 

            if izc + 1 < len(zc):
                sigma_np[izc + 1:, j, i] = np.nan
                hfacc[izc + 1:, j, i] = 0.0

    # Remove velocities north of latSO
    j_north_c = np.where(YC > latSO)[0]
    hfacc[:, j_north_c, :] = 0.0

    # -------------------------
    # Density classes 
    # -------------------------
    sigma_SO = sigma_np[:, j_idx_c, :] 
    dsig_dense_to_light, minsig, maxsig = make_sigma_bins(sigma_SO, nsig=nsig, a=a)
    dsig = dsig_dense_to_light[::-1]
    # -------------------------
    # Velocities (GM + residual)
    # -------------------------
    vgm_da, vres_da = gen_vel(dirF, indT, grid_file=grid_file, oce_file=oce_file)

    if flag_roc == 0:
        VELO_da = vres_da
    elif flag_roc == 2:
        VELO_da = vgm_da
    elif flag_roc == 1:
        # Load Eulerian meridional velocity from oceDiag
        oce_path = resolve_nc(dirF, oce_file, "oceDiag.glob.nc")
        ocediag = open_nc(
            oce_path,
            strange_axes={"Zmd000029": "ZC", "Zld000029": "ZL"},
            grid=grid,
        )
        ocediag.close()
        VELO_da = pick_time(ocediag.VVELMASS, indT)
    else:
        raise ValueError("flag_roc must be 0 (vres), 1 (v), or 2 (vgm).")

    ## Compute the transport in each cell
    transport       = (VELO_da * grid.dxG * grid.drF).where(grid.HFacS > 0)
    ## Convert to numpy for your loops and get the south face
    transport_south = transport.values[:, :-1, :]
    ## Apply mask 
    transport_south[hfacc == 0] = np.nan
 
    # -------------------------
    # Allocate output arrays
    # -------------------------
    mocsig = np.full((nsig, ny), np.nan)

    for j in j_idx_c:
        for k in range(nsig):
            mocrho = 0.0
            nz = 0

            for i in range(nx):
                sig_prof = sigma_np[:,j,i]

                ind = np.where(sig_prof <= dsig[k])[0]

                if ind.size != 0:
                    mocrho += np.nansum(transport_south[ind, j, i])
                    nz += 1

            if nz > 0:
                mocsig[k, j] = mocrho / 1e6

        if np.any(~np.isnan(mocsig[:, j])):
            indnan = np.where(~np.isnan(mocsig[:, j]))[0][0]
            mocsig[indnan, j] = 0.0

    return mocsig, dsig

def dens_rocATL(dirF, rocfile, ilat, ilon): 
    """
    Compute densest density upwelling in North Atlantic and density at
    the maximum of the residual circulation in the North Atlantic 

    :param:
    .........dirF: work directory
    .....rocfile : name file for rocsig 
    .........ilat: index of latitude of Southern Ocean limit 
    .........ilon: index of longitude that separates the 2 bassins 

    :return:
    sigmn (kg/m3): density where the residual circulation is maximum in the North Atlantic  
    sigmx (kg/m3): densest density that upwells in the North Atlantic  
    """
  
    ### load variables in netcdf file
    ds = xr.open_dataset(dirF + rocfile) 
    # extract dimensions
    ny = len(ds.YC)
    nz = len(ds.ZC)
    ### zonal-averaged of potential density in ATL
    # initialize values
    zosigA = np.zeros((nz,ny))
    # take the zonal mean
    zosigA[:,:ilat] = ds.sigma.isel(YC=slice(None,ilat)).mean('XC')
    zosigA[:,ilat:] = ds.sigma.isel(YC=slice(ilat,None),XC=slice(None,ilon)).mean('XC')
    # latitude index north of Equator
    indN = np.where(ds.YC>0)[0]
    ### Find z-index such that z<-800 in NATL
    ind800 = np.where(ds.zsigA.values[:,indN]<-800)
    ### Compute max(psires(NATL))
    mrocNATL = np.nanmax(ds.rocsigA.values[:,indN][ind800])
    ### Extract index where max(psires(NATL))
    indmroc = np.where(ds.rocsigA==mrocNATL)
    # for the latitude
    indy = indmroc[1][0]
    # for the depth
    indz = np.abs(zosigA[:,indy] - ds.SIG.isel(SIG=indmroc[0][0]).values).argmin()
    # density where maximum of psi_res in NATL
    sigmn = np.round(zosigA[indz,indy],1)
    # densest density that upwells in NATL
    sigmx = np.round(np.nanmax(zosigA[0,indN]),1)
    if sigmx < sigmn: 
       tmp   = sigmn
       sigmn = sigmx
       sigmx = tmp

    return sigmn, sigmx

def gen_rocsig2B_SO_v0(
    dirF,
    indT=-1,
    ilon=34,
    Pref=2000,
    nsig=80,
    a=1.5,
    *,
    grid_file="grid.nc",
    oce_file="oceDiag.nc",
    surf_file="surfDiag.nc",
):
    grid_path = resolve_nc(dirF, grid_file, "grid.glob.nc")
    grid, xgrid = mitgcm_tools.loadgrid(grid_path, basin_masks=False)
    grid.close()

    dxv   = grid["dxG"].values
    dzc   = grid["drF"].values
    zc    = grid["RC"].values
    hfacv = grid["HFacS"].values

    ilat = 11

    def find_nearest_value(array, value):
        array = np.asarray(array)
        return int(np.abs(array - value).argmin())

    # density, using new reader
    sigma = gen_potdens(
        dirF,
        indT,
        Pref,
        grid_file=grid_file,
        oce_file=oce_file,
    )

    # MLD, using new reader
    surf_path = resolve_nc(dirF, surf_file, "surfDiag.glob.nc")
    surfdiag = open_nc(
        surf_path,
        strange_axes={"Zmd000001": "ZC", "Zd000001": "ZL"},
        grid=grid,
    )

    MLDc = pick_time(surfdiag["MXLDEPTH"], indT)
    MLDg = xgrid.interp(MLDc, axis="Y")
    surfdiag.close()

    ny, nx = MLDc.shape

    # exact old masking logic
    for ii in range(nx):
        for jj in range(ilat + 2):
            izc = find_nearest_value(zc, -MLDc.isel(YC=jj, XC=ii).values)

            if izc + 1 < len(zc):
                sigma_subset = sigma.isel(YC=jj, XC=ii)
                sigma.isel(YC=jj, XC=ii)[:] = sigma_subset.where(
                    sigma["ZC"] > sigma["ZC"].isel(ZC=izc + 1),
                    np.nan,
                )

            izg = find_nearest_value(zc, -MLDg.isel(YG=jj, XC=ii).values)
            hfacv[izg + 1 :, jj, ii] = 0

    hfacv[:, ilat:, :] = 0

    minsig = sigma.isel(YC=slice(0, ilat)).min().values
    maxsig = sigma.isel(YC=slice(0, ilat)).max().values

    sdflog = (np.logspace(-1, 1, nsig) / 10) ** a
    sdf = sdflog - sdflog[-1]

    dsig = ((sdf / sdf[0]) * (maxsig - minsig) + minsig)[::-1]

    vgm, vres = gen_vel(
        dirF,
        indT,
        grid_file=grid_file,
        oce_file=oce_file,
    )

    VELO = vres.values
    VELO[np.where(hfacv == 0)] = np.nan

    mocsig = np.nan * np.ones((nsig, ny))

    for j in range(ny):
        for k in range(nsig):
            mocrho = 0.0
            zrho = 0.0
            nz = 0

            for i in range(nx):
                zdsig = np.interp(dsig, sigma.isel(YC=j, XC=i), zc)

                ind = np.where(sigma.isel(YC=j, XC=i) <= dsig[k])

                if len(ind[0]) != 0:
                    zmax = zdsig[k]

                    mocrho = np.nansum([
                        mocrho,
                        np.nansum(VELO[ind, j, i] * dxv[j, i] * dzc[ind])
                    ])

                    zrho = np.nansum([zrho, zmax])
                    nz = nz + 1

            if nz > 0:
                mocsig[k, j] = mocrho / 1e6

        if np.any(~np.isnan(mocsig[:, j])):
            indnan = np.where(~np.isnan(mocsig[:, j]))[0][0]
            mocsig[indnan, j] = 0

    return mocsig, dsig
