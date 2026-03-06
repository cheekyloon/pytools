#!/usr/bin/env python 

### import modules
#import xgcm
import mitgcm_tools
import numpy             as np
import xarray            as xr
import diagtools         as dg
import MITgcmutils.jmd95 as jmd95
from utils   import resolve_nc
from utils   import open_nc
from utils   import pick_time

def load_ice_lat(dir_exp, indT,ilon):
    """
    extract the maximum latitude of the zonal-averaged 
    sea ice edge in the South Hemisphere (lats)
    and the North Atlantic (latn)
    return lats and latn 

    :param:
    dir_exp: directory path 
    ...indT: time index 
    ...ilon: longitude index for Africa 

    :return:
    lats: latitude of the sea-ice in SH
    latn: latitude of the sea-ice in NATL
    """
    ### load grid
    grid, xgrid = mitgcm_tools.loadgrid(dir_exp + 'grid.glob.nc', basin_masks=False)
    grid.close()
    ### load ice fraction
    seaice = mitgcm_tools.open_ncfile(dir_exp + 'iceDiag.glob.nc',\
        strange_axes={'Zmd000001':'ZC','Zd000001':'ZL'},grid=grid)['SIarea'].isel(T=indT)
    # Mask where there is no ice
    seaice = seaice.where(seaice > 0)
    # Find latitude of ice in SH
    # Mean over longitude (XC axis)
    ZAI_SH = seaice.mean(dim='XC')
    lats   = ZAI_SH.isel(YC=slice(0, 35)).where(ZAI_SH > 0, drop=True).YC.max().item()
    # Find latitude of ice in NATL 
    # Mean over longitude (restricted to ilon)
    ZAI_NATL       = seaice.isel(XC=slice(0, ilon)).mean(dim='XC')
    ZAI_NATL_north = ZAI_NATL.isel(YC=slice(35, None)).where(ZAI_NATL > 0, drop=True)
    if ZAI_NATL_north.size > 0:
        latn = ZAI_NATL_north.YC.min().item()
    else:
        # Default to last latitude if no ice
        latn =  grid.YC.isel(YC=-1)

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
        dg.join(dirF, oce_file),
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

    return sigma.where(grid.HFacC>0,0.0)

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
    dsig = (sdf / sdf[0]) * (maxsig - minsig) + minsig  # dense -> light

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
    dxc = grid.dxF.values     
    dzc = grid.drF.values
    zc  = grid.RC.values         # negative

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

    # --- Convert VELO from YG to YC (ny=80) to match sigma
    # VELO_da dims: (ZC, YG, XC)
    VELOc_da = xgrid.interp(VELO_da, "Y").where(grid.HFacC > 0).fillna(0.0) 

    # Convert to numpy for your loops
    VELO = VELOc_da.values

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
                if ind.size > 0:
                    contrib = np.nansum(VELO[ind, j, i] * dxc[j, i] * dzc[ind])
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
    dxc = grid.dxF.values     
    dzc = grid.drF.values
    zc  = grid.RC.values         # negative

    # -------------------------
    # Potential density (sigma) at Pref
    # -------------------------
    sigma_da = gen_potdens(dirF, indT, Pref, grid_file=grid_file, oce_file=oce_file)
    sigma_np = sigma_da.values

    ny = sigma.shape[1]
    nx = sigma.shape[2]

    # --- load MLD
    surf_path = resolve_nc(dirF, surf_file, "surfDiag.glob.nc")
    surfdiag  = open_nc(
        surf_path,
        strange_axes={"Zmd000001": "ZC", "Zd000001": "ZL"},
        grid=grid
    )
    MLD = pick_time(surfdiag.MXLDEPTH, indT)
    surfdiag.close()

    # --- build masks
    SO_mask = (grid.YC <= latSO)
    in_mld  = (grid.ZC >= -MLD)
    keep    = SO_mask & in_mld

    # --- apply mask
    hfacc = grid.HFacC.where(keep, 0.0).values
    sigma_subset = sigma_da.where(keep).values

    # -------------------------
    # Density classes (dense -> light)
    # -------------------------
    dsig, minsig, maxsig = make_sigma_bins(sigma_subset, nsig=nsig, a=a)

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

    # --- Convert VELO from YG to YC (ny=80) to match sigma and apply mask
    # VELO_da dims: (ZC, YG, XC)
    VELOc_da = xgrid.interp(VELO_da, "Y").where(keep,0.0) 

    # Convert to numpy for your loops
    VELO = VELOc_da.values

    # -------------------------
    # Allocate output arrays
    # -------------------------
    mocsig = np.zeros((nsig, ny))

    # indices lat au sud de latSO
    j_idx = np.where(grid.YC.values <= latSO)[0]

    for j in j_idx:
        for k in range(nsig):
            mocrho = 0.0
            for i in range(nx):
                # Find index of lighter waters
                ind = np.where(sigma_np[:, j, i] <= dsig[k])[0]
                if ind.size:
                    mocrho += np.nansum(VELO[ind, j, i] * dxc[j, i] * dzc[ind])

            # Global 
            mocsig[k, j] = mocrho / 1e6

    return mocsig

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


