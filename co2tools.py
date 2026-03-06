#!/usr/bin/env python

###### import modules
import os
import gsw
import co2sys
import numpy     as np
import diagtools as dg
import xarray    as xr
import mitgcm_tools
from scipy   import ndimage as nd
from sklearn import linear_model

def write_to_binary(data, filename, precision='single'):
    """
    write variable from np.array to filename with precision 

    :param:
    .....data: data you want to write in the file
    .filename: name of file (e.g., pickup_ptracers.0000000001.data)
    precision: single or double
    """
    # write data to binary files
    fid   = open(filename, "wb")
    flatdata = data.flatten()
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

def getPdbar(
    dirF,
    flag: int = 0,
    indT: int | None = -1,
    grid_file="grid.nc",
    oce_file="oceDiag.nc",
    ):
    """
    Return pressure in dbar.

    Parameters
    ----------
    dirF : str
        Directory containing the grid and (optionally) ocean diagnostics.
    flag : int
        0 -> use gsw.p_from_z (pressure from depth)
        1 -> use model hydrostatic pressure from density anomaly (RHOAnoma)
    indT : int or None
        Time index used only if ocean diagnostics contain dimension 'T'.
        If files are snapshots without 'T', this argument is ignored.
    grid_file : str
        Grid filename (Dryad default: "grid.nc"; legacy might be "grid.glob.nc").
    oce_file : str
        Ocean diagnostics filename (Dryad default: "oceDiag.nc"; legacy might be "oceDiag.glob.nc").

    Returns
    -------
    Pdbar : xarray.DataArray
        Pressure in dbar (same 3D shape as ZC/YC/XC).
    """
    # --- load grid
    grid_path = resolve_nc(dirF, grid_file, "grid.glob.nc")
    grid, xgrid = mitgcm_tools.loadgrid(grid_path, basin_masks=False)

    # --- compute pressure in dbar
    if flag == 0:
        # Pressure from depth (z) and latitude.
        # Note: gsw.p_from_z expects z in meters (negative below sea level),
        # and lat in degrees.
        Pdbar = gsw.p_from_z(grid.ZC, grid.YC)

    elif flag == 1:
        # Load ocean variables needed for hydrostatic pressure
        oce_path = resolve_nc(dirF, oce_file, "oceDiag.glob.nc")
        ocediag = dg.open_nc(
            oce_path,
            strange_axes={"Zmd000029": "ZC", "Zld000029": "ZL"},
            grid=grid
        )

        rho = grid.HFacC * (pick_time(ocediag.RHOAnoma, indT) + 1035.0)  # kg/m3
        # hydrostatic pressure increment ~ g * rho * dz
        grhodz = 9.81 * grid.ZC * rho

        # cumulative integral from bottom upward (reverse ZC, cumsum, reverse back)
        Pdbar = grhodz[::-1, :, :].cumsum("ZC")[::-1, :, :]

    else:
        raise ValueError("flag must be 0 or 1")

    return grid.HFacC * Pdbar

def satO2(dirS, indT, dirT=None):
    """Compute oxygen saturation from temperature and salinity
    Oxygen saturation value is the volume of oxygen gas absorbed from humidity-saturated
    air at a total pressure of one atmosphere, per unit volume of the liquid at the temperature
    of measurement (ml/l)
    :param:
    .dirS: work directory for salinity 
    .indT: time index 
    .dirT: work directory for temperature

    :return:
    satO2: Saturation concentration of dissolved O2 (units="mol/m3", shape (Z,Y,X))
    """

    # Assign defaults if not provided
    if dirT is None:
        dirT  = dirS

    ### load grid
    grid, xgrid = mitgcm_tools.loadgrid(dirS + 'grid.glob.nc', basin_masks=False)
    grid.close()
    ### load arrays
    ocediagS    = mitgcm_tools.open_ncfile(dirS + 'oceDiag.glob.nc',\
      strange_axes={'Zmd000029':'ZC','Zld000029':'ZL'},grid=grid)
    ocediagS.close()
    ocediagT    = mitgcm_tools.open_ncfile(dirT + 'oceDiag.glob.nc',\
      strange_axes={'Zmd000029':'ZC','Zld000029':'ZL'},grid=grid)
    ocediagT.close()
    ### load salinity and theta
    S   = ocediagS.SALT.isel(T=indT) 
    T   = ocediagT.THETA.isel(T=indT) 

    oA0   =  2.00907
    oA1   =  3.22014
    oA2   =  4.05010
    oA3   =  4.94457
    oA4   = -2.56847E-1
    oA5   =  3.88767
    oB0   = -6.24523E-3
    oB1   = -7.37614E-3
    oB2   = -1.03410E-2
    oB3   = -8.17083E-3
    oC0   = -4.88682E-7

    aTT   = 298.15-T
    aTK   = 273.15+T
    aTS   = np.log(aTT/aTK)
    aTS2  = aTS*aTS
    aTS3  = aTS2*aTS
    aTS4  = aTS3*aTS
    aTS5  = aTS4*aTS

    ocnew = grid.HFacC * np.exp(oA0 + oA1*aTS + oA2*aTS2 + oA3*aTS3 + oA4*aTS4 + oA5*aTS5
        + S*(oB0 + oB1*aTS + oB2*aTS2 + oB3*aTS3) + oC0*(S*S))

    ### Saturation concentration of dissolved O2 (mol/m3)
    return ocnew/22391.6*1000.0

def MRL_alk(dirF, indT, dirS=None):
    """Compute the multilinear regression Alk_surf = a1 + a2 * SSS + a3 * PO_surf
    to obtain preformed Alkalinity. PO = O2 + 170 * PO4. 
    
    :param:
    ......dirF: work directory 
    ......indT: time index 
    ......dirS: work directory for salinity

    :return:
    a1, a2, a3: Alk_surf = a1 + a2 * SSS + a3 * PO_surf
    """
    
    # Assign defaults if not provided
    if dirS is None:
        dirS  = dirF

    ### load grid
    grid, xgrid = mitgcm_tools.loadgrid(dirF + 'grid.glob.nc', basin_masks=False)
    grid.close()
    # load salinity file
    ocediag  = mitgcm_tools.open_ncfile(dirS + 'oceDiag.glob.nc',\
      strange_axes={'Zmd000029':'ZC','Zld000029':'ZL'},grid=grid)
    ocediag.close()
    # load salinity 
    S        = grid.HFacC.where(grid.HFacC>0) * ocediag.SALT.isel(T=indT) 
    # load DIC file
    dicdiag  = mitgcm_tools.open_ncfile(dirF + 'dicDiag.glob.nc',\
           strange_axes={'Zmd000029':'ZC'},grid=grid)
    dicdiag.close()
    # problem with YC values in dicdiag
    dicdiag.coords['YC'] = grid.YC
    # load Alk, PO4 and O2 (mol/m3)
    Alk      = grid.HFacC.where(grid.HFacC>0) * dicdiag.TRAC02.isel(T=indT) 
    PO4      = grid.HFacC.where(grid.HFacC>0) * dicdiag.TRAC03.isel(T=indT) 
    O2       = grid.HFacC.where(grid.HFacC>0) * dicdiag.TRAC05.isel(T=indT) 
    # At the surface
    PO   = O2 + 170 * PO4
    SSS  = S.isel(ZC=0).values
    SPO  = PO.isel(ZC=0).values
    SAlk = Alk.isel(ZC=0).values
    # remove nan 
    SSS  = SSS[~np.isnan(SSS)]
    SPO  = SPO[~np.isnan(SPO)]
    SAlk = SAlk[~np.isnan(SAlk)]
    X    = np.transpose(np.array([SSS, SPO]))
    SAlk = SAlk.reshape(len(SAlk),1)
    regr = linear_model.LinearRegression()
    regr.fit(X, SAlk)
    a1   = regr.intercept_[0]
    a2   = regr.coef_[0][0]
    a3   = regr.coef_[0][1]

    return a1, a2, a3 

def get_pSi(
    dirF,
    indT: int | None = -1,
    grid_file="grid.nc",
    dic_file="dicDiag.nc",
    ):
    """
    Compute preformed silicate proxy from preformed phosphate.

    This version supports:
      - time-series files with dimension "T"
      - snapshot files without time axis

    Parameters
    ----------
    dirF : str
        Directory containing grid and dic diagnostics.
    indT : int or None
        Time index if 'T' dimension exists. Ignored if no 'T'.
    grid_file : str
        Grid filename (Dryad default: "grid.nc"; legacy: "grid.glob.nc").
    dic_file : str
        DIC diagnostics filename (Dryad default: "dicDiag.nc"; legacy: "dicDiag.glob.nc").

    Returns
    -------
    pSi : xarray.DataArray
        Preformed silicate proxy (mol/m3), shape (Z, Y, X).
    """
    # --- load grid
    grid_path = resolve_nc(dirF, grid_file, "grid.glob.nc")
    grid, xgrid = mitgcm_tools.loadgrid(grid_path, basin_masks=False)

    # --- load dic diagnostics
    dic_path = resolve_nc(dirF, dic_file, "oceDiag.glob.nc")
    dicdiag = dg.open_nc(
        dic_path,
        strange_axes={"Zmd000029": "ZC"},
        grid=grid
    )

    # Fix YC coordinate if needed
    dicdiag.coords["YC"] = grid.YC

    # --- extract preformed phosphate (TRAC08)
    pPO4 = pick_time(dicdiag.TRAC08, indT)

    # --- compute preformed silicate proxy
    pSi = 15.0 * pPO4

    return pSi

def get_pSi_AOU(dirF, indT, dirS=None, dirT=None):
    """Compute the preformed silicate with AOU 
 
    :param:
    ..........dirF: work directory 
    ..........indT: time index 
    ..........dirS: work directory for salinity
    ..........dirT: work directory for temperature

    :return:
    presi (mol/m3): preformed silicate (shape (Z,Y,X)) 
    """

    # Assign defaults if not provided
    if dirS is None:
        dirS  = dirF
    # Assign defaults if not provided
    if dirT is None:
        dirT  = dirF

    ### load grid
    grid, xgrid = mitgcm_tools.loadgrid(dirF + 'grid.glob.nc', basin_masks=False)
    grid.close()
    ### load silicate
    dir_sil = '/Users/sandy/Documents/ISMER/Postdoc_LPN/carbon_project/mingan/Silicate/'
    file_sil = dir_sil + 'silicate_82.bin'
    raw = np.fromfile(file_sil, dtype='>d')
    sil = np.reshape(raw, grid.HFacC.shape)
    # load DIC file
    dicdiag  = mitgcm_tools.open_ncfile(dirF + 'dicDiag.glob.nc',\
           strange_axes={'Zmd000029':'ZC'},grid=grid)
    dicdiag.close()
    # problem with YC values in dicdiag
    dicdiag.coords['YC'] = grid.YC
    ### load O2 in (mol/m3) 
    O2       = dicdiag.TRAC05.isel(T=indT) 
    ### Compute saturated O2
    O2sat    = dg.satO2(dirS, indT, dirT)
    ### compute preformed silicate
    #pSi      = grid.HFacC * (sil - (15/170) * (O2 - O2sat))
    pSi      = grid.HFacC * (sil - (15/170) * (O2sat - O2))
    
    return pSi

def get_Csat(dirF, indT, AOU=False, dirT=None, dirS=None, pco2=None):
    """Compute the saturated DIC (mol/m3) 
 
    :param:
    dir_exp: directory for experiment 
    ...indT: time index for experiment 
    ....AOU: method to compute Csat
                   the default is preformed tracers
                   AOU estimate is an option 
    ...dirS: work directory for salinity
    ...dirT: work directory for temperature
    ...pco2: pco2 value in atm 

    :return:
    Csat (mol/m3): saturated DIC (shape (Z,Y,X)) 
    """

    # Assign defaults if not provided
    if dirS is None:
        dirS    = dirF
    # Assign defaults if not provided
    if dirT is None:
        dirT    = dirF

    # Change units from the input of mol/m^3 -> mol/kg:
    # (1 mol/m^3) x (1 /1024.5 kg/m^3)
    permil=1/1024.5
    # convert mol/m3 to umol/kg
    permumolkg=permil*1e6

    ### load grid
    grid, xgrid = mitgcm_tools.loadgrid(dirF + 'grid.glob.nc', basin_masks=False)
    grid.close()
    ### load arrays
    ocediagS    = mitgcm_tools.open_ncfile(dirS + 'oceDiag.glob.nc',\
      strange_axes={'Zmd000029':'ZC','Zld000029':'ZL'},grid=grid)
    ocediagS.close()
    ocediagT    = mitgcm_tools.open_ncfile(dirT + 'oceDiag.glob.nc',\
      strange_axes={'Zmd000029':'ZC','Zld000029':'ZL'},grid=grid)
    ocediagT.close()
    dicdiag     = mitgcm_tools.open_ncfile(dirF + 'dicDiag.glob.nc',\
      strange_axes={'Zmd000029':'ZC'},grid=grid)
    dicdiag.close()
    surfdiag    = mitgcm_tools.open_ncfile(dirF + 'surfDiag.glob.nc',\
      strange_axes={'Zmd000001':'ZC','Zd000001':'ZL'},grid=grid)
    surfdiag.close()
    ### load salinity and theta
    S     = ocediagS.SALT.isel(T=indT) 
    T     = ocediagT.THETA.isel(T=indT)
    ### AOU 
    if AOU: 
       ### load PO4 and O2 in (mol/m3)
       ### convert mol/m3 to umol/kg 
       PO4  = dicdiag.TRAC03.isel(T=indT) 
       O2   = dicdiag.TRAC05.isel(T=indT)
       ### Compute saturated O2
       O2sat = dg.satO2(dirF, indT)
       ### Compute AOU 
       AOU = O2sat - O2
       ### Compute preformed alkalinity with AOU estimate
       ### convert mol/m3 to umol/kg 
       a1,a2,a3 = dg.MRL_alk(dirF, indT)
       pAT  = permumolkg * (a1 + a2 * S + a3 * (O2 + 170 * PO4))
       ### compute preformed PO4 in (mol/m3)
       pPO4 = permumolkg * (PO4 - (1/170) * AOU)
       ### compute preformed silicate in (mol/m3)
       ### convert mol/m3 to umol/kg 
       pSi  = permumolkg * dg.get_pSi_AOU(dirF, indT)
    ### preformed tracers 
    else: 
       ### load preformed Alkalinity and preformed PO4 in (mol/m3)
       ### convert mol/m3 to umol/kg 
       pAT   = permumolkg * dicdiag.TRAC07.isel(T=indT) 
       pPO4  = permumolkg * dicdiag.TRAC08.isel(T=indT) 
       ### compute preformed silicate in (mol/m3)
       ### convert mol/m3 to umol/kg 
       pSi  = permumolkg * dg.get_pSi(dirF, indT)
    ### compute pressure in dbar
    Pdbar = dg.getPdbar(dirF,0) + 10.1325 #absolute pressure
    ### extract pco2 and convert from atm to uatm 
    if pco2 is None:
       pCO2 = 1e6 * surfdiag.DICATCO2.isel(T=indT,YC=0,XC=0).values
    else:
       pCO2 = 1e6 * pco2
    ### compute saturated DIC (umol/kg)
    #co = co2sys.CarbonateSystem(temp=PT, salt=S, pres=Pdbar, temp_out=PT, pres_out=Pdbar, TA=preAlk, pCO2=pCO2, PO4=prePO4, Si=presi)
    #TC = hfacc * co.TC
    Csat = grid.HFacC * dg.calc_carbon(pCO2, pAT, pPO4, pSi, T, S, Pdbar)/permumolkg
 
    return Csat

def get_Csat_Part(
    dir_ctrl: str,
    dir_exp: str,
    dirT: str,
    dirS: str,
    indTctrl: int | None = -1,
    indT: int | None = None,
    indTT: int | None = None,
    indTS: int | None = None,
    pco2_ctrl: float | None = None,
    pco2: float | None = None,
    grid_file: str = "grid.nc",
    oce_file: str = "oceDiag.nc",
    dic_file: str = "dicDiag.nc",
    surf_file: str = "surfDiag.nc",
    dir_oce_ctrl: str | None = None,
    ):
    """
    Compute saturated DIC decomposition (mol/m3) with respect to a control experiment.

    Works with:
      - time-series files containing dimension "T"
      - snapshot files where time dimension has been removed

    Parameters
    ----------
    dir_ctrl : str
        Directory for control experiment (must contain grid + diagnostics).
    dir_exp : str
        Directory for experiment.
    dirT : str
        Directory used to extract potential temperature (THETA).
    dirS : str
        Directory used to extract salinity (SALT).
    indTctrl : int or None
        Time index if 'T' dimension exists. Ignored if no 'T'.
    indT : int or None
        Time index for experiment (used only if T exists). 
    indTT : int or None
        Time index for temperature (used only if T exists).
    indTS : int or None
        Time index for salinity (used only if T exists).
    pco2_ctrl : float or None
        pCO2 control in atm (if provided, overrides file value).
    pco2 : float or None
        pCO2 experiment in atm (if provided, overrides file value).

    grid_file, oce_file, dic_file, surf_file : str
        Filenames for Dryad-style files (no mnc_glob). Change if needed.

    dir_oce_ctrl : str or None
        OPTIONAL override directory used ONLY to open oceDiag for the CONTROL state.
        This is useful for Dryad exports where oceDiag exists only in
        "Physics_and_Biogeochemistry" (PB) and not in "Physics".
        If None, defaults to dir_ctrl (legacy behavior).

    Returns
    -------
    Csat_ctrl, Csat, Csat_278, Csat_pCO2, Csat_pAT, Csat_T, Csat_S
    """

    # -------------------------
    # Units conversion
    # -------------------------
    permil = 1 / 1024.5          # mol/m3 -> mol/kg
    permumolkg = permil * 1e6    # mol/m3 -> umol/kg

    # Defaults: if not given, use indTctrl everywhere (when T exists)
    if indT is None:
        indT = indTctrl
    if indTT is None:
        indTT = indTctrl
    if indTS is None:
        indTS = indTctrl

    # Where to read CONTROL oceDiag from (only needed for Dryad "Physics-only" layout)
    if dir_oce_ctrl is None:
        dir_oce_ctrl = dir_ctrl

    # -------------------------
    # Load grid
    # -------------------------
    grid_path = resolve_nc(dir_ctrl, grid_file, "grid.glob.nc")
    grid, xgrid = mitgcm_tools.loadgrid(grid_path, basin_masks=False)

    # -------------------------
    # Load diagnostics
    # -------------------------
    oce_path_ctrl = resolve_nc(dir_oce_ctrl, oce_file, "oceDiag.glob.nc")
    ocediag_ctrl = dg.open_nc(
        oce_path_ctrl,
        strange_axes={"Zmd000029": "ZC", "Zld000029": "ZL"},
        grid=grid
    )

    T_path = resolve_nc(dirT, oce_file, "oceDiag.glob.nc")
    ocediagT = dg.open_nc(
        T_path,
        strange_axes={"Zmd000029": "ZC", "Zld000029": "ZL"},
        grid=grid
    )

    S_path = resolve_nc(dirS, oce_file, "oceDiag.glob.nc")
    ocediagS = dg.open_nc(
        S_path,
        strange_axes={"Zmd000029": "ZC", "Zld000029": "ZL"},
        grid=grid
    )

    dic_path_ctrl = resolve_nc(dir_ctrl, dic_file, "dicDiag.glob.nc")
    dicdiag_ctrl = dg.open_nc(
        dic_path_ctrl,
        strange_axes={"Zmd000029": "ZC"},
        grid=grid
    )

    dic_path_exp = resolve_nc(dir_exp, dic_file, "dicDiag.glob.nc")
    dicdiag_exp = dg.open_nc(
        dic_path_exp,
        strange_axes={"Zmd000029": "ZC"},
        grid=grid
    )

    # Fix YC coordinate if needed
    dicdiag_ctrl.coords["YC"] = grid.YC
    dicdiag_exp.coords["YC"] = grid.YC

    surf_path_ctrl = resolve_nc(dir_ctrl, surf_file, "surfDiag.glob.nc")
    surfdiag_ctrl = dg.open_nc(
        surf_path_ctrl,
        strange_axes={"Zmd000001": "ZC", "Zd000001": "ZL"},
        grid=grid
    )

    surf_path_exp = resolve_nc(dir_exp, surf_file, "surfDiag.glob.nc")
    surfdiag_exp = dg.open_nc(
        surf_path_exp,
        strange_axes={"Zmd000001": "ZC", "Zd000001": "ZL"},
        grid=grid
    )

    # -------------------------
    # Extract S and T (handles both time-series and snapshots)
    # -------------------------
    Sctrl = pick_time(ocediag_ctrl.SALT, indTctrl)
    Tctrl = pick_time(ocediag_ctrl.THETA, indTctrl)

    S = pick_time(ocediagS.SALT, indTS)
    T = pick_time(ocediagT.THETA, indTT)

    # -------------------------
    # Preformed Alk and PO4 (mol/m3 -> umol/kg)
    # -------------------------
    pATctrl  = permumolkg * pick_time(dicdiag_ctrl.TRAC07, indTctrl)
    pPO4ctrl = permumolkg * pick_time(dicdiag_ctrl.TRAC08, indTctrl)

    pAT  = permumolkg * pick_time(dicdiag_exp.TRAC07, indT)
    pPO4 = permumolkg * pick_time(dicdiag_exp.TRAC08, indT)

    # -------------------------
    # Pressure (dbar)
    # -------------------------
    Pdbar_ctrl = dg.getPdbar(dir_ctrl, 0) + 10.1325
    Pdbar      = dg.getPdbar(dir_exp,  0) + 10.1325

    # -------------------------
    # pCO2 (uatm)
    # If no time axis, pick_time is a no-op.
    # -------------------------
    if pco2_ctrl is None:
        val = pick_time(surfdiag_ctrl.DICATCO2, indTctrl).isel(YC=0, XC=0).values
        pCO2_ctrl = 1e6 * val
    else:
        pCO2_ctrl = 1e6 * pco2_ctrl

    if pco2 is None:
        val = pick_time(surfdiag_exp.DICATCO2, indT).isel(YC=0, XC=0).values
        pCO2 = 1e6 * val
    else:
        pCO2 = 1e6 * pco2

    # -------------------------
    # Preformed silicate (mol/m3 -> umol/kg)
    # -------------------------
    pSi_ctrl = permumolkg * dg.get_pSi(dir_ctrl)
    pSi      = permumolkg * dg.get_pSi(dir_exp)

    # -------------------------
    # Saturation DIC (umol/kg) -> convert back to mol/m3 using /permumolkg
    # -------------------------
    Csat_ctrl = grid.HFacC * dg.calc_carbon(pCO2_ctrl, pATctrl, pPO4ctrl, pSi_ctrl, Tctrl, Sctrl, Pdbar_ctrl) / permumolkg
    Csat      = grid.HFacC * dg.calc_carbon(pCO2,      pAT,     pPO4,     pSi,      T,     S,     Pdbar)      / permumolkg

    Csat_278  = grid.HFacC * dg.calc_carbon(pCO2_ctrl, pAT,     pPO4,     pSi,      T,     S,     Pdbar)      / permumolkg
    Csat_pCO2 = grid.HFacC * dg.calc_carbon(pCO2,      pATctrl, pPO4ctrl, pSi_ctrl, Tctrl, Sctrl, Pdbar_ctrl) / permumolkg
    Csat_pAT  = grid.HFacC * dg.calc_carbon(pCO2_ctrl, pAT,     pPO4ctrl, pSi_ctrl, Tctrl, Sctrl, Pdbar_ctrl) / permumolkg
    Csat_T    = grid.HFacC * dg.calc_carbon(pCO2_ctrl, pATctrl, pPO4ctrl, pSi_ctrl, T,     Sctrl, Pdbar_ctrl) / permumolkg
    Csat_S    = grid.HFacC * dg.calc_carbon(pCO2_ctrl, pATctrl, pPO4ctrl, pSi_ctrl, Tctrl, S,     Pdbar_ctrl) / permumolkg

    return Csat_ctrl, Csat, Csat_278, Csat_pCO2, Csat_pAT, Csat_T, Csat_S

def get_Csat_Part_AOU(dir_ctrl, dir_exp, dirT, dirS, indTctrl, indT=None, indTT=None, indTS=None):
    """Compute the saturated DIC decomposition (mol/m3)
    with respect to control experiment using AOU 
 
    :param:
    .....dir_ctrl: directory for control experiment 
    ......dir_exp: directory for experiment 
    .......dir_PT: directory for extracting potential temperature 
    .......dir_S: directory for extracting salinity 
    ....indT_ctrl: time index for control experiment 
    .........indT: time index for experiment 
    ......indT_PT: time index for potential temperature 
    ......indT_S: time index for salinity 

    :return:
    Csat (mol/m3): saturated DIC (shape (Z,Y,X)) 
    """

    # Change units from the input of mol/m^3 -> mol/kg:
    # (1 mol/m^3) x (1 /1024.5 kg/m^3)
    permil=1/1024.5
    # convert mol/m3 to umol/kg
    permumolkg=permil*1e6

    # Assign defaults if not provided
    if indT is None:
        indT  = indTctrl
    if indTT is None:
        indTT = indTctrl
    if indTS is None:
        indTS = indTctrl

    ### load grid
    grid, xgrid   = mitgcm_tools.loadgrid(dir_ctrl + 'grid.glob.nc', basin_masks=False)
    grid.close()
    ### load arrays
    ocediag_ctrl  = mitgcm_tools.open_ncfile(dir_ctrl + 'oceDiag.glob.nc',\
      strange_axes={'Zmd000029':'ZC','Zld000029':'ZL'},grid=grid)
    ocediag_ctrl.close()
    ocediagT      = mitgcm_tools.open_ncfile(dirT + 'oceDiag.glob.nc',\
      strange_axes={'Zmd000029':'ZC','Zld000029':'ZL'},grid=grid)
    ocediagT.close()
    ocediagS      = mitgcm_tools.open_ncfile(dirS + 'oceDiag.glob.nc',\
      strange_axes={'Zmd000029':'ZC','Zld000029':'ZL'},grid=grid)
    ocediagS.close()
    dicdiag_ctrl  = mitgcm_tools.open_ncfile(dir_ctrl + 'dicDiag.glob.nc',\
      strange_axes={'Zmd000029':'ZC'},grid=grid)
    dicdiag_ctrl.close()
    dicdiag_exp   = mitgcm_tools.open_ncfile(dir_exp + 'dicDiag.glob.nc',\
      strange_axes={'Zmd000029':'ZC'},grid=grid)
    dicdiag_exp.close()
    # problem with YC values in dicdiag
    dicdiag_exp.coords['YC']=grid.YC
    surfdiag_ctrl = mitgcm_tools.open_ncfile(dir_ctrl + 'surfDiag.glob.nc',\
      strange_axes={'Zmd000001':'ZC','Zd000001':'ZL'},grid=grid)
    surfdiag_ctrl.close()
    surfdiag_exp  = mitgcm_tools.open_ncfile(dir_exp + 'surfDiag.glob.nc',\
      strange_axes={'Zmd000001':'ZC','Zd000001':'ZL'},grid=grid)
    surfdiag_exp.close()
    ### load salinity and theta
    Sctrl         = ocediag_ctrl.SALT.isel(T=indTctrl) 
    Tctrl         = ocediag_ctrl.THETA.isel(T=indTctrl) 
    S             = ocediagS.SALT.isel(T=indTS) 
    T             = ocediagT.THETA.isel(T=indTT) 
    ### load PO4 and O2 
    ### load Alkalinity and PO4 in (mol/m3)
    PO4ctrl       = dicdiag_ctrl.TRAC03.isel(T=indT) 
    O2ctrl        = dicdiag_ctrl.TRAC05.isel(T=indT) 
    PO4           = dicdiag_exp.TRAC03.isel(T=indT) 
    O2            = dicdiag_exp.TRAC05.isel(T=indT) 
    # Compute saturated O2 
    # I should update dg.satO2 to include variations in indT for T and S
    O2satctrl     = dg.satO2(dir_ctrl, indT)
    O2sat         = dg.satO2(dirS, indT, dirT)
    ### Compute AOU 
    AOUctrl       = O2satctrl - O2ctrl
    AOU           = O2sat - O2
    ### compute preformed PO4 in (mol/m3)
    pPO4ctrl = permumolkg * (PO4ctrl - (1/170) * AOUctrl)
    pPO4     = permumolkg * (PO4 - (1/170) * AOU)
    ### Compute preformed alkalinity with AOU estimate
    ### convert mol/m3 to umol/kg 
    a1,a2,a3 = dg.MRL_alk(dir_ctrl, indT)
    pATctrl  = permumolkg * (a1 + a2 * Sctrl + a3 * (O2ctrl + 170 * PO4ctrl))
    a4,a5,a6 = dg.MRL_alk(dir_exp, indT, dirS)
    pAT      = permumolkg * (a4 + a5 * S + a6 * (O2 + 170 * PO4))
    ### compute absolute pressure in dbar
    Pdbar_ctrl    = dg.getPdbar(dir_ctrl,0) + 10.1325 
    Pdbar         = dg.getPdbar(dir_exp,0) + 10.1325 
    #Pdbar_ctrl    = dg.getPdbar(dir_ctrl,1) 
    #Pdbar         = dg.getPdbar(dir_exp,1) 
    ### extract pco2 and convert from atm to uatm 
    pCO2_ctrl     = 1e6 * surfdiag_ctrl.DICATCO2.isel(T=indTctrl,YC=0,XC=0).values 
    pCO2          = 1e6 * surfdiag_exp.DICATCO2.isel(T=indT,YC=0,XC=0).values 
    ### compute preformed silicate in (mol/m3)
    ### convert mol/m3 to umol/kg 
    pSi_ctrl      = permumolkg * dg.get_pSi_AOU(dir_ctrl, indTctrl)
    pSi           = permumolkg * dg.get_pSi_AOU(dir_exp, indT, dirS, dirT)
    ### compute saturated DIC (umol/kg)
    Csat_ctrl     = grid.HFacC * dg.calc_carbon(pCO2_ctrl, pATctrl, pPO4ctrl, pSi_ctrl, Tctrl, Sctrl, Pdbar_ctrl) /permumolkg
    Csat          = grid.HFacC * dg.calc_carbon(pCO2,      pAT,     pPO4,     pSi,      T,     S,     Pdbar)      /permumolkg
    Csat_278      = grid.HFacC * dg.calc_carbon(pCO2_ctrl, pAT,     pPO4,     pSi,      T,     S,     Pdbar)      /permumolkg
    Csat_pCO2     = grid.HFacC * dg.calc_carbon(pCO2,      pATctrl, pPO4ctrl, pSi_ctrl, Tctrl, Sctrl, Pdbar_ctrl) /permumolkg
    Csat_pAT      = grid.HFacC * dg.calc_carbon(pCO2_ctrl, pAT,     pPO4ctrl, pSi_ctrl, Tctrl, Sctrl, Pdbar_ctrl) /permumolkg
    Csat_T        = grid.HFacC * dg.calc_carbon(pCO2_ctrl, pATctrl, pPO4ctrl, pSi_ctrl, T,     Sctrl, Pdbar_ctrl) /permumolkg
    Csat_S        = grid.HFacC * dg.calc_carbon(pCO2_ctrl, pATctrl, pPO4ctrl, pSi_ctrl, Tctrl, S,     Pdbar_ctrl) /permumolkg

    return Csat_ctrl, Csat, Csat_278, Csat_pCO2, Csat_pAT, Csat_T, Csat_S

def get_Csoft(
    dirF: str,
    indT: int | None = -1,
    Rcp: float | None = None,
    *,
    grid_file: str = "grid.nc",
    dic_file: str = "dicDiag.nc",
):
    """
    Compute the soft-tissue carbon contribution Csoft (mol/m3).

    Compatible with:
      - standard MNC outputs (grid.glob.nc, dicDiag.glob.nc, with T + diag_levels)
      - Dryad-style snapshots (grid.nc, dicDiag.nc, without diag_levels and T)

    Parameters
    ----------
    dirF : str
        Directory containing the grid and dic diagnostics.
    indT : int or None
        Time index used only if dimension 'T' exists (ignored otherwise).
    Rcp : float or None
        C:P ratio. Default is 117.
    grid_file : str
        Preferred grid filename (default "grid.nc"). Falls back to "grid.glob.nc".
    dic_file : str
        Preferred dicDiag filename (default "dicDiag.nc"). Falls back to "dicDiag.glob.nc".

    Returns
    -------
    Csoft : xarray.DataArray
        Soft-tissue contribution (mol/m3), typically on (ZC, YC, XC).
    """
    if Rcp is None:
        Rcp = 117.0

    # Resolve filenames (Dryad vs legacy)
    grid_path = dg.resolve_nc(dirF, grid_file, "grid.glob.nc")
    dic_path  = dg.resolve_nc(dirF, dic_file,  "dicDiag.glob.nc")

    # Load grid
    grid, xgrid = mitgcm_tools.loadgrid(grid_path, basin_masks=False)
    grid.close()

    # Load dic diagnostics (works for both formats via dg.open_nc)
    dicdiag = dg.open_nc(dic_path, strange_axes={"Zmd000029": "ZC"}, grid=grid)
    dicdiag.close()

    # Make sure YC coordinate matches grid
    dicdiag = dicdiag.assign_coords(YC=grid.YC)

    # Extract PO4 and preformed PO4 (mol/m3)
    PO4  = dg.pick_time(dicdiag.TRAC03, indT)
    pPO4 = dg.pick_time(dicdiag.TRAC08, indT)

    # Csoft = HFacC * Rcp * (PO4 - pPO4)
    Csoft = grid.HFacC * (Rcp * (PO4 - pPO4))

    return Csoft

def get_Csoft_AOU(dirF, indT, dirS=None, dirT=None, Rco=None):
    """Compute the soft-tissue contribution Csoft (mol/m3) with AOU 
 
    :param:
    ..........dirF: work directory 
    ..........indT: time index 
    ..........dirS: work directory for salinity
    ..........dirT: work directory for temperature

    :return:
    Csoft (mol/m3): soft-tissue contribution (shape (Z,Y,X)) 
    """

    # Assign defaults if not provided
    if dirS is None:
        dirS  = dirF
    # Assign defaults if not provided
    if dirT is None:
        dirT  = dirF
    # Assign defaults if not provided
    if Rco is None:
        Rco  = -117/170

    # load grid
    grid, xgrid = mitgcm_tools.loadgrid(dirF + 'grid.glob.nc', basin_masks=False)
    grid.close()
    # load DIC file
    dicdiag     = mitgcm_tools.open_ncfile(dirF + 'dicDiag.glob.nc',\
           strange_axes={'Zmd000029':'ZC'},grid=grid)
    dicdiag.close()
    # problem with YC values in dicdiag
    dicdiag.coords['YC'] = grid.YC
    # load O2 (mol/m3)
    O2   = dicdiag.TRAC05.isel(T=indT) 
    # Compute saturated O2
    O2sat = dg.satO2(dirS, indT, dirT)
    ### Compute AOU 
    AOU   = O2sat - O2
    ### Compute Csoft 
    ### Rco = -106/170 in Williams and Follows 2011
    ### Rco = -117/170 in MITgcm 
    Csoft = grid['HFacC'] * (-Rco * AOU)
 
    return Csoft

def get_Ccarb(
    dirF: str,
    indT: int | None = -1,
    *,
    grid_file: str = "grid.nc",
    dic_file: str = "dicDiag.nc",
):
    """
    Compute the carbonate contribution Ccarb (mol/m3).

    Compatible with:
      - standard MNC outputs (grid.glob.nc, dicDiag.glob.nc, with T + diag_levels)
      - Dryad-style snapshots (grid.nc, dicDiag.nc, without diag_levels and possibly without T)

    Parameters
    ----------
    dirF : str
        Directory containing the grid and dic diagnostics.
    indT : int or None
        Time index used only if dimension 'T' exists (ignored otherwise).
    grid_file : str
        Preferred grid filename (default "grid.nc"). Falls back to "grid.glob.nc".
    dic_file : str
        Preferred dicDiag filename (default "dicDiag.nc"). Falls back to "dicDiag.glob.nc".

    Returns
    -------
    Ccarb : xarray.DataArray
        Carbonate contribution (mol/m3), typically on (ZC, YC, XC).
    """
    # Resolve filenames (Dryad vs legacy)
    grid_path = dg.resolve_nc(dirF, grid_file, "grid.glob.nc")
    dic_path  = dg.resolve_nc(dirF, dic_file,  "dicDiag.glob.nc")

    # Load grid
    grid, xgrid = mitgcm_tools.loadgrid(grid_path, basin_masks=False)
    grid.close()

    # Load dic diagnostics (works for both formats via dg.open_nc)
    dicdiag = dg.open_nc(dic_path, strange_axes={"Zmd000029": "ZC"}, grid=grid)
    dicdiag.close()

    # Make sure YC coordinate matches grid
    dicdiag = dicdiag.assign_coords(YC=grid.YC)

    # Load tracers (mol/m3)
    Alk  = dg.pick_time(dicdiag.TRAC02, indT)
    pAlk = dg.pick_time(dicdiag.TRAC07, indT)
    PO4  = dg.pick_time(dicdiag.TRAC03, indT)
    pPO4 = dg.pick_time(dicdiag.TRAC08, indT)

    # Carbonate contribution:
    # Ccarb = 0.5 * HFacC * (Alk - pAlk + 16 * (PO4 - pPO4))
    Ccarb = 0.5 * grid.HFacC * (Alk - pAlk + 16.0 * (PO4 - pPO4))

    return Ccarb
 
def get_Ccarb_AOU(dirF, indT, dirS=None, dirT=None):
    """Compute the carbonate contribution Ccarb (mol/m3) with AOU 
 
    :param:
    ..........dirF: work directory 
    ..........indT: time index 
    ..........dirS: work directory for salinity
    ..........dirT: work directory for temperature

    :return:
    Ccarb (mol/m3): carbonate contribution (shape (Z,Y,X)) 
    """

    # Assign defaults if not provided
    if dirS is None:
        dirS  = dirF
    # Assign defaults if not provided
    if dirT is None:
        dirT  = dirF

    # load grid
    grid, xgrid = mitgcm_tools.loadgrid(dirF + 'grid.glob.nc', basin_masks=False)
    grid.close()
    # load salinity file
    ocediag  = mitgcm_tools.open_ncfile(dirS + 'oceDiag.glob.nc',\
      strange_axes={'Zmd000029':'ZC','Zld000029':'ZL'},grid=grid)
    ocediag.close()
    # load salinity 
    S        = ocediag.SALT.isel(T=indT) 
    # load DIC file
    dicdiag  = mitgcm_tools.open_ncfile(dirF + 'dicDiag.glob.nc',\
           strange_axes={'Zmd000029':'ZC'},grid=grid)
    dicdiag.close()
    # problem with YC values in dicdiag
    dicdiag.coords['YC'] = grid.YC
    ### load Alk, PO4 and O2 (mol/m3)
    Alk      = dicdiag.TRAC02.isel(T=indT) 
    PO4      = dicdiag.TRAC03.isel(T=indT) 
    O2       = dicdiag.TRAC05.isel(T=indT) 
    ### Compute saturated O2
    O2sat    = dg.satO2(dirS, indT, dirT)
    ### Compute AOU 
    AOU      = O2sat - O2
    ### Compute preformed alkalinity with AOU estimate
    a1,a2,a3 = dg.MRL_alk(dirF, indT, dirS)
    pAlk_aou = grid['HFacC'] * (a1 + a2 * S + a3 * (O2 + 170 * PO4))
    ### Compute Ccarb
    Ccarb    = (1/2) * grid['HFacC'] * (Alk - pAlk_aou + (16/170) * AOU)
 
    return Ccarb 

def _calc_co2sys_tc(s,t,pz,at,atmpco2,pt,sit):
    import co2sys
    
    co=co2sys.calc_co2_system(s, t, 
                              pres    = pz, 
                              TA      = at,
                              pCO2    = atmpco2,
                              PO4     = pt,
                              Si      = sit,
                              K1K2    = "Millero_1995", 
                              KBver   = "Uppstrom", 
                              KSver   = "Dickson",
                              KFver   = "Dickson",
                              pHScale = 1
                              )
    return co.TC

def _calc_co2sys_pco2(s,t,pz,at,tc,pt,sit):
    import co2sys
    
    co=co2sys.calc_co2_system(s, t, 
                              pres    = pz, 
                              TA      = at,
                              TC      = tc,
                              PO4     = pt,
                              Si      = sit,
                              K1K2    = "Millero_1995", 
                              KBver   = "Uppstrom", 
                              KSver   = "Dickson",
                              KFver   = "Dickson",
                              pHScale = 1
                              )
    return co.pCO2

def calc_pco2(dic,alk,po4,sit,theta,salt,pressure):
    """
    calc_pco2(dic,alk,po4,sit,theta,salt,pressure)
    
    Compute seawater pCO2 concentration from dic and alkalinity
    at local pressure, temperature, salinity and nutrient conc.
    """
    import xarray as xr
    pco2 = xr.apply_ufunc(_calc_co2sys_pco2,
                            salt,
                            theta,
                            pressure,
                            alk,
                            dic,
                            po4,
                            sit,
                            dask='parallelized', 
                            output_dtypes=[float],
                            )
    return pco2

def calc_carbon(pco2,alk,po4,sit,theta,salt,pressure):
    """
    calc_carbon(pco2,alk,po4,sit,theta,salt,pressure)
    
    Compute DIC concentration from seawater pCO2 and alkalinity
    at local pressure, temperature, salinity and nutrient conc.
    
    For saturated carbon, use preformed alkalinity and nutrients, as
    well at a 3d field of atmospheric pCO2.
    """
    import xarray as xr
    dic = xr.apply_ufunc(_calc_co2sys_tc,
                            salt,
                            theta,
                            pressure,
                            alk,
                            pco2,
                            po4,
                            sit,
                            dask='parallelized', 
                            output_dtypes=[float],
                            )
    return dic

