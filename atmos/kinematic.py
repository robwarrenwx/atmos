import numpy as np
import warnings
from atmos.utils import (interpolate_vector_to_height_level,
                         height_layer_mean_vector)


def wind_speed(u, v):
    """
    Computes wind speed from wind components.

    Args:
        u (float or ndarray): eastward component of wind (m/s)
        v (float or ndarray): northward component of wind (m/s)

    Returns:
        wspd (float or ndarray): wind speed (m/s)

    """
    wspd = np.hypot(u, v)
    
    return wspd


def wind_direction(u, v):
    """
    Computes wind direction from wind components.

    Args:
        u (float or ndarray): eastward component of wind (m/s)
        v (float or ndarray): northward component of wind (m/s)

    Returns:
        wdir (float or ndarray): wind from direction measured clockwise from
            north (degrees)

    """
    wdir = 90. - np.degrees(np.arctan2(-v, -u))
    
    u = np.atleast_1d(u)
    v = np.atleast_1d(v)
    wdir = np.atleast_1d(wdir)
    
    # Correct negative values
    neg_mask = np.array(wdir < 0.)
    if np.any(neg_mask):
        wdir[neg_mask] += 360.

    # Set to 0 where wind speed is zero
    calm_mask = (u == 0.) & (v == 0.)
    if np.any(calm_mask):
        wdir[calm_mask] = 0.
        
    wdir = wdir.squeeze()

    return wdir
    

def wind_components(wspd, wdir):
    """
    Computes wind components from wind speed and direction.

    Args:
        wspd (float or ndarray): wind speed (m/s)
        wdir (float or ndarray): wind from direction measured clockwise from
            north (degrees)

    Returns:
        u (float or ndarray): eastward component of wind (m/s)
        v (float or ndarray): northward component of wind (m/s)

    """
    u = -wspd * np.sin(np.radians(wdir))
    v = -wspd * np.cos(np.radians(wdir))
    
    return u, v


def bulk_wind_difference(z, u, v, z_bot, z_top, z_sfc=None, u_sfc=None,
                         v_sfc=None, vertical_axis=0):
    """
    Computes bulk wind difference across a specified layer.

    Args:
        z (float or ndarray): height (m)
        u (float or ndarray): eastward component of wind (m/s)
        v (float or ndarray): northward component of wind (m/s)
        z_bot (float or ndarray): height of bottom of layer (m)
        z_top (float or ndarray): height of top of layer (m)
        z_sfc (float or ndarray, optional): surface height (m)
        u_sfc (float or ndarray, optional): eastward component of wind at
            surface (m/s)
        v_sfc (float or ndarray, optional): northward component of wind at
            surface (m/s)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)

    Returns:
        BWDu (float or ndarray): eastward component of bulk wind difference
            (m/s)
        BWDv (float or ndarray): northward component of bulk wind difference
            (m/s)

    """

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        z = np.moveaxis(z, vertical_axis, 0)
        u = np.moveaxis(u, vertical_axis, 0)
        v = np.moveaxis(v, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if z.ndim == 1:
        z = np.atleast_2d(z).T  # transpose to preserve vertical axis
        u = np.atleast_2d(u).T
        v = np.atleast_2d(v).T

    # If surface fields are not provided, use lowest level
    if u_sfc is None:
        bottom = 'lowest level'
        k_start = 1  # start loop from second level
        z_sfc = z[0]
        u_sfc = u[0]
        v_sfc = v[0]
    else:
        bottom = 'surface'
        k_start = 0  # start loop from first level
        if z_sfc is None:
            z_sfc = np.zeros_like(u_sfc)  # assumes height AGL

    # Make sure that surface fields are at least 1D
    z_sfc = np.atleast_1d(z_sfc)
    u_sfc = np.atleast_1d(u_sfc)
    v_sfc = np.atleast_1d(v_sfc)

    # Make sure that z_bot and z_top match dimensions of surface arrays
    if np.isscalar(z_bot):
        z_bot = np.full_like(z_sfc, z_bot)
    if np.isscalar(z_top):
        z_top = np.full_like(z_sfc, z_top)

    # Check if bottom of layer is above top of layer
    if np.any(z_bot > z_top):
        n_pts = np.count_nonzero(z_bot > z_top)
        raise ValueError(f'z_bot is above z_top for {n_pts} points')

    # Check if bottom of layer is below surface
    if np.any(z_bot < z_sfc):
        n_pts = np.count_nonzero(z_bot < z_sfc)
        warnings.warn(f'z_bot is below {bottom} for {n_pts} points')

    # Check if top of layer is above highest level
    if np.any(z_top > z[-1]):
        n_pts = np.count_nonzero(z_top > z[-1])
        warnings.warn(f'z_top is above highest level for {n_pts} points')

    # Note the number of vertical levels
    n_lev = z.shape[0]

    # Initialise winds at bottom and top of layer
    u_bot = u_sfc.copy()
    v_bot = v_sfc.copy()
    u_top = np.atleast_1d(u[-1])
    v_top = np.atleast_1d(v[-1])

    # Initialise level 2 fields
    z2 = z_sfc.copy()
    u2 = u_sfc.copy()
    v2 = v_sfc.copy()

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 fields
        z1 = z2.copy()
        u1 = u2.copy()
        v1 = v2.copy()
        if np.all(z1 >= z_top):
            # can break out of loop
            break

        # Update level 2 fields
        above_sfc = (z[k] > z_sfc)
        z2 = np.where(above_sfc, z[k], z_sfc)
        u2 = np.where(above_sfc, u[k], u_sfc)
        v2 = np.where(above_sfc, v[k], v_sfc)
        if np.all(z2 <= z_bot):
            # can skip this level
            continue

        # Set winds at bottom of layer
        cross_bot = (z1 <= z_bot) & (z2 > z_bot)
        if np.any(cross_bot):
            weight = (z_bot[cross_bot] - z1[cross_bot]) / \
                (z2[cross_bot] - z1[cross_bot])
            u_bot[cross_bot] = (1 - weight) * u1[cross_bot] + \
                weight * u2[cross_bot]
            v_bot[cross_bot] = (1 - weight) * v1[cross_bot] + \
                weight * v2[cross_bot]

        # Set winds at top of layer
        cross_top = (z1 < z_top) & (z2 >= z_top)
        if np.any(cross_top):
            weight = (z_top[cross_top] - z1[cross_top]) / \
                (z2[cross_top] - z1[cross_top])
            u_top[cross_top] = (1 - weight) * u1[cross_top] + \
                weight * u2[cross_top]
            v_top[cross_top] = (1 - weight) * v1[cross_top] + \
                weight * v2[cross_top]

    # Compute the bulk wind difference components
    BWDu = u_top - u_bot
    BWDv = v_top - v_bot

    # Deal with points where z_bot or z_top are undefined
    is_nan = np.isnan(z_bot) | np.isnan(z_top)
    if np.any(is_nan):
        BWDu[is_nan] = np.nan
        BWDv[is_nan] = np.nan

    return BWDu.squeeze(), BWDv.squeeze()


def bunkers_storm_motion(z, u, v, z_sfc=None, u_sfc=None, v_sfc=None,
                         vertical_axis=0,
                         level_weights=None, surface_weight=None,
                         mean_layer_base=0.0, mean_layer_top=6000.0,
                         shear_layer_base=0.0, shear_layer_top=6000.0,
                         shear_layer_base_average=500.0,
                         shear_layer_top_average=500.0,
                         deviation_left=7.5, deviation_right=7.5):
    """
    Computes Bunkers left and right storm motion vectors.

    Args:
        z (ndarray): height (m)
        u (ndarray): eastward component of wind (m/s)
        v (ndarray): northward component of wind (m/s)
        z_sfc (float or ndarray, optional): surface height (m)
        u_sfc (float or ndarray, optional): eastward component of wind at
            surface (m/s)
        v_sfc (float or ndarray, optional): northward component of wind at
            surface (m/s)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)
        level_weights (ndarray, optional): weights to apply to winds at each
            level in mean wind calculations (default is None, in which case no
            weighting is applied)
        surface_weight (float or ndarray, optional): weight to apply to vector
            at surface (default is None, in which case no weighting is applied)
        mean_layer_base (float or ndarray, optional): height of base of mean
            wind layer (m) (default is 0.0)
        mean_layer_top (float or ndarray, optional): height of top of mean
            wind layer (m) (default is 6000.0)
        shear_layer_base (float or ndarray, optional): height of base of shear
            layer (m) (default is 0.0)
        shear_layer_top (float or ndarray, optional): height of top of shear
            layer (m) (default is 6000.0)
        shear_layer_base_average (float, optional): depth over which to average
            winds at base of shear layer (m) (default is 500.0)
        shear_layer_top_average (float, optional): depth over which to average
            winds at top of shear layer (m) (default is 500.0)
        deviation_left (float, optional): magnitude of deviation for left
            mover (m/s) (default is 7.5)
        deviation_right (float, optional): magnitude of deviation for right
            mover (m/s) (default is 7.5)

    Returns:
        u_bl (float or ndarray): eastward component of Bunkers left storm
            motion (m/s)
        v_bl (float or ndarray): northward component of Bunkers left storm
            motion (m/s)
        u_br (float or ndarray): eastward component of Bunkers right storm
            motion (m/s)
        v_br (float or ndarray): northward component of Bunkers right storm
            motion (m/s)

    """

    # Compute advective component of storm motion
    z_bot = mean_layer_base
    z_top = mean_layer_top
    u_adv, v_adv = height_layer_mean_vector(
        z, u, v, z_bot, z_top, z_sfc=z_sfc, u_sfc=u_sfc, v_sfc=v_sfc,
        level_weights=level_weights, surface_weight=surface_weight,
        vertical_axis=vertical_axis
    )

    # Compute winds at bottom of shear layer
    z1 = shear_layer_base
    if shear_layer_base_average == 0.0:
        u1, v1 = interpolate_vector_to_height_level(
            z, u, v, z1, z_sfc=z_sfc, u_sfc=u_sfc, v_sfc=v_sfc,
            vertical_axis=vertical_axis
        )
    else:
        dz = shear_layer_base_average
        u1, v1 = height_layer_mean_vector(
            z, u, v, z1, z1+dz, z_sfc=z_sfc, u_sfc=u_sfc, v_sfc=v_sfc,
            level_weights=level_weights, surface_weight=surface_weight,
            vertical_axis=vertical_axis
        )

    # Compute winds at top of shear layer
    z2 = shear_layer_top
    if shear_layer_top_average == 0.0:
        u2, v2 = interpolate_vector_to_height_level(
            z, u, v, z2, z_sfc=z_sfc, u_sfc=u_sfc, v_sfc=v_sfc,
            vertical_axis=vertical_axis
        )
    else:
        dz = shear_layer_top_average
        u2, v2 = height_layer_mean_vector(
            z, u, v, z2-dz, z2, z_sfc=z_sfc, u_sfc=u_sfc, v_sfc=v_sfc,
            level_weights=level_weights, surface_weight=surface_weight,
            vertical_axis=vertical_axis
        )

    # Compute shear vector components
    u_shr = u2 - u1
    v_shr = v2 - v1

    # Compute shear magnitude
    shr = np.hypot(u_shr, v_shr)

    # Compute Bunkers left and Bunkers right storm motion components
    u_bl = u_adv - deviation_left * v_shr / shr
    v_bl = v_adv + deviation_left * u_shr / shr
    u_br = u_adv + deviation_right * v_shr / shr
    v_br = v_adv - deviation_right * u_shr / shr

    # Deal with points where the shear is zero
    shr_eq_zero = (shr == 0.0)
    if np.any(shr_eq_zero):
        u_bl[shr_eq_zero] = u_adv[shr_eq_zero]
        v_bl[shr_eq_zero] = v_adv[shr_eq_zero]
        u_br[shr_eq_zero] = u_adv[shr_eq_zero]
        v_br[shr_eq_zero] = v_adv[shr_eq_zero]

    return u_bl, v_bl, u_br, v_br


def storm_relative_helicity(z, u, v, u_storm, v_storm, z_bot, z_top,
                            z_sfc=None, u_sfc=None, v_sfc=None,
                            vertical_axis=0):
    
    """
    Computes storm-relative helicity across a specified layer for a given
    storm motion.

    Args:
        z (ndarray): height (m)
        u (ndarray): eastward component of wind (m/s)
        v (ndarray): northward component of wind (m/s)
        u_storm (ndarray): eastward component of storm motion (m/s)
        v_storm (ndarray): northward component of storm motion (m/s)
        z_bot (float or ndarray): height of bottom of layer (m)
        z_top (float or ndarray): height of top of layer (m)
        z_sfc (float or ndarray, optional): surface height (m)
        u_sfc (float or ndarray, optional): eastward component of wind at
            surface (m/s)
        v_sfc (float or ndarray, optional): northward component of wind at
            surface (m/s)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)

    Returns:
        SRH (float or ndarray): storm-relative helicity (m2/s2)

    """

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        z = np.moveaxis(z, vertical_axis, 0)
        u = np.moveaxis(u, vertical_axis, 0)
        v = np.moveaxis(v, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if z.ndim == 1:
        z = np.atleast_2d(z).T  # transpose to preserve vertical axis
        u = np.atleast_2d(u).T
        v = np.atleast_2d(v).T

    # If surface fields are not provided, use lowest level
    if u_sfc is None:
        bottom = 'lowest level'
        k_start = 1  # start loop from second level
        z_sfc = z[0]
        u_sfc = u[0]
        v_sfc = v[0]
    else:
        bottom = 'surface'
        k_start = 0  # start loop from first level
        if z_sfc is None:
            z_sfc = np.zeros_like(u_sfc)  # assumes height AGL

    # Make sure that surface fields are at least 1D
    z_sfc = np.atleast_1d(z_sfc)
    u_sfc = np.atleast_1d(u_sfc)
    v_sfc = np.atleast_1d(v_sfc)

    # Make sure that u_storm and v_storm match dimensions of surface arrays
    if np.isscalar(u_storm):
        u_storm = np.full_like(u_sfc, u_storm)
    if np.isscalar(v_storm):
        v_storm = np.full_like(v_sfc, v_storm)

    # Make sure that z_bot and z_top match dimensions of surface arrays
    if np.isscalar(z_bot):
        z_bot = np.full_like(z_sfc, z_bot)
    if np.isscalar(z_top):
        z_top = np.full_like(z_sfc, z_top)

    # Check if bottom of layer is above top of layer
    if np.any(z_bot > z_top):
        n_pts = np.count_nonzero(z_bot > z_top)
        raise ValueError(f'z_bot is above z_top for {n_pts} points')

    # Check if bottom of layer is below surface
    if np.any(z_bot < z_sfc):
        n_pts = np.count_nonzero(z_bot < z_sfc)
        warnings.warn(f'z_bot is below {bottom} for {n_pts} points')

    # Check if top of layer is above highest level
    if np.any(z_top > z[-1]):
        n_pts = np.count_nonzero(z_top > z[-1])
        warnings.warn(f'z_top is above highest level for {n_pts} points')

    # Note the number of vertical levels
    n_lev = z.shape[0]

    # Initialise level 2 fields
    z2 = z_sfc.copy()
    u2 = u_sfc.copy()
    v2 = v_sfc.copy()

    # Initialise SRH
    SRH = np.zeros_like(z2)

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 fields
        z1 = z2.copy()
        u1 = u2.copy()
        v1 = v2.copy()
        if np.all(z1 >= z_top):
            # can break out of loop
            break

        # Update level 2 fields
        above_sfc = (z[k] > z_sfc)
        z2 = np.where(above_sfc, z[k], z_sfc)
        u2 = np.where(above_sfc, u[k], u_sfc)
        v2 = np.where(above_sfc, v[k], v_sfc)
        if np.all(z2 <= z_bot):
            # can skip this level
            continue
        
        # If crossing bottom of layer, reset level 1
        cross_bot = (z1 < z_bot) & (z2 > z_bot)
        if np.any(cross_bot):
            weight = (z_bot[cross_bot] - z1[cross_bot]) / \
                (z2[cross_bot] - z1[cross_bot])
            u1[cross_bot] = (1 - weight) * u1[cross_bot] + \
                weight * u2[cross_bot]
            v1[cross_bot] = (1 - weight) * v1[cross_bot] + \
                weight * v2[cross_bot]
            z1[cross_bot] = z_bot[cross_bot]

        # If crossing top of layer, reset level 2
        cross_top = (z1 < z_top) & (z2 > z_top)
        if np.any(cross_top):
            weight = (z_top[cross_top] - z1[cross_top]) / \
                (z2[cross_top] - z1[cross_top])
            u2[cross_top] = (1 - weight) * u1[cross_top] + \
                weight * u2[cross_top]
            v2[cross_top] = (1 - weight) * v1[cross_top] + \
                weight * v2[cross_top]
            z2[cross_top] = z_top[cross_top]

        # If within layer, update SRH
        in_layer = (z1 >= z_bot) & (z2 <= z_top)
        if np.any(in_layer):
            SRH[in_layer] += (u2[in_layer] - u_storm[in_layer]) * \
                (v1[in_layer] - v_storm[in_layer]) - \
                (u1[in_layer] - u_storm[in_layer]) * \
                (v2[in_layer] - v_storm[in_layer])

    # Deal with points where z_bot or z_top are undefined
    is_nan = np.isnan(z_bot) | np.isnan(z_top)
    if np.any(is_nan):
        SRH[is_nan] = np.nan

    return SRH.squeeze()
