import numpy as np
from atmos.utils import interp_vector_to_height_level, layer_mean_vector


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
    
    wdir = np.atleast_1d(wdir)

    
    # Correct negative values
    neg_mask = np.array(wdir < 0.)
    if np.any(neg_mask):
        wdir[neg_mask] += 360.

    # Set to 0 where wind speed is zero
    calm_mask = (np.asanyarray(u) == 0.) & (np.asanyarray(v) == 0.)
    if np.any(calm_mask):
        wdir[calm_mask] = 0.
        
    if len(wdir) == 1:
        wdir = wdir[0]

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


def bulk_wind_difference(z, u, v, z_bot, z_top, vertical_axis=0):
    """
    Computes bulk wind difference across a specified layer.

    Args:
        z (float or ndarray): height (m)
        u (float or ndarray): zonal wind velocity (m/s)
        v (float or ndarray): meridional wind velocity (m/s)
        z_bot (float or ndarray): height of bottom of layer (m)
        z_top (float or ndarray): height of top of layer (m)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)

    Returns:
        BWDmag (float or ndarray): bulk wind difference magnitude (m/s)
        BWDdir (float or ndarray, optional): bulk wind difference direction (deg)

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

    # Make sure that z_bot and z_top are at least 1D
    z_bot = np.atleast_1d(z_bot)
    z_top = np.atleast_1d(z_top)

    # Note the number of vertical levels
    n_lev = z.shape[0]

    # Initialise winds at bottom and top of layer
    u_bot = u[0]
    v_bot = v[0]
    u_top = u[-1]
    v_top = v[-1]

    # Initialise level 2 fields
    z2 = z[0]
    u2 = u[0]
    v2 = v[0]

    # Check if bottom of layer is below lowest level
    if np.any(z_bot < z[0]):
        n_pts = np.count_nonzero(z_bot < z[0])
        print(f'WARNING: z_bot is below lowest level for {n_pts} points')

    # Check if top of layer is above highest level
    if np.any(z_top > z[-1]):
        n_pts = np.count_nonzero(z_top > z[-1])
        print(f'WARNING: z_top is above highest level for {n_pts} points')

    # Loop over levels
    for k in range(1, n_lev):

        # Update level 1 fields
        z1 = z2.copy()
        u1 = u2.copy()
        v1 = v2.copy()

        # Update level 2 fields
        z2 = z[k]
        u2 = u[k]
        v2 = v[k]

        if np.all(z2 < z_bot):
            # can skip this level
            continue
        if np.all(z1 >= z_top):
            # can break out of loop
            break

        # Set winds at bottom of layer
        cross_bot = (z1 < z_bot) & (z2 >= z_bot)
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

    if len(BWDu) == 1:
        return BWDu[0], BWDv[0]
    else:
        return BWDu, BWDv


def bunkers_storm_motion(z, u, v, vertical_axis=0, level_weights=None,
                         mean_wind_layer_base=0, mean_wind_layer_top=6000.0,
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
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)
        level_weights (ndarray, optional): weights to apply to winds at each
            level in mean wind calculations (default is None, in which case no
            weighting is applied)
        mean_wind_layer_base (float or ndarray, optional): height of base of
            mean wind layer (m) (default is 0.0)
        mean_wind_layer_top (float or ndarray, optional): height of top of
            mean wind layer (m) (default is 6000.0)
        shear_layer_base (float or ndarray, optional): height of base of shear
            layer (m) (default is 0.0)
        shear_layer_top (float or ndarray, optional): height of top of shear
            layer (m) (default is 0.0)
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
    u_adv, v_adv = layer_mean_vector(
        z, u, v, mean_wind_layer_base, mean_wind_layer_top,
        vertical_axis=vertical_axis, level_weights=level_weights
    )
    
    # Compute shear vector
    if shear_layer_base_average == 0.0:
        u_bot, v_bot = interp_vector_to_height_level(
            z, u, v, shear_layer_base, vertical_axis=vertical_axis
        )
    else:
        u_bot, v_bot = layer_mean_vector(
            z, u, v, shear_layer_base, shear_layer_base+shear_layer_base_average,
            vertical_axis=vertical_axis, level_weights=level_weights
        )
    if shear_layer_top_average == 0.0:
        u_top, v_top = interp_vector_to_height_level(
            z, u, v, shear_layer_top, vertical_axis=vertical_axis
        )
    else:
        u_top, v_top = layer_mean_vector(
            z, u, v, shear_layer_top-shear_layer_top_average, shear_layer_top,
            vertical_axis=vertical_axis, level_weights=level_weights
        )
    u_shr = u_top - u_bot
    v_shr = v_top - v_bot

    # Compute the shear magnitude
    shr = np.hypot(u_shr, v_shr)

    # Compute the Bunkers left and Bunkers right storm motion components
    u_bl = u_adv - deviation_left * v_shr / shr
    v_bl = v_adv + deviation_left * u_shr / shr
    u_br = u_adv + deviation_right * v_shr / shr
    v_br = v_adv - deviation_right * u_shr / shr

    return u_bl, v_bl, u_br, v_br


def storm_relative_helicity(z, u, v, u_storm, v_storm, z_bot, z_top,
                            vertical_axis=0):
    
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

    # Make sure that u_storm and v_storm are at least 1D
    u_storm = np.atleast_1d(u_storm)
    v_storm = np.atleast_1d(v_storm)

    # Make sure that z_bot and z_top are at least 1D
    z_bot = np.atleast_1d(z_bot)
    z_top = np.atleast_1d(z_top)

    # Note the number of vertical levels
    n_lev = z.shape[0]

    # Check if bottom of layer is below lowest level
    if np.any(z_bot < z[0]):
        n_pts = np.count_nonzero(z_bot < z[0])
        print(f'WARNING: z_bot is below lowest level for {n_pts} points')

    # Check if top of layer is above highest level
    if np.any(z_top > z[-1]):
        n_pts = np.count_nonzero(z_top > z[-1])
        print(f'WARNING: z_top is above highest level for {n_pts} points')

    # Initialise level 2 fields
    z2 = z[0]
    u2 = u[0]
    v2 = v[0]

    # Initialise SRH
    SRH = np.zeros_like(z2)

    # Loop over levels
    for k in range(1, n_lev):

        # Update level 1 fields
        z1 = z2.copy()
        u1 = u2.copy()
        v1 = v2.copy()

        # Update level 2 fields
        z2 = z[k]
        u2 = u[k]
        v2 = v[k]

        if np.all(z2 <= z_bot):
            # can skip this level
            continue
        if np.all(z1 >= z_top):
            # can break out of loop
            break
        
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

    if len(SRH) == 1:
        return SRH[0]
    else:
        return SRH

