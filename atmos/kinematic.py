import numpy as np


def wind_speed(u, v):
    """
    Computes wind speed from wind components.

    Args:
        u: zonal wind velocity (m/s)
        v: meridional wind velocity (m/s)

    Returns:
        wspd: wind speed (m/s)

    """
    wspd = np.hypot(u, v)
    
    return wspd


def wind_direction(u, v):
    """
    Computes wind direction from wind components.

    Args:
        u: zonal wind velocity (m/s)
        v: meridional wind velocity (m/s)

    Returns:
        wdir: wind direciton (deg)

    """
    wdir = 90. - np.degrees(np.arctan2(-v, -u))
    
    origshape = wdir.shape
    wdir = np.atleast_1d(wdir)
    
    # Correct negative values
    neg_mask = np.array(wdir < 0.)
    if np.any(neg_mask):
        wdir[neg_mask] += 360.

    # Set to 0 where wind speed is zero
    calm_mask = (np.asanyarray(u) == 0.) & (np.asanyarray(v) == 0.)
    if np.any(calm_mask):
        wdir[calm_mask] = 0.
        
    return wdir.reshape(origshape)
    

def wind_components(wspd, wdir):
    """
    Computes wind components from wind speed and direction.

    Args:
        wspd: wind speed (m/s)
        wdir: wind direciton (deg)

    Returns:
        u: zonal wind velocity (m/s)
        v: meridional wind velocity (m/s)

    """
    u = -wspd * np.sin(np.radians(wdir))
    v = -wspd * np.cos(np.radians(wdir))
    
    return u, v
