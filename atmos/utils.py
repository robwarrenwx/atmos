import numpy as np


def height_of_pressure_level(p, z, pi, p_sfc=None, z_sfc=None,
                             vertical_axis=0):
    """
    Finds the height corresponding to a specified pressure level, assuming
    that height varies linearly with log(pressure).

    Args:
        p (ndarray): pressure profile(s) (Pa)
        z (ndarray): height profile(s) (m)
        pi (float or ndarray): pressure of level (Pa)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        z_sfc (float or ndarray, optional): surface height (m)
        vertical_axis (int, optional):profile array axis corresponding to
            vertical dimension (default is 0)

    Returns:
        zi (float or ndarray): height of level (m)

    """
    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        p = np.moveaxis(p, vertical_axis, 0)
        z = np.moveaxis(z, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if p.ndim == 1:
        p = np.atleast_2d(p).T  # transpose to preserve vertical axis
        z = np.atleast_2d(z).T

    # If surface pressure is not provided, use lowest level
    if p_sfc is None:
        bottom = 'lowest level'
        k_start = 1  # start loop from second level
        p_sfc = p[0]
        z_sfc = z[0]
    else:
        bottom = 'surface'
        k_start = 0  # start loop from first level
        if z_sfc is None:
            z_sfc = np.zeros_like(p_sfc)  # assumes height AGL

    # Make sure that surface pressure and height are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    z_sfc = np.atleast_1d(z_sfc)

    # Make sure that pi matches shape of surface fields
    if np.isscalar(pi):
        pi = np.full_like(p_sfc, pi)

    # Note the number of vertical levels
    n_lev = p.shape[0]

    # Check if pi is below the surface
    if np.any(pi > p_sfc):
        n_pts = np.count_nonzero(pi > p_sfc)
        print(f'WARNING: pi is below {bottom} for {n_pts} points')

    # Check if pi is above highest level
    if np.any(pi < p[-1]):
        n_pts = np.count_nonzero(pi < p[-1])
        print(f'WARNING: pi is above highest level for {n_pts} points')

    # Initialise height of pi
    zi = np.full_like(z_sfc, np.nan)

    # Initialise level 2 pressure and height
    p2 = p_sfc.copy()
    z2 = z_sfc.copy()

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 pressure and height
        p1 = p2.copy()
        z1 = z2.copy()
        if np.all(p1 <= pi):
            # can break out of loop
            break

        # Set level 2 pressure and height
        above_sfc = (p[k] < p_sfc)
        p2 = np.where(above_sfc, p[k], p_sfc)
        z2 = np.where(above_sfc, z[k], z_sfc)
        if np.all(p2 > p_sfc):
            # can skip this level
            continue

        # Interpolate to get height of pi
        crossed = (p1 > pi) & (p2 <= pi)
        if np.any(crossed):
            weight = np.log(p1[crossed] / pi[crossed]) / \
                np.log(p1[crossed] / p2[crossed])
            zi[crossed] = (1 - weight) * z1[crossed] + \
                weight * z2[crossed]

    # Deal with points where pi is at the surface
    pi_at_sfc = (pi == p_sfc)
    if np.any(pi_at_sfc):
        zi[pi_at_sfc] = z_sfc[pi_at_sfc]

    if len(zi) == 1:
        return zi.item()
    else:
        return zi


def pressure_of_height_level(z, p, zi, z_sfc=None, p_sfc=None,
                             vertical_axis=0):
    """
    Finds the height corresponding to a specified pressure level, assuming
    that height varies linearly with log(pressure).

    Args:
        z (ndarray): height profile(s) (m)
        p (ndarray): pressure profile(s) (Pa)
        zi (float or ndarray): height of level (m)
        z_sfc (float or ndarray, optional): surface height (m)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        vertical_axis (int, optional):profile array axis corresponding to
            vertical dimension (default is 0)

    Returns:
        pi (float or ndarray): pressure of level (Pa)

    """
    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        z = np.moveaxis(z, vertical_axis, 0)
        p = np.moveaxis(p, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if z.ndim == 1:
        z = np.atleast_2d(z).T  # transpose to preserve vertical axis
        p = np.atleast_2d(p).T

    # If surface pressure is not provided, use lowest level
    if p_sfc is None:
        bottom = 'lowest level'
        k_start = 1  # start loop from second level
        p_sfc = p[0]
        z_sfc = z[0]
    else:
        bottom = 'surface'
        k_start = 0  # start loop from first level
        if z_sfc is None:
            z_sfc = np.zeros_like(p_sfc)  # assumes height AGL

    # Make sure that surface pressure and height are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    z_sfc = np.atleast_1d(z_sfc)

    # Make sure that zi matches shape of surface fields
    if np.isscalar(zi):
        zi = np.full_like(z_sfc, zi)

    # Note the number of vertical levels
    n_lev = p.shape[0]

    # Check if zi is below surface
    if np.any(zi < 0):
        n_pts = np.count_nonzero(zi < 0)
        print(f'WARNING: zi is below {bottom} for {n_pts} points')

    # Check if zi is above highest level
    if np.any(zi > z[-1]):
        n_pts = np.count_nonzero(zi > z[-1])
        print(f'WARNING: zi is above highest level for {n_pts} points')

    # Initialise pressure of zi
    pi = np.full_like(p_sfc, np.nan)

    # Initialise level 2 height and pressure
    z2 = z_sfc.copy()
    p2 = p_sfc.copy()

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 height and pressure
        z1 = z2.copy()
        p1 = p2.copy()
        if np.all(z1 >= zi):
            # can break out of loop
            break

        # Set level 2 height and pressure
        above_sfc = (p[k] < p_sfc)
        z2 = np.where(above_sfc, z[k], z_sfc)
        p2 = np.where(above_sfc, p[k], p_sfc)
        if np.all(z2 < zi):
            # can skip this level
            continue

        # Interpolate to get height of pi
        crossed = (z1 < zi) & (z2 >= zi)
        if np.any(crossed):
            weight = (zi[crossed] - z1[crossed]) / \
                (z2[crossed] - z1[crossed])
            pi[crossed] = p1 ** (1 - weight) + \
                p2[crossed] ** weight

    # Deal with points where zi is at the surface
    zi_at_sfc = (zi == z_sfc)
    if np.any(zi_at_sfc):
        pi[zi_at_sfc] = p_sfc[zi_at_sfc]

    if len(pi) == 1:
        return pi.item()
    else:
        return pi


def height_of_temperature_level(z, T, Ti, z_sfc=None, T_sfc=None,
                                vertical_axis=0):
    """
    Finds the lowest height corresponding to a specified temperature, assuming
    that temperature varies linearly with height.

    Args:
        z (ndarray): height profile(s) (m)
        T (ndarray): temperature profile(s) (K)
        Ti (float): temperature level (K)
        z_sfc (float or ndarray, optional): surface height (m)
        T_sfc (float or ndarray, optional): surface temperature (K)
        vertical_axis (int, optional): profile array axis corresponding to
            vertical dimension (default is 0)

    Returns:
        zi (float or ndarray): height of temperature level (m)

    """

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        z = np.moveaxis(z, vertical_axis, 0)
        T = np.moveaxis(T, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if z.ndim == 1:
        z = np.atleast_2d(z).T  # transpose to preserve vertical axis
        T = np.atleast_2d(T).T

    # If surface temperature is not provided, use lowest level
    if T_sfc is None:
        bottom = 'lowest level'
        k_start = 1  # start loop from second level
        T_sfc = T[0]
        z_sfc = z[0]
    else:
        bottom = 'surface'
        k_start = 0  # start loop from first level
        if z_sfc is None:
            z_sfc = np.zeros_like(T_sfc)  # assumes height AGL

    # Make sure that surface fields are at least 1D
    z_sfc = np.atleast_1d(z_sfc)
    T_sfc = np.atleast_1d(T_sfc)

    # Check if Ti is above the surface temperature
    if np.any(Ti > T_sfc):
        n_pts = np.count_nonzero(Ti > T_sfc)
        print(f'WARNING: Ti is above {bottom} temperature for {n_pts} points')

    # Note the number of vertical levels
    n_lev = z.shape[0]

    # Create array for height at temperature level
    zi = np.full_like(z_sfc, np.nan)

    # Create boolean array to indicate whether level has been found
    found = np.zeros_like(zi).astype(bool)

    # Initialise level 2 fields
    z2 = z_sfc.copy()
    T2 = T_sfc.copy()

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 fields
        z1 = z2.copy()
        T1 = T2.copy()

        # Set level 2 fields
        above_sfc = (z[k] > z_sfc)
        z2 = np.where(above_sfc, z[k], z_sfc)
        T2 = np.where(above_sfc, T[k], T_sfc)

        if np.all(T2 > Ti):
            # can skip this level
            continue

        # Interpolate to get height at Ti
        crossed = (T1 > Ti) & (T2 <= Ti) & np.logical_not(found)
        if np.any(crossed):
            weight = (T1[crossed] - Ti) / \
                (T1[crossed] - T2[crossed])
            zi[crossed] = (1 - weight) * z1[crossed] + weight * z2[crossed]
            found[crossed] = True
            if np.all(found):
                break

    # Deal with points where Ti is at the surface
    Ti_at_sfc = (Ti == T_sfc) & np.logical_not(found)
    if np.any(Ti_at_sfc):
        zi[Ti_at_sfc] = z_sfc[Ti_at_sfc]

    if len(zi) == 1:
        return zi.item()
    else:
        return zi


def pressure_of_temperature_level(p, T, Ti, p_sfc=None, T_sfc=None,
                                  vertical_axis=0):
    """
    Finds the highest pressure corresponding to a specified temperature,
    assuming that temperature varies linearly with log(pressure).

    Args:
        p (ndarray): pressure profile(s) (Pa)
        T (ndarray): temperature profile(s) (K)
        Ti (float): temperature level (K)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        T_sfc (float or ndarray, optional): surface temperature (K)
        vertical_axis (int, optional): profile array axis corresponding to
            vertical dimension (default is 0)

    Returns:
        pi (float or ndarray): pressure of temperature level (Pa)

    """

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        p = np.moveaxis(p, vertical_axis, 0)
        T = np.moveaxis(T, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if p.ndim == 1:
        p = np.atleast_2d(p).T  # transpose to preserve vertical axis
        T = np.atleast_2d(T).T

    # If surface temperature is not provided, use lowest level
    if T_sfc is None:
        bottom = 'lowest level'
        k_start = 1  # start loop from second level
        T_sfc = T[0]
        p_sfc = p[0]
    else:
        bottom = 'surface'
        k_start = 0  # start loop from first level

    # Make sure that surface fields are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    T_sfc = np.atleast_1d(T_sfc)

    # Check if Ti is above the surface temperature
    if np.any(Ti > T_sfc):
        n_pts = np.count_nonzero(Ti > T_sfc)
        print(f'WARNING: Ti is above {bottom} temperature for {n_pts} points')

    # Note the number of vertical levels
    n_lev = p.shape[0]

    # Create array for pressure at temperature level
    pi = np.full_like(p_sfc, np.nan)

    # Create boolean array to indicate whether level has been found
    found = np.zeros_like(pi).astype(bool)

    # Initialise level 2 fields
    p2 = p_sfc.copy()
    T2 = T_sfc.copy()

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 fields
        p1 = p2.copy()
        T1 = T2.copy()

        # Set level 2 fields
        above_sfc = (p[k] < p_sfc)
        p2 = np.where(above_sfc, p[k], p_sfc)
        T2 = np.where(above_sfc, T[k], T_sfc)

        if np.all(T2 > Ti):
            # can skip this level
            continue

        # Interpolate to get pressure at Ti
        crossed = (T1 > Ti) & (T2 <= Ti) & np.logical_not(found)
        if np.any(crossed):
            weight = (T1[crossed] - Ti) / \
                (T1[crossed] - T2[crossed])
            pi[crossed] = p1[crossed] ** (1 - weight) + p2[crossed] ** weight
            found[crossed] = True
            if np.all(found):
                break

    # Deal with points where Ti is at the surface
    Ti_at_sfc = (Ti == T_sfc) & np.logical_not(found)
    if np.any(Ti_at_sfc):
        pi[Ti_at_sfc] = p_sfc[Ti_at_sfc]

    if len(pi) == 1:
        return pi.item()
    else:
        return pi


def interpolate_scalar_to_height_level(z, s, zi, z_sfc=None, s_sfc=None,
                                       vertical_axis=0):
    """
    Interpolates a scalar variable to a specified height level, assuming
    linear variation with height.

    Args:
        z (ndarray): height profile(s) (m)
        s (ndarray): profile(s) of scalar variable
        zi (float or ndarray): height of level (m)
        z_sfc (float or ndarray, optional): surface height (m)
        s_sfc (float or ndarray, optional): scalar variable at surface
        vertical_axis (int, optional): profile array axis corresponding to
            vertical dimension (default is 0)

    Returns:
        si (float or ndarray): scalar variable at level

    """

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        z = np.moveaxis(z, vertical_axis, 0)
        s = np.moveaxis(s, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if z.ndim == 1:
        z = np.atleast_2d(z).T  # transpose to preserve vertical axis
        s = np.atleast_2d(s).T

    # If surface fields are not provided, use lowest level
    if s_sfc is None:
        bottom = 'lowest level'
        k_start = 1  # start loop from second level
        z_sfc = z[0]
        s_sfc = s[0]
    else:
        bottom = 'surface'
        k_start = 0  # start loop from first level
        if z_sfc is None:
            z_sfc = np.zeros_like(s_sfc)  # assumes height AGL

    # Make sure that surface fields are at least 1D
    z_sfc = np.atleast_1d(z_sfc)
    s_sfc = np.atleast_1d(s_sfc)

    # Make sure that zi matches shape of surface fields
    if np.isscalar(zi):
        zi = np.full_like(z_sfc, zi)

    # Check if zi is below the surface
    if np.any(zi < z_sfc):
        n_pts = np.count_nonzero(zi < z_sfc)
        print(f'WARNING: zi is below {bottom} for {n_pts} points')

    # Check if zi is above highest level
    if np.any(zi > z[-1]):
        n_pts = np.count_nonzero(zi > z[-1])
        print(f'WARNING: zi is above highest level for {n_pts} points')

    # Note the number of vertical levels
    n_lev = z.shape[0]

    # Initialise scalar at zi
    si = np.full_like(s_sfc, np.nan)

    # Initialise level 2 fields
    z2 = z_sfc.copy()
    s2 = s_sfc.copy()

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 fields
        z1 = z2.copy()
        s1 = s2.copy()
        if np.all(z1 >= zi):
            # can break out of loop
            break

        # Set level 2 fields
        above_sfc = (z[k] > z_sfc)
        z2 = np.where(above_sfc, z[k], z_sfc)
        s2 = np.where(above_sfc, s[k], s_sfc)
        if np.all(z2 < zi):
            # can skip this level
            continue

        # Interpolate to get scalar at zi
        crossed = (z1 < zi) & (z2 >= zi)
        if np.any(crossed):
            weight = (zi[crossed] - z1[crossed]) / \
                (z2[crossed] - z1[crossed])
            si[crossed] = (1 - weight) * s1[crossed] + \
                weight * s2[crossed]

    # Deal with points where zi is at the surface
    zi_at_sfc = (zi == z_sfc)
    if np.any(zi_at_sfc):
        si[zi_at_sfc] = s_sfc[zi_at_sfc]

    if len(si) == 1:
        return si.item()
    else:
        return si


def interpolate_vector_to_height_level(z, u, v, zi, z_sfc=None, u_sfc=None,
                                       v_sfc=None, vertical_axis=0):
    """
    Interpolates vector components to a specified height level, assuming
    linear variation with height.

    Args:
        z (ndarray): height profile(s) (m)
        u (ndarray): profile(s) of eastward component of vector
        v (ndarray): profile(s) northward component of vector
        zi (float or ndarray): height of level (m)
        z_sfc (float or ndarray, optional): surface height (m)
        u_sfc (float or ndarray, optional): eastward component of vector at
            surface
        v_sfc (float or ndarray, optional): northward component of vector at
            surface
        vertical_axis (int, optional): profile array axis corresponding to
            vertical dimension (default is 0)

    Returns:
        ui (float or ndarray): eastward component of vector at level (m/s)
        vi (float or ndarray): northward component of vector at level (m/s)

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

    # If surface value of vector is not provided, use lowest level
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

    # Make sure that zi matches shape of surface fields
    if np.isscalar(zi):
        zi = np.full_like(z_sfc, zi)

    # Check if zi is below surface
    if np.any(zi < z_sfc):
        n_pts = np.count_nonzero(zi < z_sfc)
        print(f'WARNING: zi is below {bottom} for {n_pts} points')

    # Check if zi is above highest level
    if np.any(zi > z[-1]):
        n_pts = np.count_nonzero(zi > z[-1])
        print(f'WARNING: zi is above highest level for {n_pts} points')

    # Note the number of vertical levels
    n_lev = z.shape[0]

    # Initialise vector components at zi
    ui = np.full_like(u_sfc, np.nan)
    vi = np.full_like(v_sfc, np.nan)

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
        if np.all(z1 >= zi):
            # can break out of loop
            break

        # Set level 2 fields
        above_sfc = (z[k] > z_sfc)
        z2 = np.where(above_sfc, z[k], z_sfc)
        u2 = np.where(above_sfc, u[k], u_sfc)
        v2 = np.where(above_sfc, v[k], v_sfc)
        if np.all(z2 < zi):
            # can skip this level
            continue
        
        # Interpolate to get vector components at zi
        crossed = (z1 < zi) & (z2 >= zi)
        if np.any(crossed):
            weight = (zi[crossed] - z1[crossed]) / \
                (z2[crossed] - z1[crossed])
            ui[crossed] = (1 - weight) * u1[crossed] + \
                weight * u2[crossed]
            vi[crossed] = (1 - weight) * v1[crossed] + \
                weight * v2[crossed]

    # Deal with points where zi is at the surface
    zi_at_sfc = (zi == z_sfc)
    if np.any(zi_at_sfc):
        ui[zi_at_sfc] = u_sfc[zi_at_sfc]
        vi[zi_at_sfc] = v_sfc[zi_at_sfc]

    if len(ui) == 1:
        return ui.item(), vi.item()
    else:
        return ui, vi


def interpolate_scalar_to_pressure_level(p, s, pi, p_sfc=None, s_sfc=None,
                                         vertical_axis=0):
    """
    Interpolates a scalar variable to a specified pressure level, assuming
    linear variation with log(pressure).

    Args:
        p (ndarray): pressure profile(s) (Pa)
        s (ndarray): prrofile(s) of scalar variable
        pi (float or ndarray): pressure of level (Pa)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        s_sfc (float or ndarray, optional): scalar variable at surface
        vertical_axis (int, optional): profile array axis corresponding to
            vertical dimension (default is 0)

    Returns:
        si (float or ndarray): scalar variable at level

    """

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        p = np.moveaxis(p, vertical_axis, 0)
        s = np.moveaxis(s, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if p.ndim == 1:
        p = np.atleast_2d(p).T  # transpose to preserve vertical axis
        s = np.atleast_2d(s).T

    # If surface fields are not provided, use lowest level
    if p_sfc is None:
        bottom = 'lowest level'
        k_start = 1  # start loop from second level
        p_sfc = p[0]
        s_sfc = s[0]
    else:
        bottom = 'surface'
        k_start = 0  # start loop from first level

    # Make sure that surface fields are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    s_sfc = np.atleast_1d(s_sfc)

    # Make sure that pi matches shape of surface fields
    if np.isscalar(pi):
        pi = np.full_like(p_sfc, pi)

    # Check if pi is below surface
    if np.any(pi > p[0]):
        n_pts = np.count_nonzero(pi > p_sfc)
        print(f'WARNING: pi is below {bottom} for {n_pts} points')

    # Check if pi is above highest level
    if np.any(pi < p[-1]):
        n_pts = np.count_nonzero(pi < p[-1])
        print(f'WARNING: pi is above highest level for {n_pts} points')

    # Note the number of vertical levels
    n_lev = p.shape[0]

    # Initialise scalar at zi
    si = np.full_like(s_sfc, np.nan)

    # Initialise level 2 fields
    p2 = p_sfc.copy()
    s2 = s_sfc.copy()

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 fields
        p1 = p2.copy()
        s1 = s2.copy()
        if np.all(p1 <= pi):
            # can break out of loop
            break

        # Set level 2 fields
        above_sfc = (p[k] < p_sfc)
        p2 = np.where(above_sfc, p[k], p_sfc)
        s2 = np.where(above_sfc, s[k], s_sfc)
        if np.all(p2 > pi):
            # can skip this level
            continue

        # Interpolate to get scalar at pi
        crossed = (p1 > pi) & (p2 <= pi)
        if np.any(crossed):
            weight = np.log(p1[crossed] / pi[crossed]) / \
                np.log(p1[crossed] / p2[crossed])
            si[crossed] = (1 - weight) * s1[crossed] + \
                weight * s2[crossed]

    # Deal with points where pi is at the surface
    pi_at_sfc = (pi == p_sfc)
    if np.any(pi_at_sfc):
        si[pi_at_sfc] = s_sfc[pi_at_sfc]

    if len(si) == 1:
        return si.item()
    else:
        return si


def interpolate_vector_to_pressure_level(p, u, v, pi, p_sfc=None, u_sfc=None,
                                         v_sfc=None, vertical_axis=0):
    """
    Interpolates vector components to a specified pressure level, assuming
    linear variation with log(pressure).

    Args:
        p (ndarray): pressure profile(s) (Pa)
        u (ndarray): profile(s) of eastward component of vector
        v (ndarray): profile(s) of northward component of vector
        pi (float or ndarray): pressure of level (Pa)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        u_sfc (float or ndarray, optional): eastward component of vector at
            surface
        v_sfc (float or ndarray, optional): northward component of vector at
            surface
        vertical_axis (int, optional): profile array axis corresponding to
            vertical dimension (default is 0)

    Returns:
        ui (float or ndarray): eastward component of vector at level (m/s)
        vi (float or ndarray): northward component of vector at level (m/s)

    """

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        p = np.moveaxis(p, vertical_axis, 0)
        u = np.moveaxis(u, vertical_axis, 0)
        v = np.moveaxis(v, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if p.ndim == 1:
        p = np.atleast_2d(p).T  # transpose to preserve vertical axis
        u = np.atleast_2d(u).T
        v = np.atleast_2d(v).T

    # If surface pressure is not provided, use lowest level
    if u_sfc is None:
        bottom = 'lowest level'
        k_start = 1  # start loop from second level
        p_sfc = p[0]
        u_sfc = u[0]
        v_sfc = v[0]
    else:
        bottom = 'surface'
        k_start = 0  # start loop from first level

    # Make sure that surface fields are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    u_sfc = np.atleast_1d(u_sfc)
    v_sfc = np.atleast_1d(v_sfc)

    # Make sure that pi matches shape of surface fields
    if np.isscalar(pi):
        pi = np.full_like(p_sfc, pi)

    # Check if pi is below surface
    if np.any(pi > p_sfc):
        n_pts = np.count_nonzero(pi > p_sfc)
        print(f'WARNING: pi is below {bottom} for {n_pts} points')

    # Check if pi is above highest level
    if np.any(pi < p[-1]):
        n_pts = np.count_nonzero(pi < p[-1])
        print(f'WARNING: pi is above highest level for {n_pts} points')

    # Note the number of vertical levels
    n_lev = p.shape[0]

    # Initialise vector components at zi
    ui = np.full_like(u_sfc, np.nan)
    vi = np.full_like(v_sfc, np.nan)

    # Initialise level 2 fields
    p2 = p_sfc.copy()
    u2 = u_sfc.copy()
    v2 = v_sfc.copy()

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 fields
        p1 = p2.copy()
        u1 = u2.copy()
        v1 = v2.copy()
        if np.all(p1 <= pi):
            # can break out of loop
            break

        # Set level 2 fields
        above_sfc = (p[k] < p_sfc)
        p2 = np.where(above_sfc, p[k], p_sfc)
        u2 = np.where(above_sfc, u[k], u_sfc)
        v2 = np.where(above_sfc, v[k], v_sfc)
        if np.all(p2 > pi):
            # can skip this level
            continue
        
        # Interpolate to get vector components at zi
        crossed = (p1 > pi) & (p2 <= pi)
        if np.any(crossed):
            weight = np.log(p1[crossed] / pi[crossed]) / \
                np.log(p1[crossed] / p2[crossed])
            ui[crossed] = (1 - weight) * u1[crossed] + \
                weight * u2[crossed]
            vi[crossed] = (1 - weight) * v1[crossed] + \
                weight * v2[crossed]

    # Deal with points where pi is at the surface
    pi_at_sfc = (pi == p_sfc)
    if np.any(pi_at_sfc):
        ui[pi_at_sfc] = u_sfc[pi_at_sfc]
        vi[pi_at_sfc] = v_sfc[pi_at_sfc]

    if len(ui) == 1:
        return ui.item(), vi.item()
    else:
        return ui, vi


def layer_mean_scalar(z, s, z_bot, z_top, z_sfc=None, s_sfc=None,
                      vertical_axis=0, level_weights=None,
                      surface_weight=None):
    """
    Computes the mean of a scalar variable between two specified height levels,
    with optional weighting.

    Args:
        z (ndarray): height profile(s) (m)
        s (ndarray): profile(s) of scalar variable
        z_bot (float or ndarray): height of bottom of layer (m)
        z_top (float or ndarray): height of top of layer (m)
        z_sfc (float or ndarray, optional): surface height (m)
        s_sfc (float or ndarray, optional): scalar variable at surface
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)
        level_weights (ndarray, optional): weights to apply to scalar at each
            level (default is None, in which case no weighting is applied)
        surface_weight (float or ndarray, optional): weight to apply to scalar
            at surface (default is None, in which case no weighting is applied)

    Returns:
        s_mean (float or ndarray): layer-mean scalar variable

    """

    if level_weights is None:
        w = np.ones_like(z)
    else:
        w = level_weights

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        z = np.moveaxis(z, vertical_axis, 0)
        s = np.moveaxis(s, vertical_axis, 0)
        w = np.moveaxis(w, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if z.ndim == 1:
        z = np.atleast_2d(z).T  # transpose to preserve vertical axis
        s = np.atleast_2d(s).T
        w = np.atleast_2d(w).T

    # If surface fields are not provided, use lowest level
    if s_sfc is None:
        bottom = 'lowest level'
        k_start = 1  # start loop from second level
        z_sfc = z[0]
        s_sfc = s[0]
        w_sfc = w[0]
    else:
        bottom = 'surface'
        k_start = 0  # start loop from first level
        if z_sfc is None:
            z_sfc = np.zeros_like(s_sfc)  # assumes height AGL
        if surface_weight is None:
            w_sfc = np.ones_like(s_sfc)

    # Make sure that surface fields are at least 1D
    z_sfc = np.atleast_1d(z_sfc)
    s_sfc = np.atleast_1d(s_sfc)
    w_sfc = np.atleast_1d(w_sfc)

    # Make sure that z_bot and z_top match shape of surface arrays
    if np.isscalar(z_bot):
        z_bot = np.full_like(z_sfc, z_bot)
    if np.isscalar(z_top):
        z_top = np.full_like(z_sfc, z_top)

    # Check if bottom of layer is above top of layer
    if np.any(z_bot > z_top):
        n_pts = np.count_nonzero(z_bot > z_top)
        print(f'WARNING: z_bot is above z_top for {n_pts} points')

    # Check if bottom of layer is below surface
    if np.any(z_bot < z_sfc):
        n_pts = np.count_nonzero(z_bot < z_sfc)
        print(f'WARNING: z_bot is below {bottom} for {n_pts} points')

    # Check if top of layer is above highest level
    if np.any(z_top > z[-1]):
        n_pts = np.count_nonzero(z_top > z[-1])
        print(f'WARNING: z_top is above highest level for {n_pts} points')

    # Note the number of vertical levels
    n_lev = z.shape[0]

    # Initialise level 2 fields
    z2 = z_sfc.copy()
    s2 = s_sfc.copy()
    w2 = w_sfc.copy()

    # Initialise intergrals
    s_int = np.zeros_like(z2)
    w_int = np.zeros_like(z2)

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 fields
        z1 = z2.copy()
        s1 = s2.copy()
        w1 = w2.copy()
        if np.all(z1 >= z_top):
            # can break out of loop
            break

        # Update level 2 fields
        above_sfc = (z[k] > z_sfc)
        z2 = np.where(above_sfc, z[k], z_sfc)
        s2 = np.where(above_sfc, s[k], s_sfc)
        w2 = np.where(above_sfc, w[k], w_sfc)
        if np.all(z2 <= z_bot):
            # can skip this level
            continue
        
        # If crossing bottom of layer, reset level 1
        cross_bot = (z1 < z_bot) & (z2 > z_bot)
        if np.any(cross_bot):
            weight = (z_bot[cross_bot] - z1[cross_bot]) / \
                (z2[cross_bot] - z1[cross_bot])
            s1[cross_bot] = (1 - weight) * s1[cross_bot] + \
                weight * s2[cross_bot]
            w1[cross_bot] = (1 - weight) * w1[cross_bot] + \
                weight * w2[cross_bot]
            z1[cross_bot] = z_bot[cross_bot]

        # If crossing top of layer, reset level 2
        cross_top = (z1 < z_top) & (z2 > z_top)
        if np.any(cross_top):
            weight = (z_top[cross_top] - z1[cross_top]) / \
                (z2[cross_top] - z1[cross_top])
            s2[cross_top] = (1 - weight) * s1[cross_top] + \
                weight * s2[cross_top]
            w2[cross_top] = (1 - weight) * w1[cross_top] + \
                weight * w2[cross_top]
            z2[cross_top] = z_top[cross_top]

        # If within layer, update intergrals
        in_layer = (z1 >= z_bot) & (z2 <= z_top)
        if np.any(in_layer):
            s_int[in_layer] += 0.5 * (w1[in_layer] + w2[in_layer]) * \
                0.5 * (s1[in_layer] + s2[in_layer]) * \
                    (z2[in_layer] - z1[in_layer])
            w_int[in_layer] += 0.5 * (w1[in_layer] + w2[in_layer]) * \
                (z2[in_layer] - z1[in_layer])

    # Compute layer mean
    s_mean = s_int / w_int

    if len(s_mean) == 1:
        return s_mean.item()
    else:
        return s_mean


def layer_mean_vector(z, u, v, z_bot, z_top, z_sfc=None, u_sfc=None,
                      v_sfc=None, vertical_axis=0, level_weights=None,
                      surface_weight=None):
    """
    Computes the mean vector components between two specified height levels,
    with optional weighting.

    Args:
        z (ndarray): height profile(s) (m)
        u (ndarray): profile(s) of eastward component of vector
        v (ndarray): profile(s) of northward component of vector
        z_bot (float or ndarray): height of bottom of layer (m)
        z_top (float or ndarray): height of top of layer (m)
        z_sfc (float or ndarray, optional): surface height (m)
        u_sfc (float or ndarray, optional): eastward component of vector at
            surface
        v_sfc (float or ndarray, optional): northward component of vector at
            surface
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)
        level_weights (ndarray, optional): weights to apply to vector at each
            level (default is None, in which case no weighting is applied)
        surface_weight (float or ndarray, optional): weight to apply to vector
            at surface (default is None, in which case no weighting is applied)

    Returns:
        u_mean (float or ndarray): eastward component of layer-mean vector (m/s)
        v_mean (float or ndarray): northward component of layer-mean vector (m/s)

    """

    if level_weights is None:
        w = np.ones_like(z)
    else:
        w = level_weights

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        z = np.moveaxis(z, vertical_axis, 0)
        u = np.moveaxis(u, vertical_axis, 0)
        v = np.moveaxis(v, vertical_axis, 0)
        w = np.moveaxis(w, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if z.ndim == 1:
        z = np.atleast_2d(z).T  # transpose to preserve vertical axis
        u = np.atleast_2d(u).T
        v = np.atleast_2d(v).T
        w = np.atleast_2d(w).T

    # If surface fields are not provided, use lowest level
    if u_sfc is None:
        bottom = 'lowest level'
        k_start = 1  # start loop from second level
        z_sfc = z[0]
        u_sfc = u[0]
        v_sfc = v[0]
        w_sfc = w[0]
    else:
        bottom = 'surface'
        k_start = 0  # start loop from first level
        if z_sfc is None:
            z_sfc = np.zeros_like(u_sfc)  # assumes height AGL
        if surface_weight is None:
            w_sfc = np.ones_like(u_sfc)

    # Make sure that surface fields are at least 1D
    z_sfc = np.atleast_1d(z_sfc)
    u_sfc = np.atleast_1d(u_sfc)
    v_sfc = np.atleast_1d(v_sfc)
    w_sfc = np.atleast_1d(w_sfc)

    # Make sure that z_bot and z_top match shape of surface arrays
    if np.isscalar(z_bot):
        z_bot = np.full_like(z_sfc, z_bot)
    if np.isscalar(z_top):
        z_top = np.full_like(z_sfc, z_top)

    # Check if bottom of layer is above top of layer
    if np.any(z_bot > z_top):
        n_pts = np.count_nonzero(z_bot > z_top)
        print(f'WARNING: z_bot is above z_top for {n_pts} points')

    # Check if bottom of layer is below surface
    if np.any(z_bot < z_sfc):
        n_pts = np.count_nonzero(z_bot < z_sfc)
        print(f'WARNING: z_bot is below {bottom} for {n_pts} points')

    # Check if top of layer is above highest level
    if np.any(z_top > z[-1]):
        n_pts = np.count_nonzero(z_top > z[-1])
        print(f'WARNING: z_top is above highest level for {n_pts} points')

    # Note the number of vertical levels
    n_lev = z.shape[0]

    # Initialise level 2 fields
    z2 = z_sfc.copy()
    u2 = u_sfc.copy()
    v2 = v_sfc.copy()
    w2 = w_sfc.copy()

    # Initialise intergrals
    u_int = np.zeros_like(z2)
    v_int = np.zeros_like(z2)
    w_int = np.zeros_like(z2)

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 fields
        z1 = z2.copy()
        u1 = u2.copy()
        v1 = v2.copy()
        w1 = w2.copy()
        if np.all(z1 >= z_top):
            # can break out of loop
            break

        # Update level 2 fields
        above_sfc = (z[k] > z_sfc)
        z2 = np.where(above_sfc, z[k], z_sfc)
        u2 = np.where(above_sfc, u[k], u_sfc)
        v2 = np.where(above_sfc, v[k], v_sfc)
        w2 = np.where(above_sfc, w[k], w_sfc)
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
            w1[cross_bot] = (1 - weight) * w1[cross_bot] + \
                weight * w2[cross_bot]
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
            w2[cross_top] = (1 - weight) * w1[cross_top] + \
                weight * w2[cross_top]
            z2[cross_top] = z_top[cross_top]

        # If within layer, update intergrals
        in_layer = (z1 >= z_bot) & (z2 <= z_top)
        if np.any(in_layer):
            u_int[in_layer] += 0.5 * (w1[in_layer] + w2[in_layer]) * \
                0.5 * (u1[in_layer] + u2[in_layer]) * \
                (z2[in_layer] - z1[in_layer])
            v_int[in_layer] += 0.5 * (w1[in_layer] + w2[in_layer]) * \
                0.5 * (v1[in_layer] + v2[in_layer]) * \
                (z2[in_layer] - z1[in_layer])
            w_int[in_layer] += 0.5 * (w1[in_layer] + w2[in_layer]) * \
                (z2[in_layer] - z1[in_layer])

    # Compute layer mean vector components
    u_mean = u_int / w_int
    v_mean = v_int / w_int

    if len(u_mean) == 1:
        return u_mean.item(), v_mean.item()
    else:
        return u_mean, v_mean
