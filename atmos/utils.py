import numpy as np
from atmos.constant import g, Rd
from atmos.thermo import virtual_temperature


def interp_pressure_level_to_height(p, z, pi, p_sfc=None, z_sfc=None,
                                    vertical_axis=0):
    """
    Interpolates a pressure level to height, assuming that height varies
    linearly with log(p).

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
        k_start = 1  # start loop from second level
        p_sfc = p[0]
        z_sfc = z[0]
    else:
        k_start = 0  # start loop from first level
        if z_sfc is None:
            z_sfc = np.zeros_like(p_sfc)  # assumes height AGL

    # Make sure that surface pressure and height are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    z_sfc = np.atleast_1d(z_sfc)

    # Note the number of vertical levels
    n_lev = p.shape[0]

    # Check if pi is below the surface
    if np.any(pi > p_sfc):
        n_pts = np.count_nonzero(pi > p_sfc)
        print(f'WARNING: pi is below surface for {n_pts} points')

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

        if np.all(p[k] > p_sfc):
            # can skip this level
            continue

        if np.all(p[k-1] <= pi):
            # can break out of loop
            break

        # Update level 1 pressure and height
        p1 = p2.copy()
        z1 = z2.copy()

        # Set level 2 pressure and height
        above_sfc = (p[k] < p_sfc)
        p2 = np.where(above_sfc, p[k], p_sfc)
        z2 = np.where(above_sfc, z[k], z_sfc)

        # Interpolate to get height of pi
        crossed = (p1 > pi) & (p2 <= pi)
        if np.any(crossed):
            weight = np.log(p1[crossed] / pi[crossed]) / \
                np.log(p1[crossed] / p2[crossed])
            zi[crossed] = (1 - weight) * z1[crossed] + \
                weight * z2[crossed]

    # Deal with points where pi is at the surface
    pi_at_sfc = (pi == p_sfc)
    zi[pi_at_sfc] = z_sfc[pi_at_sfc]

    if len(zi) == 1:
        return zi.item()
    else:
        return zi


def interp_height_level_to_pressure(z, p, zi, z_sfc=None, p_sfc=None,
                                    vertical_axis=0):
    """
    Interpolates a height level to pressure, assuming that p varies
    exponentially with height.

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
        k_start = 1  # start loop from second level
        p_sfc = p[0]
        z_sfc = z[0]
    else:
        k_start = 0  # start loop from first level
        if z_sfc is None:
            z_sfc = np.zeros_like(p_sfc)  # assumes height AGL

    # Make sure that surface pressure and height are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    z_sfc = np.atleast_1d(z_sfc)

    # Note the number of vertical levels
    n_lev = p.shape[0]

    # Check if zi is below surface
    if np.any(zi < 0):
        n_pts = np.count_nonzero(zi < 0)
        print(f'WARNING: zi is below surface for {n_pts} points')

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

        if np.all(z[k] < zi):
            # can skip this level
            continue

        if np.all(z[k-1] >= zi):
            # can break out of loop
            break

        # Update level 1 height and pressure
        z1 = z2.copy()
        p1 = p2.copy()

        # Set level 2 height and pressure
        above_sfc = (p[k] < p_sfc)
        z2 = np.where(above_sfc, z[k], z_sfc)
        p2 = np.where(above_sfc, p[k], p_sfc)

        # Interpolate to get height of pi
        crossed = (z1 < zi) & (z2 >= zi)
        if np.any(crossed):
            weight = (zi[crossed] - z1[crossed]) / \
                (z2[crossed] - z1[crossed])
            pi[crossed] = p1 ** (1 - weight) + \
                p2[crossed] ** weight

    # Deal with points where zi is at the surface
    zi_at_sfc = (zi == z_sfc)
    pi[zi_at_sfc] = p_sfc[zi_at_sfc]

    if len(pi) == 1:
        return pi.item()
    else:
        return pi


def interp_scalar_to_height_level(z, s, zi, z_sfc=None, s_sfc=None,
                                  vertical_axis=0):
    """
    Interpolates a scalar variable to a specified height level, assuming
    linear variation with height.

    Args:
        z (ndarray): height profile(s) (m)
        s (ndarray): scalar variable profile(s)
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
        k_start = 1  # start loop from second level
        z_sfc = z[0]
        s_sfc = s[0]
    else:
        k_start = 0  # start loop from first level
        if z_sfc is None:
            z_sfc = np.zeros_like(s_sfc)  # assumes height AGL

    # Make sure that surface fields are at least 1D
    z_sfc = np.atleast_1d(z_sfc)
    s_sfc = np.atleast_1d(s_sfc)

    # Check if zi is below the surface
    if np.any(zi < z_sfc):
        n_pts = np.count_nonzero(zi < z_sfc)
        print(f'WARNING: zi is below surface for {n_pts} points')

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

        if np.all(z[k] < zi):
            # can skip this level
            continue

        if np.all(z[k-1] >= zi):
            # can break out of loop
            break

        # Update level 1 fields
        z1 = z2.copy()
        s1 = s2.copy()

        # Set level 2 fields
        above_sfc = (z[k] > z_sfc)
        z2 = np.where(above_sfc, z[k], z_sfc)
        s2 = np.where(above_sfc, s[k], s_sfc)
        
        # Interpolate to get scalar at zi
        crossed = (z1 < zi) & (z2 >= zi)
        if np.any(crossed):
            weight = (zi[crossed] - z1[crossed]) / \
                (z2[crossed] - z1[crossed])
            si[crossed] = (1 - weight) * s1[crossed] + \
                weight * s2[crossed]

    if len(si) == 1:
        return si.item()
    else:
        return si


def interp_vector_to_height_level(z, u, v, zi, z_sfc=None, u_sfc=None,
                                  v_sfc=None, vertical_axis=0):
    """
    Interpolates vector components to a specified height level, assuming
    linear variation with height.

    Args:
        z (ndarray): height (m)
        u (ndarray): eastward component of vector (m/s)
        v (ndarray): northward component of vector (m/s)
        zi (float or ndarray): height of level (m)
        z_sfc (float or ndarray, optional): surface height (m)
        u_sfc (float or ndarray, optional): eastward component of vector at
            surface (m/s)
        v_sfc (float or ndarray, optional): northward component of vector at
            surface (m/s)
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
        k_start = 1  # start loop from second level
        z_sfc = z[0]
        u_sfc = u[0]
        v_sfc = v[0]
    else:
        k_start = 0  # start loop from first level
        if z_sfc is None:
            z_sfc = np.zeros_like(u_sfc)  # assumes height AGL

    # Make sure that surface fields are at least 1D
    z_sfc = np.atleast_1d(z_sfc)
    u_sfc = np.atleast_1d(u_sfc)
    v_sfc = np.atleast_1d(v_sfc)

    # Check if zi is below surface
    if np.any(zi < z_sfc):
        n_pts = np.count_nonzero(zi < z_sfc)
        print(f'WARNING: zi is below surface for {n_pts} points')

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

        if np.all(z[k] < zi):
            # can skip this level
            continue

        if np.all(z[k-1] >= zi):
            # can break out of loop
            break

        # Update level 1 fields
        z1 = z2.copy()
        u1 = u2.copy()
        v1 = v2.copy()

        # Set level 2 fields
        above_sfc = (z[k] > z_sfc)
        z2 = np.where(above_sfc, z[k], z_sfc)
        u2 = np.where(above_sfc, u[k], u_sfc)
        v2 = np.where(above_sfc, v[k], v_sfc)
        
        # Interpolate to get vector components at zi
        crossed = (z1 < zi) & (z2 >= zi)
        if np.any(crossed):
            weight = (zi[crossed] - z1[crossed]) / \
                (z2[crossed] - z1[crossed])
            ui[crossed] = (1 - weight) * u1[crossed] + \
                weight * u2[crossed]
            vi[crossed] = (1 - weight) * v1[crossed] + \
                weight * v2[crossed]

    if len(ui) == 1:
        return ui.item(), vi.item()
    else:
        return ui, vi


def interp_scalar_to_pressure_level(p, s, pi, p_sfc=None, s_sfc=None,
                                    vertical_axis=0):
    """
    Interpolates a scalar variable to a specified pressure level, assuming
    linear variation with log(p).

    Args:
        p (ndarray): pressure (Pa)
        s (ndarray): scalar variable
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
        k_start = 1  # start loop from second level
        p_sfc = p[0]
        s_sfc = s[0]
    else:
        k_start = 0  # start loop from first level

    # Make sure that surface fields are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    s_sfc = np.atleast_1d(s_sfc)

    # Check if pi is below surface
    if np.any(pi > p[0]):
        n_pts = np.count_nonzero(pi > p_sfc)
        print(f'WARNING: pi is below surface for {n_pts} points')

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

        if np.all(p[k] > pi):
            # can skip this level
            continue

        if np.all(p[k-1] <= pi):
            # can break out of loop
            break
        
        # Update level 1 fields
        p1 = p2.copy()
        s1 = s2.copy()

        # Set level 2 fields
        above_sfc = (p[k] < p_sfc)
        p2 = np.where(above_sfc, p[k], p_sfc)
        s2 = np.where(above_sfc, s[k], s_sfc)

        # Interpolate to get scalar at pi
        crossed = (p1 > pi) & (p2 <= pi)
        if np.any(crossed):
            weight = np.log(p1[crossed] / pi[crossed]) / \
                np.log(p1[crossed] / p2[crossed])
            si[crossed] = (1 - weight) * s1[crossed] + \
                weight * s2[crossed]

    if len(si) == 1:
        return si.item()
    else:
        return si


def interp_vector_to_pressure_level(p, u, v, pi, p_sfc=None, u_sfc=None,
                                    v_sfc=None, vertical_axis=0):
    """
    Interpolates vector components to a specified pressure level, assuming
    linear variation with log(p).

    Args:
        p (ndarray): pressure (Pa)
        u (ndarray): eastward component of vector (m/s)
        v (ndarray): northward component of vector (m/s)
        pi (float or ndarray): pressure of level (Pa)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        u_sfc (float or ndarray, optional): eastward component of vector at
            surface (m/s)
        v_sfc (float or ndarray, optional): northward component of vector at
            surface (m/s)
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
        k_start = 1  # start loop from second level
        p_sfc = p[0]
        u_sfc = u[0]
        v_sfc = v[0]
    else:
        k_start = 0  # start loop from first level

    # Make sure that surface fields are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    u_sfc = np.atleast_1d(u_sfc)
    v_sfc = np.atleast_1d(v_sfc)

    # Check if pi is below surface
    if np.any(pi > p_sfc):
        n_pts = np.count_nonzero(pi > p_sfc)
        print(f'WARNING: pi is below surface for {n_pts} points')

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

        if np.all(p[k] > pi):
            # can skip this level
            continue

        if np.all(p[k-1] <= pi):
            # can break out of loop
            break

        # Update level 1 fields
        p1 = p2.copy()
        u1 = u2.copy()
        v1 = v2.copy()

        # Set level 2 fields
        above_sfc = (p[k] < p_sfc)
        p2 = np.where(above_sfc, p[k], p_sfc)
        u2 = np.where(above_sfc, u[k], u_sfc)
        v2 = np.where(above_sfc, v[k], v_sfc)
        
        # Interpolate to get vector components at zi
        crossed = (p1 > pi) & (p2 <= pi)
        if np.any(crossed):
            weight = np.log(p1[crossed] / pi[crossed]) / \
                np.log(p1[crossed] / p2[crossed])
            ui[crossed] = (1 - weight) * u1[crossed] + \
                weight * u2[crossed]
            vi[crossed] = (1 - weight) * v1[crossed] + \
                weight * v2[crossed]

    if len(ui) == 1:
        return ui.item(), vi.item()
    else:
        return ui, vi


def height_of_temperature_level(z, T, Ti, vertical_axis=0):
    """
    Finds the lowest height corresponding to a specified temperature, assuming
    temperature varies linearly with height.

    Args:
        z (ndarray): height (m)
        T (ndarray): temperature (K)
        Ti (float): temperature level (K)
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

    # Note the number of vertical levels
    n_lev = z.shape[0]

    # Create array for height at temperature level
    zi = np.atleast_1d(np.full_like(z[0], np.nan))

    # Create boolean array to indicate whether level has been found
    found = np.zeros_like(zi).astype(bool)

    # Loop over levels
    for k in range(1, n_lev):

        if np.all(T[k] > Ti):
            # can skip this level
            continue
        
        # Interpolate to get height at Ti
        crossed = (T[k-1] > Ti) & (T[k] <= Ti) & np.logical_not(found)
        if np.any(crossed):
            weight = (T[k-1][crossed] - Ti) / \
                (T[k-1][crossed] - T[k][crossed])
            zi[crossed] = (1 - weight) * z[k-1][crossed] + \
                weight * z[k][crossed]
            found[crossed] = True
            if np.all(found):
                break

    if len(zi) == 1:
        return zi.item()
    else:
        return zi


def pressure_of_temperature_level(p, T, Ti, vertical_axis=0):
    """
    Finds the highest pressure corresponding to a specified temperature,
    assuming temperature varies linearly with log(p).

    Args:
        p (ndarray): pressure profile(s) (Pa)
        T (ndarray): temperature profile(s) (K)
        Ti (float): temperature level (K)
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

    # Note the number of vertical levels
    n_lev = p.shape[0]

    # Create array for pressure at temperature level
    pi = np.atleast_1d(np.full_like(p[0], np.nan))

    # Create boolean array to indicate whether level has been found
    found = np.zeros_like(pi).astype(bool)

    # Loop over levels
    for k in range(1, n_lev):

        if np.all(T[k] > Ti):
            # can skip this level
            continue

        # Interpolate to get pressure at Ti
        crossed = (T[k-1] > Ti) & (T[k] <= Ti) & np.logical_not(found)
        if np.any(crossed):
            weight = (T[k-1][crossed] - Ti) / \
                (T[k-1][crossed] - T[k][crossed])
            pi[crossed] = p[k-1][crossed] ** (1 - weight) + \
                p[k][crossed] ** weight
            found[crossed] = True
            if np.all(found):
                break

    if len(pi) == 1:
        return pi.item()
    else:
        return pi


def geopotential_height(p, T, q, p_sfc=None, T_sfc=None, q_sfc=None,
                        Z_sfc=None, vertical_axis=0):
    """
    Computes geopotential height from profiles of pressure, temperature, and
    specific humidity using the hypsometric equation.

    Args:
        p (ndarray): pressure profile(s) (Pa)
        T (ndarray): temperature profile(s) (K)
        q (ndarray): specific humidity profile(s) (kg/kg)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        T_sfc (float or ndarray, optional): surface temperature (K)
        q_sfc (float or ndarray, optional): surface specific humidity (kg/kg)
        Z_sfc (float or ndarray, optional): surface geopotential height (m)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)

    Returns:
        Z (float or ndarray): geopotential height (m)

    """

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        p = np.moveaxis(p, vertical_axis, 0)
        T = np.moveaxis(T, vertical_axis, 0)
        q = np.moveaxis(q, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if p.ndim == 1:
        p = np.atleast_2d(p).T  # transpose to preserve vertical axis
        T = np.atleast_2d(T).T
        q = np.atleast_2d(q).T

    # If surface-level fields not provided, use lowest level values
    if p_sfc is None:
        k_start = 1  # start loop from second level
        p_sfc = p[0]
        T_sfc = T[0]
        q_sfc = q[0]
        Z_sfc = np.zeros_like(p_sfc)  # assumes surface is at MSL
    else:
        k_start = 0  # start loop from first level

    # Make sure that surface fields are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    T_sfc = np.atleast_1d(T_sfc)
    q_sfc = np.atleast_1d(q_sfc)
    Z_sfc = np.atleast_1d(Z_sfc)

    # Note the number of vertical levels
    n_lev = p.shape[0]

    # Create array for geopotential height
    Z = np.zeros_like(p)

    # Initialise level 2 pressure and geopotential height
    p2 = p_sfc.copy()
    Z2 = Z_sfc.copy()

    # Compute level 2 virtual temperature
    Tv2 = virtual_temperature(T_sfc, q_sfc)

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 fields
        p1 = p2.copy()
        Z1 = Z2.copy()
        Tv1 = Tv2.copy()

        # Find points above the surface
        above_sfc = (p[k] <= p_sfc)
        if np.any(above_sfc):

            # Update level 2 pressure
            p2[above_sfc] = p[k][above_sfc]

            # Compute level 2 virtual temperature
            Tv2[above_sfc] = virtual_temperature(T[k][above_sfc],
                                                 q[k][above_sfc])

            # Compute level 2 geopotential height
            Z2[above_sfc] = Z1[above_sfc] + (Rd / g) * \
                0.5 * (Tv1[above_sfc] + Tv2[above_sfc]) * \
                np.log(p1[above_sfc] / p2[above_sfc])

            # Save geopotential height for this level
            Z[k][above_sfc] = Z2[above_sfc]

    return Z.squeeze()  # remove dimensions of length 1


def layer_mean_scalar(z, s, z_bot, z_top, vertical_axis=0,
                      level_weights=None):
    """
    Computes the mean of a scalar variable between two specified height levels,
    with optional weighting.

    Args:
        z (ndarray): height (m)
        s (ndarray): scalar variable
        z_bot (float or ndarray): height of bottom of layer (m)
        z_top (float or ndarray): height of top of layer (m)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)
        level_weights (ndarray, optional): weights to apply to winds at each
            level (default is None, in which case no weighting is applied)

    Returns:
        s_mean (float or ndarray): layer-mean scalar variable

    """

    if level_weights is None:
        wt = np.ones_like(z)
    else:
        wt = level_weights

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        z = np.moveaxis(z, vertical_axis, 0)
        s = np.moveaxis(s, vertical_axis, 0)
        wt = np.moveaxis(wt, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if z.ndim == 1:
        z = np.atleast_2d(z).T  # transpose to preserve vertical axis
        s = np.atleast_2d(s).T
        wt = np.atleast_2d(wt).T

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
    s2 = s[0]
    wt2 = wt[0]

    # Initialise intergrals
    s_int = np.zeros_like(z2)
    wt_int = np.zeros_like(z2)

    # Loop over levels
    for k in range(1, n_lev):

        # Update level 1 fields
        z1 = z2.copy()
        s1 = s2.copy()
        wt1 = wt2.copy()

        # Update level 2 fields
        z2 = z[k]
        s2 = s[k]
        wt2 = wt[k]

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
            s1[cross_bot] = (1 - weight) * s1[cross_bot] + \
                weight * s2[cross_bot]
            wt1[cross_bot] = (1 - weight) * wt1[cross_bot] + \
                weight * wt2[cross_bot]
            z1[cross_bot] = z_bot[cross_bot]

        # If crossing top of layer, reset level 2
        cross_top = (z1 < z_top) & (z2 > z_top)
        if np.any(cross_top):
            weight = (z_top[cross_top] - z1[cross_top]) / \
                (z2[cross_top] - z1[cross_top])
            s2[cross_top] = (1 - weight) * s1[cross_top] + \
                weight * s2[cross_top]
            wt2[cross_top] = (1 - weight) * wt1[cross_top] + \
                weight * wt2[cross_top]
            z2[cross_top] = z_top[cross_top]

        # If within layer, update intergrals
        in_layer = (z1 >= z_bot) & (z2 <= z_top)
        if np.any(in_layer):
            s_int[in_layer] += 0.5 * (wt1[in_layer] + wt2[in_layer]) * \
                0.5 * (s1[in_layer] + s2[in_layer]) * \
                    (z2[in_layer] - z1[in_layer])
            wt_int[in_layer] += 0.5 * (wt1[in_layer] + wt2[in_layer]) * \
                (z2[in_layer] - z1[in_layer])

    # Compute layer mean
    s_mean = s_int / wt_int

    if len(s_mean) == 1:
        return s_mean[0]
    else:
        return s_mean


def layer_mean_vector(z, u, v, z_bot, z_top, vertical_axis=0,
                      level_weights=None):
    """
    Computes the mean vector components between two specified height levels,
    with optional weighting.

    Args:
        z (ndarray): height (m)
        u (ndarray): eastward component of vector (m/s)
        v (ndarray): northward component of vector (m/s)
        z_bot (float or ndarray): height of bottom of layer (m)
        z_top (float or ndarray): height of top of layer (m)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)
        level_weights (ndarray, optional): weights to apply to winds at each
            level (default is None, in which case no weighting is applied)

    Returns:
        u_mean (float or ndarray): eastward component of layer-mean vector (m/s)
        v_mean (float or ndarray): northward component of layer-mean vector (m/s)

    """

    if level_weights is None:
        wt = np.ones_like(z)
    else:
        wt = level_weights

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        z = np.moveaxis(z, vertical_axis, 0)
        u = np.moveaxis(u, vertical_axis, 0)
        v = np.moveaxis(v, vertical_axis, 0)
        wt = np.moveaxis(wt, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if z.ndim == 1:
        z = np.atleast_2d(z).T  # transpose to preserve vertical axis
        u = np.atleast_2d(u).T
        v = np.atleast_2d(v).T
        wt = np.atleast_2d(wt).T

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
    wt2 = wt[0]

    # Initialise intergrals
    u_int = np.zeros_like(z2)
    v_int = np.zeros_like(z2)
    wt_int = np.zeros_like(z2)

    # Loop over levels
    for k in range(1, n_lev):

        # Update level 1 fields
        z1 = z2.copy()
        u1 = u2.copy()
        v1 = v2.copy()
        wt1 = wt2.copy()

        # Update level 2 fields
        z2 = z[k]
        u2 = u[k]
        v2 = v[k]
        wt2 = wt[k]

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
            wt1[cross_bot] = (1 - weight) * wt1[cross_bot] + \
                weight * wt2[cross_bot]
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
            wt2[cross_top] = (1 - weight) * wt1[cross_top] + \
                weight * wt2[cross_top]
            z2[cross_top] = z_top[cross_top]

        # If within layer, update intergrals
        in_layer = (z1 >= z_bot) & (z2 <= z_top)
        if np.any(in_layer):
            u_int[in_layer] += 0.5 * (wt1[in_layer] + wt2[in_layer]) * \
                0.5 * (u1[in_layer] + u2[in_layer]) * \
                (z2[in_layer] - z1[in_layer])
            v_int[in_layer] += 0.5 * (wt1[in_layer] + wt2[in_layer]) * \
                0.5 * (v1[in_layer] + v2[in_layer]) * \
                (z2[in_layer] - z1[in_layer])
            wt_int[in_layer] += 0.5 * (wt1[in_layer] + wt2[in_layer]) * \
                (z2[in_layer] - z1[in_layer])

    # Compute layer mean vector components
    u_mean = u_int / wt_int
    v_mean = v_int / wt_int

    if len(u_mean) == 1:
        return u_mean[0], v_mean[0]
    else:
        return u_mean, v_mean
