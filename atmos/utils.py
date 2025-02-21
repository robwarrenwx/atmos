import numpy as np


def interp_scalar_to_height_level(z, s, zi, vertical_axis=0):
    """
    Interpolates scalar variable to a specified height level, assuming
    linear variation with height.

    Args:
        z (ndarray): height (m)
        s (ndarray): scalar variable
        zi (float or ndarray): height of level (m)
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

    # Make sure that zi is least 1D
    zi = np.atleast_1d(zi)

    # Note the number of vertical levels
    n_lev = z.shape[0]

    # Check if zi is below lowest level
    if np.any(zi < z[0]):
        n_pts = np.count_nonzero(zi < z[0])
        print(f'WARNING: zi is below lowest level for {n_pts} points')

    # Check if zi is above highest level
    if np.any(zi > z[-1]):
        n_pts = np.count_nonzero(zi > z[-1])
        print(f'WARNING: zi is above highest level for {n_pts} points')

    # Initialise level 2 fields
    z2 = z[0]
    s2 = s[0]

    # Initialise scalar at zi
    si = s2.copy()

    # Loop over levels
    for k in range(1, n_lev):

        # Update level 1 fields
        z1 = z2.copy()
        s1 = s2.copy()

        # Update level 2 fields
        z2 = z[k]
        s2 = s[k]

        if np.all(z2 < zi):
            # can skip this level
            continue

        if np.all(z1 >= zi):
            # can break out of loop
            break
        
        # Interpolate to get scalar at zi
        crossed = (z1 < zi) & (z2 >= zi)
        if np.any(crossed):
            weight = (zi[crossed] - z1[crossed]) / \
                (z2[crossed] - z1[crossed])
            si[crossed] = (1 - weight) * s1[crossed] + \
                weight * s2[crossed]

    if len(si) == 1:
        return si[0]
    else:
        return si


def interp_vector_to_height_level(z, u, v, zi, vertical_axis=0):
    """
    Interpolates vector components to a specified height level, assuming
    linear variation with height.

    Args:
        z (ndarray): height (m)
        u (ndarray): eastward component of vector (m/s)
        v (ndarray): northward component of vector (m/s)
        zi (float or ndarray): height of level (m)
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

    # Make sure that zi is least 1D
    zi = np.atleast_1d(zi)

    # Note the number of vertical levels
    n_lev = z.shape[0]

    # Check if zi is below lowest level
    if np.any(zi < z[0]):
        n_pts = np.count_nonzero(zi < z[0])
        print(f'WARNING: zi is below lowest level for {n_pts} points')

    # Check if zi is above highest level
    if np.any(zi > z[-1]):
        n_pts = np.count_nonzero(zi > z[-1])
        print(f'WARNING: zi is above highest level for {n_pts} points')

    # Initialise level 2 fields
    z2 = z[0]
    u2 = u[0]
    v2 = v[0]

    # Initialise vector components at zi
    ui = u2.copy()
    vi = v2.copy()

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

        if np.all(z2 < zi):
            # can skip this level
            continue

        if np.all(z1 >= zi):
            # can break out of loop
            break
        
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
        return ui[0], vi[0]
    else:
        return ui, vi


def interp_scalar_to_pressure_level(p, s, pi, vertical_axis=0):
    """
    Interpolates scalar variable to a specified pressure level, assuming
    linear variation with log(p).

    Args:
        p (ndarray): pressure (Pa)
        s (ndarray): scalar variable
        pi (float or ndarray): pressure of level (Pa)
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

    # Make sure that pi is least 1D
    pi = np.atleast_1d(pi)

    # Note the number of vertical levels
    n_lev = p.shape[0]

    # Check if pi is below lowest level
    if np.any(pi > p[0]):
        n_pts = np.count_nonzero(pi > p[0])
        print(f'WARNING: pi is below lowest level for {n_pts} points')

    # Check if pi is above highest level
    if np.any(pi < p[-1]):
        n_pts = np.count_nonzero(pi < p[-1])
        print(f'WARNING: pi is above highest level for {n_pts} points')

    # Initialise level 2 fields
    p2 = p[0]
    s2 = s[0]

    # Initialise scalar at zi
    si = s2.copy()

    # Loop over levels
    for k in range(1, n_lev):

        # Update level 1 fields
        p1 = p2.copy()
        s1 = s2.copy()

        # Update level 2 fields
        p2 = p[k]
        s2 = s[k]

        if np.all(p2 > pi):
            # can skip this level
            continue

        if np.all(p1 <= pi):
            # can break out of loop
            break
        
        # Interpolate to get scalar at zi
        crossed = (p1 > pi) & (p2 <= pi)
        if np.any(crossed):
            weight = np.log(p1[crossed] / pi[crossed]) / \
                np.log(p1[crossed] / p2[crossed])
            si[crossed] = (1 - weight) * s1[crossed] + \
                weight * s2[crossed]

    if len(si) == 1:
        return si[0]
    else:
        return si


def interp_vector_to_pressure_level(p, u, v, pi, vertical_axis=0):
    """
    Interpolates vector components to a specified pressure level, assuming
    linear variation with log(p).

    Args:
        p (ndarray): pressure (Pa)
        u (ndarray): eastward component of vector (m/s)
        v (ndarray): northward component of vector (m/s)
        pi (float or ndarray): pressure of level (Pa)
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

    # Make sure that pi is least 1D
    pi = np.atleast_1d(pi)

    # Note the number of vertical levels
    n_lev = p.shape[0]

    # Check if pi is below lowest level
    if np.any(pi > p[0]):
        n_pts = np.count_nonzero(pi > p[0])
        print(f'WARNING: pi is below lowest level for {n_pts} points')

    # Check if pi is above highest level
    if np.any(pi < p[-1]):
        n_pts = np.count_nonzero(pi < p[-1])
        print(f'WARNING: pi is above highest level for {n_pts} points')

    # Initialise level 2 fields
    p2 = p[0]
    u2 = u[0]
    v2 = v[0]

    # Initialise vector components at zi
    ui = u2.copy()
    vi = v2.copy()

    # Loop over levels
    for k in range(1, n_lev):

        # Update level 1 fields
        p1 = p2.copy()
        u1 = u2.copy()
        v1 = v2.copy()

        # Update level 2 fields
        p2 = p[k]
        u2 = u[k]
        v2 = v[k]

        if np.all(p2 > pi):
            # can skip this level
            continue

        if np.all(p1 <= pi):
            # can break out of loop
            break
        
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
        return ui[0], vi[0]
    else:
        return ui, vi


def layer_mean_scalar(z, s, z_bot, z_top, vertical_axis=0,
                      level_weights=None):
    """
    Computes mean o scalar variable between two specified height levels, with
    optional weighting.

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
    Computes mean vector components between two specified height levels, with
    optional weighting.

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
