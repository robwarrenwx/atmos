import numpy as np
from atmos.constant import Rd
from atmos.thermo import (saturation_specific_humidity,
                          virtual_temperature,
                          potential_temperature,
                          lifting_condensation_level,
                          lifting_deposition_level,
                          lifting_saturation_level,
                          ice_fraction,
                          follow_dry_adiabat,
                          follow_moist_adiabat,
                          wet_bulb_potential_temperature)


def parcel_ascent(p, T, q, p_lpl, Tp_lpl, qp_lpl, k_lpl=None, p_sfc=None,
                  T_sfc=None, q_sfc=None, vertical_axis=0, output_scalars=True,
                  which_lfc='first', which_el='last',
                  count_cape_below_lcl=False, count_cin_below_lcl=True,
                  count_cin_above_lfc=True, phase='liquid', pseudo=True,
                  polynomial=True, explicit=False, dp=500.0):
    """
    Performs a parcel ascent from a specified lifted parcel level (LPL) and 
    returns the resulting convective available potential energy (CAPE) and
    convective inhibition (CIN), together with the lifting condensation level
    (LCL), level of free convection (LFC), and equilibrium level (EL). In the
    case of multiple layers of positive buoyancy, the final LFC and EL are
    selected according to 'which_lfc' and 'which_el', where
        * 'first' corresponds to the first layer of positive buoyancy
        * 'maxcape' corresponds to the layer of positive buoyancy with the
          largest CAPE
        * 'last' corresponds to the last layer of positive buoyancy
    Options also exist to count layers of positive buoyancy below the LCL
    towards CAPE and to count layers of negative buoyancy below the LCL or
    above the LFC towards CIN.

    Args:
        p (ndarray): pressure profile(s) (Pa)
        T (ndarray): temperature profile(s) (K)
        q (ndarray): specific humidity profile(s) (kg/kg)
        p_lpl (float or ndarray): LPL pressure (Pa)
        Tp_lpl (float or ndarray): parcel temperature at the LPL (K)
        qp_lpl (float or ndarray): parcel specific humidity at the LPL (kg/kg)
        k_lpl (int, optional): index of vertical axis corresponding to the LPL
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        T_sfc (float or ndarray, optional): surface temperature (K)
        q_sfc (float or ndarray, optional): surface specific humidity (kg/kg)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)
        output_scalars (bool, optional): flag indicating whether to convert
            output arrays to scalars if input profiles are 1D (default is True)
        which_lfc (str, optional): choice for LFC (valid options are 'first', 
            'last', or 'maxcape'; default is 'first')
        which_el (str, optional): choice for EL (valid options are 'first', 
            'last', or 'maxcape'; default is 'last')
        count_cape_below_lcl (bool, optional): flag indicating whether to
            include positive areas below the LCL in CAPE (default is False)
        count_cin_below_lcl (bool, optional): flag indicating whether to 
            include negative areas below the LCL in CIN (default is True)
        count_cin_above_lfc (bool, optional): flag indicating whether to 
            include negative areas above the LFC in CIN (default is True)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        pseudo (bool, optional): flag indicating whether to perform
            pseudoadiabatic parcel ascent (default is True)
        polynomial (bool, optional): flag indicating whether to use polynomial
            fits to pseudoadiabats (default is True)
        explicit (bool, optional): flag indicating whether to use explicit
            integration of lapse rate equation (default is False)
        dp (float, optional): pressure increment for integration of lapse rate
            equation (default is 500 Pa = 5 hPa)

    Returns:
        CAPE (float or ndarray): convective available potential energy (J/kg)
        CIN (float or ndarray): convective inhibition (J/kg)
        LCL (float or ndarray): lifting condensation level (Pa)
        LFC (float or ndarray): level of free convection (Pa)
        EL (float or ndarray): equilibrium level (Pa)

    """

    # Check that LFC and EL options are valid
    if which_lfc not in ['first', 'last', 'maxcape']:
        raise ValueError(
            "which_lfc must be one of 'first', 'last', or 'maxcape'"
        )
    if which_el not in ['first', 'last', 'maxcape']:
        raise ValueError(
            "which_el must be one of 'first', 'last', or 'maxcape'"
        )

    # Check that LFC and EL options are compatable
    if (which_lfc == 'maxcape' and which_el == 'first'):
        raise ValueError(
            """Incompatible selection for which_lfc and which_el. For 
            which_lfc='maxcape', which_el must be set as either 'maxcape'."""
        )
    if (which_lfc == 'last' and which_el != 'last'):
        raise ValueError(
            """Incompatible selection for which_lfc and which_el. For 
            which_lfc='last', which_el must also be set as 'last'."""
        )

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

    # Make sure that LPL quantities are at least 1D
    p_lpl = np.atleast_1d(p_lpl)
    Tp_lpl = np.atleast_1d(Tp_lpl)
    qp_lpl = np.atleast_1d(qp_lpl)

    # If surface-level fields not provided, use lowest level values
    if p_sfc is None:
        bottom = 'lowest level'
        k_start = 1  # start loop from second level
        p_sfc = p[0]
        T_sfc = T[0]
        q_sfc = q[0]
    else:
        bottom = 'surface'
        k_start = 0  # start loop from first level

    # Make sure that surface fields are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    T_sfc = np.atleast_1d(T_sfc)
    q_sfc = np.atleast_1d(q_sfc)

    # Ensure that all thermodynamic variables have the same type
    p = p.astype(p_lpl.dtype)
    T = T.astype(p_lpl.dtype)
    q = q.astype(p_lpl.dtype)
    Tp_lpl = Tp_lpl.astype(p_lpl.dtype)
    qp_lpl = qp_lpl.astype(p_lpl.dtype)
    p_sfc = p_sfc.astype(p_lpl.dtype)
    T_sfc = T_sfc.astype(p_lpl.dtype)
    q_sfc = q_sfc.astype(p_lpl.dtype)

    # Check that array dimensions are compatible
    if p.shape != T.shape or T.shape != q.shape:
        raise ValueError(f"""Incompatible profile arrays: 
                         {p.shape}, {T.shape}, {q.shape}""")
    if p_lpl.shape != Tp_lpl.shape or Tp_lpl.shape != qp_lpl.shape:
        raise ValueError(f"""Incompatible LPL arrays: 
                         {p_lpl.shape}, {Tp_lpl.shape}, {qp_lpl.shape}""")
    if p_sfc.shape != T_sfc.shape or T_sfc.shape != q_sfc.shape:
        raise ValueError(f"""Incompatible surface arrays: 
                         {p_sfc.shape}, {T_sfc.shape}, {q_sfc.shape}""")
    if p[0].shape != p_lpl.shape:
        raise ValueError(f"""Incompatible profile and LPL arrays: 
                         {p.shape}, {p_lpl.shape}""")
    if p[0].shape != p_sfc.shape:
        raise ValueError(f"""Incompatible profile and surface arrays: 
                         {p.shape}, {p_sfc.shape}""")

    # Check that LPL is not below the surface
    lpl_below_sfc = (p_lpl > p_sfc)
    if np.any(lpl_below_sfc):
        n_pts = np.count_nonzero(lpl_below_sfc)
        raise ValueError(f'LPL below {bottom} at {n_pts} points')
    
    # Check that LPL is not above top level
    lpl_above_top = (p_lpl < p[-1])
    if np.any(lpl_above_top):
        n_pts = np.count_nonzero(lpl_above_top)
        raise ValueError(f'LPL above top level at {n_pts} points')

    # Note the number of levels
    n_lev = p.shape[0]

    # Compute the LCL/LDL/LSL pressure and parcel temperature (hereafter, we
    # refer to all of these as the LCL for simplicity)
    if phase == 'liquid':
        p_lcl, Tp_lcl = lifting_condensation_level(p_lpl, Tp_lpl, qp_lpl)
    elif phase == 'ice':
        p_lcl, Tp_lcl = lifting_deposition_level(p_lpl, Tp_lpl, qp_lpl)
    elif phase == 'mixed':
        p_lcl, Tp_lcl = lifting_saturation_level(p_lpl, Tp_lpl, qp_lpl)
    else:
        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    # Ensure that type of LCL variables matches that of input fields
    p_lcl = p_lcl.astype(p_lpl.dtype)
    Tp_lcl = Tp_lcl.astype(p_lpl.dtype)

    # Save LCL
    LCL = p_lcl

    # Set the total water mass fraction
    if pseudo:
        # not required for pseudoadiabatic ascent
        qt = None
    else:
        # equal to initial specific humidity
        qt = qp_lpl

    # Check that LCL is below top level
    lcl_above_top = (p_lcl < p[-1])
    if np.any(lcl_above_top):
        n_pts = np.count_nonzero(lcl_above_top)
        print(f'WARNING: LCL above top level for {n_pts} points')
        k_max = n_lev-1  # stop loop a level early
    else:
        k_max = n_lev

    # Create arrays for negative and positive areas
    neg_area = np.zeros_like(p_lpl)
    pos_area = np.zeros_like(p_lpl)

    # Create arrays for temporary CAPE, CIN, LFC, and EL values
    cape_layer = np.zeros_like(p_lpl)  # CAPE for most recent positive area
    cape_total = np.zeros_like(p_lpl)  # total CAPE across all positive areas
    cape_max = np.zeros_like(p_lpl)    # CAPE for largest positive area
    cin_total = np.zeros_like(p_lpl)   # total CIN across all negative areas
    lfc = np.zeros_like(p_lpl)         # LFC for most recent positive area
    el = np.zeros_like(p_lpl)          # EL for most recent positive area

    # Create arrays for final CAPE, CIN, LFC, and EL values
    CAPE = np.zeros_like(p_lpl)
    CIN = np.full_like(p_lpl, np.nan)  # undefined where CAPE = 0
    LFC = np.full_like(p_lpl, np.nan)  # undefined where CAPE = 0
    EL = np.full_like(p_lpl, np.nan)   # undefined where CAPE = 0

    # Initialise level 2 environmental fields
    if k_lpl is None:
        p2 = p_sfc.copy()
        T2 = T_sfc.copy()
        q2 = q_sfc.copy()
    else:
        k_start = k_lpl + 1
        p2 = p[k_lpl].copy()
        T2 = T[k_lpl].copy()
        q2 = q[k_lpl].copy()

    # Initialise level 2 parcel properties using LPL values
    Tp2 = Tp_lpl.copy()
    qp2 = qp_lpl.copy()

    # Initialise parcel buoyancy (virtual temperature excess) at level 2
    B2 = virtual_temperature(Tp2, qp2) - virtual_temperature(T2, q2)

    #print(p_lpl, Tp_lpl, qp_lpl)
    #print(p_lcl, Tp_lcl, qp_lcl)
    #print(Tp2, qp2, B2)

    # Loop over levels, accounting for addition of extra level for LCL
    for k in range(k_start, k_max+1):

        # Update level 1 fields
        p1 = p2.copy()
        T1 = T2.copy()
        q1 = q2.copy()
        Tp1 = Tp2.copy()
        qp1 = qp2.copy()
        B1 = B2.copy()

        # If at or above 100 hPa with negative buoyancy, break out of loop
        if np.all(p1 <= 10000.0) and np.all(B1 < 0.0):
            break

        # Find points above and below the LPL
        above_lpl = (p1 <= p_lpl)
        below_lpl = np.logical_not(above_lpl)

        # Find points above and below the LCL
        above_lcl = (p1 <= p_lcl)
        below_lcl = np.logical_not(above_lcl)

        # Set level 2 environmental fields
        if np.any(below_lcl):
            # use level k below LCL
            p2[below_lcl] = p[k][below_lcl]
            T2[below_lcl] = T[k][below_lcl]
            q2[below_lcl] = q[k][below_lcl]
        if np.any(above_lcl):
            # use level k-1 above LCL to account for additional level
            p2[above_lcl] = p[k-1][above_lcl]
            T2[above_lcl] = T[k-1][above_lcl]
            q2[above_lcl] = q[k-1][above_lcl]

        # Set level 2 environmental fields
        # (use level k-1 above LCL to account for additional level)
        #p2 = np.where(above_lcl, p[k-1], p[k])
        #T2 = np.where(above_lcl, T[k-1], T[k])
        #q2 = np.where(above_lcl, q[k-1], q[k])

        # Reset level 2 environmental fields to surface values where
        # level 2 is below the surface
        below_sfc = (p2 > p_sfc)
        p2 = np.where(below_sfc, p_sfc, p2)
        T2 = np.where(below_sfc, T_sfc, T2)
        q2 = np.where(below_sfc, q_sfc, q2)

        # If all points are below the surface, skip this level
        if np.all(below_sfc):
            continue

        # If all points are below the LPL, skip this level
        if np.all(below_lpl):
            continue

        # If crossing the LPL, reset level 1 as the LPL
        cross_lpl = (p1 > p_lpl) & (p2 < p_lpl)
        if np.any(cross_lpl):

            #print('Crossing LPL', p1, p2, p_lpl)

            # Interpolate to get environmental temperature and specific
            # humidity at the LPL
            weight = np.log(p_lpl[cross_lpl] / p1[cross_lpl]) / \
                np.log(p2[cross_lpl] / p1[cross_lpl])
            T1[cross_lpl] = (1 - weight) * T1[cross_lpl] + \
                weight * T2[cross_lpl]
            q1[cross_lpl] = (1 - weight) * q1[cross_lpl] + \
                weight * q2[cross_lpl]
                        
            # Use LPL pressure
            p1[cross_lpl] = p_lpl[cross_lpl]

            # Update masks for points above and below the LPL
            above_lpl[cross_lpl] = True
            below_lpl[cross_lpl] = False

            # Update buoyancy
            B1[cross_lpl] = virtual_temperature(
                Tp1[cross_lpl], qp1[cross_lpl]
                ) - virtual_temperature(
                T1[cross_lpl], q1[cross_lpl]
                )

        # If crossing the LCL, reset level 2 as the LCL
        cross_lcl = (p1 > p_lcl) & (p2 < p_lcl)
        if np.any(cross_lcl):

            #print('Crossing LCL', p1, p2, p_lcl)

            # Interpolate to get environmental temperature and specific 
            # humidity at the LCL
            weight = np.log(p_lcl[cross_lcl] / p1[cross_lcl]) / \
                np.log(p2[cross_lcl] / p1[cross_lcl])
            T2[cross_lcl] = (1 - weight) * T1[cross_lcl] + \
                weight * T2[cross_lcl]
            q2[cross_lcl] = (1 - weight) * q1[cross_lcl] + \
                weight * q2[cross_lcl]

            # Use LCL pressure
            p2[cross_lcl] = p_lcl[cross_lcl]

        # Set parcel temperature and specific humidity at level 2
        if np.any(above_lpl & below_lcl):

            # Follow a dry adiabat to get parcel temperature
            Tp2[above_lpl & below_lcl] = follow_dry_adiabat(
                p1[above_lpl & below_lcl], p2[above_lpl & below_lcl],
                Tp1[above_lpl & below_lcl], qp1[above_lpl & below_lcl]
                )
 
            # Specific humidity is conserved
            qp2[above_lpl & below_lcl] = qp1[above_lpl & below_lcl]

        if np.any(above_lcl):  # LCL is always at or above the LPL

            # Parcel temperature follows moist adiabat above the LCL
            if pseudo:

                # Follow a pseudoadiabat to get parcel temperature
                Tp2[above_lcl] = follow_moist_adiabat(
                    p1[above_lcl], p2[above_lcl], Tp1[above_lcl],
                    phase=phase, pseudo=True,
                    polynomial=polynomial, explicit=explicit, dp=dp
                    )

                # Specific humidity is equal to its value at saturation
                omega = ice_fraction(Tp2[above_lcl])
                qp2[above_lcl] = saturation_specific_humidity(
                    p2[above_lcl], Tp2[above_lcl],
                    phase=phase, omega=omega
                    )

            else:

                # Follow a saturated adiabat to get parcel temperature
                Tp2[above_lcl] = follow_moist_adiabat(
                    p1[above_lcl], p2[above_lcl], Tp1[above_lcl],
                    qt=qt[above_lcl], phase=phase, pseudo=False,
                    polynomial=polynomial, explicit=explicit, dp=dp
                    )

                # Specific humidity is equal to its value at saturation
                omega = ice_fraction(Tp2[above_lcl])
                qp2[above_lcl] = saturation_specific_humidity(
                    p2[above_lcl], Tp2[above_lcl], qt=qt[above_lcl],
                    phase=phase, omega=omega
                    )

        # Compute parcel buoyancy at level 2
        B2 = virtual_temperature(Tp2, qp2, qt=qt) - virtual_temperature(T2, q2)

        # Initialise mask indicating where positive area is complete
        done = np.zeros_like(p2).astype(bool)

        # Find points where parcel is within negative area
        neg_to_neg = (B1 <= 0.0) & (B2 <= 0.0)
        if np.any(neg_to_neg):

            #print('In negative area', p1, p2, B1, B2)

            # Update the negative area
            neg_area[neg_to_neg] -= Rd * 0.5 * \
                (B1[neg_to_neg] + B2[neg_to_neg]) * \
                    np.log(p1[neg_to_neg] / p2[neg_to_neg])

        # Find points where parcel is crossing from negative to positive area
        neg_to_pos = (B1 <= 0.0) & (B2 > 0.0)
        if np.any(neg_to_pos):

            #print('Crossing LFC', p1, p2, B1, B2)

            # Interpolate to get pressure at crossing level
            px = np.zeros_like(p2)
            weight = B2[neg_to_pos] / (B2[neg_to_pos] - B1[neg_to_pos])
            px[neg_to_pos] = p1[neg_to_pos] ** (1 - weight) * \
                p2[neg_to_pos] ** weight

            # Update negative and positive areas
            neg_area[neg_to_pos] -= Rd * 0.5 * B1[neg_to_pos] * \
                np.log(p1[neg_to_pos] / px[neg_to_pos])
            pos_area[neg_to_pos] += Rd * 0.5 * B2[neg_to_pos] * \
                np.log(px[neg_to_pos] / p2[neg_to_pos])
            
            # Update total CIN
            if count_cin_below_lcl:
                # update if above LPL
                cin_total[neg_to_pos & above_lpl] += neg_area[neg_to_pos & above_lpl]
            else:
                # update if above LCL
                cin_total[neg_to_pos & above_lcl] += neg_area[neg_to_pos & above_lcl]

            # Set LFC if above LCL
            lfc[neg_to_pos & above_lcl] = px[neg_to_pos & above_lcl]

            # Reset the negative area to zero
            neg_area[neg_to_pos] = 0.0

        # Find where parcel is within positive area
        pos_to_pos = (B1 > 0.0) & (B2 > 0.0)
        if np.any(pos_to_pos):

            #print('In positive area', p1, p2, B1, B2)

            # Update the positive area
            pos_area[pos_to_pos] += Rd * 0.5 * \
                (B1[pos_to_pos] + B2[pos_to_pos]) * \
                    np.log(p1[pos_to_pos] / p2[pos_to_pos])

        # Find points where parcel is crossing from positive to negative area
        pos_to_neg = (B1 > 0.0) & (B2 <= 0.0)
        if np.any(pos_to_neg):

            #print('Crossing EL', p1, p2, B1, B2)

            # Interpolate to get pressure at crossing level
            px = np.zeros_like(p2)
            weight = B2[pos_to_neg] / (B2[pos_to_neg] - B1[pos_to_neg])
            px[pos_to_neg] = p1[pos_to_neg] ** (1 - weight) * \
                p2[pos_to_neg] ** weight

            # Update positive and negative areas
            pos_area[pos_to_neg] += Rd * 0.5 * B1[pos_to_neg] * \
                np.log(p1[pos_to_neg] / px[pos_to_neg])
            neg_area[pos_to_neg] -= Rd * 0.5 * B2[pos_to_neg] * \
                np.log(px[pos_to_neg] / p2[pos_to_neg])
            
            # Update layer and total CAPE, set EL, and mark positive area as
            # complete
            if count_cape_below_lcl:
                # update if above LPL
                cape_layer[pos_to_neg & above_lpl] = pos_area[pos_to_neg & above_lpl]
                cape_total[pos_to_neg & above_lpl] += pos_area[pos_to_neg & above_lpl]
                el[pos_to_neg & above_lpl] = px[pos_to_neg & above_lpl]
                done[pos_to_neg & above_lpl] = True
            else:
                # update if above LCL
                cape_layer[pos_to_neg & above_lcl] = pos_area[pos_to_neg & above_lcl]
                cape_total[pos_to_neg & above_lcl] += pos_area[pos_to_neg & above_lcl]
                el[pos_to_neg & above_lcl] = px[pos_to_neg & above_lcl]
                done[pos_to_neg & above_lcl] = True

            # Reset the positive area to zero
            pos_area[pos_to_neg] = 0.0

        # Reset negative areas that shouldn't be counted
        if count_cin_below_lcl:
            neg_area[below_lpl] = 0.0
        else:
            neg_area[below_lcl] = 0.0

        # Reset positive areas that shouldn't be counted
        if count_cape_below_lcl:
            pos_area[below_lpl] = 0.0
        else:
            pos_area[below_lcl] = 0.0

        # If positively buoyant at LCL then set LFC = LCL
        # (use level 1 so that this also works where LCL = LPL)
        pos_at_lcl = (p1 == p_lcl) & (B1 > 0.0)
        if np.any(pos_at_lcl):
            lfc[pos_at_lcl] = p_lcl[pos_at_lcl]

        # If positively buoyant at top level then set as EL, update layer and
        # total CAPE, and set positive area as complete
        pos_at_top = (p2 == p[-1]) & (B2 > 0.0)
        if np.any(pos_at_top):
            n_pts = np.count_nonzero(pos_at_top)
            print(f'WARNING: Positive buoyancy at top level for {n_pts} points')
            el[pos_at_top] = p2[pos_at_top]
            cape_layer[pos_at_top] = pos_area[pos_at_top]
            cape_total[pos_at_top] += pos_area[pos_at_top]
            done[pos_at_top] = True

        #print(k, p2, T2, q2, Tp2, qp2, qt)
        #print(k, p1, p2, B1, B2)
        #print(k, pos_area, neg_area, cape_layer, cape_total, cape_max, cin_total, above_lpl, above_lcl)

        if np.any(done):

            # Update maximum CAPE
            is_max = (cape_layer > cape_max)
            if np.any(is_max):
                cape_max[done & is_max] = cape_layer[done & is_max]

            # Set final LFC, EL, CAPE, and CIN based on which_lfc and which_el
            if which_lfc == 'first':

                # Use LFC for first positive area
                is_first = (CAPE == 0.0)
                LFC[done & is_first] = lfc[done & is_first]

                if which_el == 'first':

                    # Use EL for first positive area
                    EL[done & is_first] = el[done & is_first]

                    # Use total CAPE up to first positive area
                    CAPE[done & is_first] = cape_total[done & is_first]

                    # Use total CIN up to largest positive area
                    CIN[done & is_first] = cin_total[done & is_first]

                elif which_el == 'maxcape':

                    # Use EL for largest positive area
                    EL[done & is_max] = el[done & is_max]

                    # Use total CAPE up to largest positive area
                    CAPE[done & is_max] = cape_total[done & is_max]

                    if count_cin_above_lfc:

                        # Use total CIN up to largest positive area
                        CIN[done & is_max] = cin_total[done & is_max]

                    else:

                        # Use total CIN up to first positive area
                        CIN[done & is_first] = cin_total[done & is_first]

                else:

                    # Use EL for last positive area
                    EL[done] = el[done]

                    # Use total CAPE up to last positive area
                    CAPE[done] = cape_total[done]

                    if count_cin_above_lfc:

                        # Use CIN up to last positive area
                        CIN[done] = cin_total[done]

                    else:

                        # Use total CIN up to first positive area
                        CIN[done & is_first] = cin_total[done & is_first]

            elif which_lfc == 'maxcape':

                # Use LFC for largest positive area
                LFC[done & is_max] = lfc[done & is_max]

                if which_el == 'maxcape':

                    # Use EL for largest positive area
                    EL[done & is_max] = el[done & is_max]

                    # Use CAPE for largest positive area
                    CAPE[done & is_max] = cape_layer[done & is_max]

                    # Use total CIN up to largest positive area
                    CIN[done & is_max] = cin_total[done & is_max]

                else:

                    # Use EL for last positive area
                    EL[done] = el[done]

                    # Use total CAPE from largest positive area upwards
                    CAPE[done & is_max] = 0.0  # reset if max-CAPE layer
                    CAPE[done] += cape_layer[done]

                    if count_cin_above_lfc:

                        # Use CIN up to last positive area
                        CIN[done] = cin_total[done]

                    else:

                        # Use total CIN up to largest positive area
                        CIN[done & is_max] = cin_total[done & is_max]

            else:

                # Use LFC for last positive area
                LFC[done] = lfc[done]

                # Use total CIN up to last positive area
                CIN[done] = cin_total[done]

                # Use EL for last positive area
                EL[done] = el[done]

                # Use CAPE for last positive area
                CAPE[done] = cape_layer[done]

    if len(CAPE) == 1 and output_scalars:
        # convert outputs to scalars
        CAPE = CAPE.item()
        CIN = CIN.item()
        LCL = LCL.item()
        LFC = LFC.item()
        EL = EL.item()

    return CAPE, CIN, LCL, LFC, EL


def surface_based_parcel_ascent(p, T, q, p_sfc=None, T_sfc=None, q_sfc=None, 
                                vertical_axis=0, **kwargs):
    """
    Performs a surface-based (SB) parcel ascent and returns the resulting
    convective available potential energy (CAPE) and convective inhibition
    (CIN), along with the lifting condensation level (LCL), level of free
    convection (LFC), and equilibrium level (EL).

    Args:
        p (ndarray): pressure profile(s) (Pa)
        T (ndarray): temperature profile(s) (K)
        q (ndarray): specific humidity profile(s) (kg/kg)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        T_sfc (float or ndarray, optional): surface temperature (K)
        q_sfc (float or ndarray, optional): surface specific humidity (kg/kg)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)
        **kwargs: additional keyword arguments passed to parcel_ascent

    Returns:
        CAPE (float or ndarray): convective available potential energy (J/kg)
        CIN (float or ndarray): convective inhibition (J/kg)
        LCL (float or ndarray): lifting condensation level (Pa)
        LFC (float or ndarray): level of free convection (Pa)
        EL (float or ndarray): equilibrium level (Pa)

    """

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        p = np.moveaxis(p, vertical_axis, 0)
        T = np.moveaxis(T, vertical_axis, 0)
        q = np.moveaxis(q, vertical_axis, 0)

    # Set LPL and associated parcel properties
    if p_sfc is None:
        p_lpl = p[0]
        Tp_lpl = T[0]
        qp_lpl = q[0]
    else:
        p_lpl = p_sfc
        Tp_lpl = T_sfc
        qp_lpl = q_sfc

    # Call code to perform parcel ascent
    CAPE, CIN, LCL, LFC, EL = parcel_ascent(
        p, T, q, p_lpl, Tp_lpl, qp_lpl,
        p_sfc=p_sfc, T_sfc=T_sfc, q_sfc=q_sfc,
        **kwargs
    )
    
    return CAPE, CIN, LCL, LFC, EL


def mixed_layer_parcel(p, T, q, p_sfc=None, T_sfc=None, q_sfc=None,
                       mixed_layer_depth=5000.0, vertical_axis=0,
                       output_scalars=False):
    """
    Computes the mixed-layer (ML) parcel temperature and specific humidity.
    ML parcel temperature is computed by averaging potential temperature and
    converting to temperature using the surface or lowest level pressure. Note
    mass weighting is implicit in the averaging due to the use of pressure as
    the vertical coordinate.

    Args:
        p (ndarray): pressure profile(s) (Pa)
        T (ndarray): temperature profile(s) (K)
        q (ndarray): specific humidity profile(s) (kg/kg)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        T_sfc (float or ndarray, optional): surface temperature (K)
        q_sfc (float or ndarray, optional): surface specific humidity (kg/kg)
        mixed_layer_depth (float, optional): mixed-layer depth (Pa) (default is
            5000 Pa = 50 hPa)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)
        output_scalars (bool, optional): flag indicating whether to convert
            single-element output arrays to scalars (default is False)

    Returns:
        Tp_ml (float or ndarray): mixed-layer parcel temperature (K)
        qp_ml (float or ndarray): mixed-layer parcel specific humidity (kg/kg)

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
    else:
        k_start = 0  # start loop from first level

    # Make sure that surface fields are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    T_sfc = np.atleast_1d(T_sfc)
    q_sfc = np.atleast_1d(q_sfc)

    # Note the number of levels
    n_lev = p.shape[0]

    # Determine the pressure at the ML top
    p_mlt = p_sfc - mixed_layer_depth

    # Check that the ML top is not above the top level
    mlt_above_top = (p_mlt < p[-1])
    if np.any(mlt_above_top):
        n_pts = np.count_nonzero(mlt_above_top)
        raise ValueError(f'Mixed-layer top above top level at {n_pts} points')

    # Create arrays to store potential temperature and specific humidity
    # of ML parcel
    thp_ml = np.zeros_like(p_sfc)
    qp_ml = np.zeros_like(p_sfc)

    # Initialise level 2 fields at surface
    p2 = p_sfc.copy()
    T2 = T_sfc.copy()
    q2 = q_sfc.copy()

    # Compute potential temperature at level 2
    th2 = potential_temperature(p2, T2, q2)

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 fields
        p1 = p2
        T1 = T2
        q1 = q2
        th1 = th2

        # Find level 2 points above the surface
        above_sfc = (p[k] < p_sfc)
        below_sfc = np.logical_not(above_sfc)
        if np.all(below_sfc):
            # if all points are below the surface, skip this level
            continue

        # Find level 1 points at or above the ML top
        above_mlt = (p1 <= p_mlt)
        below_mlt = np.logical_not(above_mlt)
        if np.all(above_mlt):
            # if all points are above the ML top, break out of loop
            break

        # Update level 2 fields
        p2 = np.where(above_sfc, p[k], p_sfc)
        T2 = np.where(above_sfc, T[k], T_sfc)
        q2 = np.where(above_sfc, q[k], q_sfc)

        # If crossing the ML top, reset level 2 as the ML top
        cross_mlt = (p1 > p_mlt) & (p2 < p_mlt)
        if np.any(cross_mlt):

            # Interpolate to get environmental temperature and specific 
            # humidity at the ML top
            weight = np.log(p_mlt[cross_mlt] / p1[cross_mlt]) / \
                np.log(p2[cross_mlt] / p1[cross_mlt])
            T2[cross_mlt] = (1 - weight) * T1[cross_mlt] + \
                weight * T2[cross_mlt]
            q2[cross_mlt] = (1 - weight) * q1[cross_mlt] + \
                weight * q2[cross_mlt]
            
            # Use ML top pressure
            p2[cross_mlt] = p_mlt[cross_mlt]

        # Compute the potential temperature at level 2
        th2 = potential_temperature(p2, T2, q2)

        # Update the ML averages
        thp_ml[above_sfc & below_mlt] += 0.5 * \
            (th1[above_sfc & below_mlt] + th2[above_sfc & below_mlt]) * \
            (p1[above_sfc & below_mlt] - p2[above_sfc & below_mlt])
        qp_ml[above_sfc & below_mlt] += 0.5 * \
            (q1[above_sfc & below_mlt] + q2[above_sfc & below_mlt]) * \
            (p1[above_sfc & below_mlt] - p2[above_sfc & below_mlt])
        
        #print(p1, T1, q1, th1, p2, T2, q2, th2, th_avg, q_avg)

    # Divide averages by ML depth to get final values
    thp_ml = thp_ml / mixed_layer_depth
    qp_ml = qp_ml / mixed_layer_depth

    # Compute corresponding temperature at the surface
    Tp_ml = follow_dry_adiabat(100000., p_sfc, thp_ml, qp_ml)

    #print(p_sfc, thp_ml, Tp_ml, qp_ml)

    if len(p_sfc) == 1 and output_scalars:
        Tp_ml = Tp_ml.item()
        qp_ml = qp_ml.item()

    return Tp_ml, qp_ml


def mixed_layer_parcel_ascent(p, T, q, p_sfc=None, T_sfc=None, q_sfc=None,
                              mixed_layer_depth=5000.0, vertical_axis=0,
                              return_parcel_properties=False, **kwargs):
    """
    Performs a mixed-layer (ML) parcel ascent and returns the resulting
    convective available potential energy (CAPE) and convective inhibition
    (CIN), along with the lifting condensation level (LCL), level of free
    convection (LFC), and equilibrium level (EL). The ML parcel is defined by
    the mass-weighted average potential temperature and specific humidity,
    computed over a specified layer depth, together with the surface pressure.

    Args:
        p (ndarray): pressure profile(s) (Pa)
        T (ndarray): temperature profile(s) (K)
        q (ndarray): specific humidity profile(s) (kg/kg)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        T_sfc (float or ndarray, optional): surface temperature (K)
        q_sfc (float or ndarray, optional): surface specific humidity (kg/kg)
        mixed_layer_depth (float, optional): mixed-layer depth (Pa) (default is
            5000 Pa = 50 hPa)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)
        return_parcel_properties (bool, optional): flag indicating whether to
            return parcel temperature and specific humidity (default is False)
        **kwargs: additional keyword arguments passed to parcel_ascent

    Returns:
        CAPE (float or ndarray): convective available potential energy (J/kg)
        CIN (float or ndarray): convective inhibition (J/kg)
        LCL (float or ndarray): lifting condensation level (Pa)
        LFC (float or ndarray): level of free convection (Pa)
        EL (float or ndarray): equilibrium level (Pa)

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

    # Get the LPL parcel temperature and specific humidity
    Tp_lpl, qp_lpl = mixed_layer_parcel(
        p, T, q, p_sfc=p_sfc, T_sfc=T_sfc, q_sfc=q_sfc,
        mixed_layer_depth=mixed_layer_depth,
    )

    # Set the LPL pressure
    if p_sfc is None:
        p_lpl = p[0]
    else:
        p_lpl = p_sfc

    # Call code to perform parcel ascent
    CAPE, CIN, LCL, LFC, EL = parcel_ascent(
        p, T, q, p_lpl, Tp_lpl, qp_lpl,
        p_sfc=p_sfc, T_sfc=T_sfc, q_sfc=q_sfc,
        **kwargs
    )

    if return_parcel_properties:
        return CAPE, CIN, LCL, LFC, EL, Tp_lpl, qp_lpl
    else:
        return CAPE, CIN, LCL, LFC, EL


def most_unstable_parcel(p, T, q, p_sfc=None, T_sfc=None, q_sfc=None,
                         min_pressure=50000.0, vertical_axis=0,
                         phase='liquid', polynomial=True,
                         output_scalars=False):
    """
    Finds the most-unstable (MU) parcel, defined as the parcel with maximum
    wet-bulb potential temperature up to some minimum pressure, and returns
    the corresponding pressure, temperature, and specific humidity.

    Args:
        p (ndarray): pressure profile(s) (Pa)
        T (ndarray): temperature profile(s) (K)
        q (ndarray): specific humidity profile(s) (kg/kg)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        T_sfc (float or ndarray, optional): surface temperature (K)
        q_sfc (float or ndarray, optional): surface specific humidity (kg/kg)
        min_pressure (float, optional): minimum pressure from which to launch
            parcel (Pa) (default is 50000 Pa = 500 hPa)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        polynomial (bool, optional): flag indicating whether to use polynomial
            fits to pseudoadiabats (default is True)
        output_scalars (bool, optional): flag indicating whether to convert
            single-element output arrays to scalars (default is False)

    Returns:
        p_mu (ndarray or float): MU parcel pressure (Pa)
        Tp_mu (ndarray or float): MU parcel temperature (K)
        qp_mu (ndarray or float): MU parcel specific humidity (K)

    """

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        p = np.moveaxis(p, vertical_axis, 0)
        T = np.moveaxis(T, vertical_axis, 0)
        q = np.moveaxis(q, vertical_axis, 0)

    # If surface-level fields not provided, use lowest level values
    if p_sfc is None:
        k_start = 1  # start loop from second level
        p_sfc = p[0]
        T_sfc = T[0]
        q_sfc = q[0]
    else:
        k_start = 0  # start loop from first level

    # Make sure that profile arrays are at least 2D
    if p.ndim == 1:
        p = np.atleast_2d(p).T  # transpose to preserve vertical axis
        T = np.atleast_2d(T).T
        q = np.atleast_2d(q).T

    # Make sure that surface fields are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    T_sfc = np.atleast_1d(T_sfc)
    q_sfc = np.atleast_1d(q_sfc)

    # Note the number of levels
    n_lev = p.shape[0]

    # Compute WBPT at the surface
    thw_sfc = wet_bulb_potential_temperature(p_sfc, T_sfc, q_sfc,
                                             phase=phase,
                                             polynomial=polynomial)
    thw_sfc = np.atleast_1d(thw_sfc)

    # Initialise the maximum WBPT
    thw_max = thw_sfc

    # Initialise the MU parcel fields using surface values
    p_mu = p_sfc.copy()
    Tp_mu = T_sfc.copy()
    qp_mu = q_sfc.copy()

    #print('sfc', p_mu, Tp_mu, qp_mu, thw_max)

    # Loop over levels
    for k in range(k_start, n_lev):

        # Find points below the surface
        below_sfc = (p[k] > p_sfc)
        if np.all(below_sfc):
            # if all points are below the surface, skip this level
            continue

        # Find points above the minimum pressure level
        above_min = (p[k] < min_pressure)
        if np.all(above_min):
            # if all points are above the minimum pressure level, break
            # out of loop
            break

        # Compute WBPT
        thw = wet_bulb_potential_temperature(p[k], T[k], q[k],
                                             phase=phase,
                                             polynomial=polynomial)
        thw = np.atleast_1d(thw)

        # For points below the surface or above the minimum pressure level,
        # replace WBPT with the value at the surface
        thw[above_min | below_sfc] = thw_sfc[above_min | below_sfc]

        # Update MU parcel properties and maximum WBPT
        is_max = (thw > thw_max)
        p_mu[is_max] = p[k][is_max]
        Tp_mu[is_max] = T[k][is_max]
        qp_mu[is_max] = q[k][is_max]
        thw_max[is_max] = thw[is_max]

        #print(k, p_mu, Tp_mu, qp_mu, thw, thw_max)

    #print(p_lpl, Tp_lpl, qp_lpl)

    if len(p_mu) == 1 and output_scalars:
        p_mu = p_mu.item()
        Tp_mu = Tp_mu.item()
        qp_mu = qp_mu.item()

    return p_mu, Tp_mu, qp_mu


def most_unstable_parcel_ascent(p, T, q, p_sfc=None, T_sfc=None, q_sfc=None,
                                mu_parcel='max_wbpt', min_pressure=50000.0,
                                eil_min_cape=100.0, eil_max_cin=250.0,
                                vertical_axis=0,
                                return_parcel_properties=False, **kwargs):
    """
    Performs a most-unstable (MU) parcel ascent and returns the resulting
    convective available potential energy (CAPE) and convective inhibition
    (CIN), along with the lifted parcel level (LPL), lifting condensation level
    (LCL), level of free convection (LFC), and equilibrium level (EL). By
    default, the MU parcel is defined using the maximum wet-bulb potential
    temperature (WBPT; thw) up to some minimum pressure. Alternatively, it can
    be defined by launching parcels from every level and identifying the one
    with maximum CAPE. In this case, the function also returns the effective
    inflow layer (EIL) base and top. The EIL is defined as the first layer
    comprising at least two levels where CAPE >= 100 J/kg and CIN <= 250 J/kg.
    Note, however, that this calculation tends to be much slower.

    Args:
        p (ndarray): pressure profile(s) (Pa)
        T (ndarray): temperature profile(s) (K)
        q (ndarray): specific humidity profile(s) (kg/kg)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        T_sfc (float or ndarray, optional): surface temperature (K)
        q_sfc (float or ndarray, optional): surface specific humidity (kg/kg)
        mu_parcel (str, optional): method for defining the most unstable parcel
            (valid options are 'max_wbpt' or 'max_cape'; default is 'max_wbpt')
        min_pressure (float, optional): minimum pressure from which to launch
            parcel (Pa) (default is 50000 Pa = 500 hPa)
        eil_min_cape (float, optional): minimum CAPE threshold used to define
            the EIL (J/kg) (default is 100 J/kg)
        eil_max_cin (float, optional): maximum CIN threshold used to define
            the EIL (J/kg) (default is 250 J/kg)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)
        return_parcel_properties (bool, optional): flag indicating whether to
            return parcel temperature and specific humidity (default is False)
        **kwargs: additional keyword arguments passed to parcel_ascent

    Returns:
        CAPE (float or ndarray): convective available potential energy (J/kg)
        CIN (float or ndarray): convective inhibition (J/kg)
        LPL (float or ndarray): lifted parcel level (Pa)
        LCL (float or ndarray): lifting condensation level (Pa)
        LFC (float or ndarray): level of free convection (Pa)
        EL (float or ndarray): equilibrium level (Pa)
        EILbase (float or ndarray): effective inflow layer base (Pa)
        EILtop (float or ndarray): effective inflow layer top (Pa)

    """

    # Check that MU parcel option is valid
    if mu_parcel not in ['max_wbpt', 'max_cape']:
        raise ValueError("mu_parcel must be either 'max_wbpt' or 'max_cape'")

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

    # Note the number of levels
    n_lev = p.shape[0]

    if mu_parcel == 'max_cape':

        # Set initial LPL and associated parcel properties
        if p_sfc is None:
            k_start = 1  # start loop from second level
            p_lpl = p[0].copy()
            Tp_lpl = T[0].copy()
            qp_lpl = q[0].copy()
        else:
            k_start = 0  # start loop from first level
            p_lpl = p_sfc.copy()
            Tp_lpl = T_sfc.copy()
            qp_lpl = q_sfc.copy()

        # Make sure that LPL fields are at least 1D
        p_lpl = np.atleast_1d(p_lpl)
        Tp_lpl = np.atleast_1d(Tp_lpl)
        qp_lpl = np.atleast_1d(qp_lpl)

        # Perform parcel ascent from surface
        cape, cin, lcl, lfc, el = parcel_ascent(
            p, T, q, p_lpl, Tp_lpl, qp_lpl,
            p_sfc=p_sfc, T_sfc=T_sfc, q_sfc=q_sfc,
            output_scalars=False, **kwargs
        )
    
        # Reset values where the LPL is above the minimum pressure level
        lpl_above_min = (p_lpl < min_pressure)
        cape[lpl_above_min] = 0.0
        cin[lpl_above_min] = np.nan
        lcl[lpl_above_min] = np.nan
        lfc[lpl_above_min] = np.nan
        el[lpl_above_min] = np.nan

        # If surface-level fields not provided, use lowest level values
        if p_sfc is None:
            k_start = 1  # start loop from second level
            p_sfc = p[0]
            T_sfc = T[0]
            q_sfc = q[0]
        else:
            k_start = 0  # start loop from first level

        # Make sure that surface fields are at least 1D
        p_sfc = np.atleast_1d(p_sfc)
        T_sfc = np.atleast_1d(T_sfc)
        q_sfc = np.atleast_1d(q_sfc)

        #print('sfc', p_sfc, T_sfc, q_sfc, cape, cin, lcl, lfc, el)

        # Initialise final CAPE, CIN, LPL, LCL, LFC, and EL arrays
        CAPE = cape
        CIN = cin
        LPL = p_lpl
        LCL = lcl
        LFC = lfc
        EL = el

        # Initialise arrays for EIL base and top
        EILbase = np.full_like(p_sfc, np.nan)
        EILtop = np.full_like(p_sfc, np.nan)

        # Check if surface is part of EIL
        in_eil = (cape >= eil_min_cape) & (cin <= eil_max_cin)
        EILbase[in_eil] = p_sfc[in_eil]

        # Loop over levels
        for k in range(k_start, n_lev):

            # Note the pressure and EIL mask from previous level
            in_eil_prev = in_eil.copy()
            p_lpl_prev = p_lpl.copy()

            # Find points above and below the surface
            above_sfc = (p[k] <= p_sfc)
            below_sfc = np.logical_not(above_sfc)
            if np.all(below_sfc):
                # if all points are below the surface, skip this level
                continue

            # Find points above the minimum pressure level
            above_min = (p[k] < min_pressure)
            if np.all(above_min):
                # if all points are above the minimum pressure level, break
                # out of loop
                break

            # Set lifted parcel level (LPL) fields
            p_lpl = np.where(above_sfc, p[k], p_sfc)
            Tp_lpl = np.where(above_sfc, T[k], T_sfc)
            qp_lpl = np.where(above_sfc, q[k], q_sfc)

            # Perform parcel ascent from the LPL
            cape, cin, lcl, lfc, el = parcel_ascent(
                p, T, q, p_lpl, Tp_lpl, qp_lpl, k_lpl=k,
                output_scalars=False, **kwargs
            )

            # Reset values where the LPL is above the minimum pressure level
            # (this should only happen where p_sfc < min_pressure)
            lpl_above_min = (p_lpl < min_pressure)
            cape[lpl_above_min] = 0.0
            cin[lpl_above_min] = np.nan
            lcl[lpl_above_min] = np.nan
            lfc[lpl_above_min] = np.nan
            el[lpl_above_min] = np.nan

            # Update the final CAPE, CIN, LFC, and EL
            is_max = (cape > CAPE)
            CAPE[is_max] = cape[is_max]
            CIN[is_max] = cin[is_max]
            LPL[is_max] = p_lpl[is_max]
            LCL[is_max] = lcl[is_max]
            LFC[is_max] = lfc[is_max]
            EL[is_max] = el[is_max]

            #print(k, p_lpl, Tp_lpl, qp_lpl, cape, cin, lfc, el, CAPE, CIN, LCL, LFC, EL)

            # Update the EIL base and top pressures
            in_eil = (cape >= eil_min_cape) & (cin <= eil_max_cin)
            is_base = in_eil & np.isnan(EILbase)
            EILbase[is_base] = p_lpl[is_base]
            is_top = in_eil_prev & np.logical_not(in_eil) & np.isnan(EILtop)
            EILtop[is_top] = p_lpl_prev[is_top]

            # Reset EIL base and top where layer comprises only a single level
            base_eq_top = (EILbase == EILtop)
            EILbase[base_eq_top] = np.nan
            EILtop[base_eq_top] = np.nan
            
        if len(CAPE) == 1:
            # convert outputs to scalars
            CAPE = CAPE.item()
            CIN = CIN.item()
            LPL = LPL.item()
            LCL = LCL.item()
            LFC = LFC.item()
            EL = EL.item()
            EILbase = EILbase.item()
            EILtop = EILtop.item()

        if return_parcel_properties:
            return (CAPE, CIN, LPL, LCL, LFC, EL, EILbase, EILtop,
                    Tp_lpl, qp_lpl)
        else:
            return CAPE, CIN, LPL, LCL, LFC, EL, EILbase, EILtop

    else:

        # Get phase and polynomial flag from kwargs
        phase = kwargs.get('phase', 'liquid')
        polynomial = kwargs.get('polynomial', True)

        # Find the pressure, parcel temperature, and parcel specific humidity
        # corresponding to the MU LPL
        p_lpl, Tp_lpl, qp_lpl = most_unstable_parcel(
            p, T, q, p_sfc=p_sfc,
            T_sfc=T_sfc, q_sfc=q_sfc,
            min_pressure=min_pressure,
            phase=phase, polynomial=polynomial
        )

        # Note the LPL
        LPL = p_lpl

        # Perform parcel ascent from the LPL
        CAPE, CIN, LCL, LFC, EL = parcel_ascent(
            p, T, q, p_lpl, Tp_lpl, qp_lpl,
            p_sfc=p_sfc, T_sfc=T_sfc, q_sfc=q_sfc,
            output_scalars=False, **kwargs
        )

        # Reset values where the LPL is above the minimum pressure level
        # (this should only happen if p_sfc < min_pressure)
        above_min = (p_lpl < min_pressure)
        CAPE[above_min] = 0.0
        CIN[above_min] = np.nan
        LPL[above_min] = np.nan
        LCL[above_min] = np.nan
        LFC[above_min] = np.nan
        EL[above_min] = np.nan

        if len(CAPE) == 1:
            # convert outputs to scalars
            CAPE = CAPE.item()
            CIN = CIN.item()
            LPL = LPL.item()
            LCL = LCL.item()
            LFC = LFC.item()
            EL = EL.item()
            Tp_lpl = Tp_lpl.item()
            qp_lpl = qp_lpl.item()

        if return_parcel_properties:
            return CAPE, CIN, LPL, LCL, LFC, EL, Tp_lpl, qp_lpl
        else:
            return CAPE, CIN, LPL, LCL, LFC, EL


def effective_parcel(p, T, q, p_sfc=None, T_sfc=None, q_sfc=None,
                     p_eib=None, p_eit=None, vertical_axis=0,
                     output_scalars=False):
    """
    Computes the effective (EFF) parcel temperature and specific humidity.
    EFF parcel temperature is computed by averaging potential temperature
    over the effective inflow layer (EIL) and converting to temperature using
    the pressure at the mid-point of EIL. Note mass weighting is implicit in
    the averaging due to the use of pressure as the vertical coordinate. If
    the EIL is not supplied, they are derived by calling the most-unstable
    parcel function.

    Args:
        p (ndarray): pressure profile(s) (Pa)
        T (ndarray): temperature profile(s) (K)
        q (ndarray): specific humidity profile(s) (kg/kg)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        T_sfc (float or ndarray, optional): surface temperature (K)
        q_sfc (float or ndarray, optional): surface specific humidity (kg/kg)
        p_eib (float or ndarray, optional): effective inflow base pressure (Pa)
        p_eit (float or ndarray, optional): effective inflow top pressure (Pa)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)
        output_scalars (bool, optional): flag indicating whether to convert
            single-element output arrays to scalars (default is False)

    Returns:
        p_mid (float or ndarray): pressure at mid-point of EIL (Pa)
        Tp_eff (float or ndarray): effective parcel temperature (K)
        qp_eff (float or ndarray): effective parcel specific humidity (kg/kg)

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

    if p_eib is None or p_eit is None:

        # Get pressure at the base and top of the EIL
        _, _, _, _, _, _, p_eib, p_eit = most_unstable_parcel_ascent(
            p, T, q, p_sfc=p_sfc, T_sfc=T_sfc, q_sfc=q_sfc, 
            mu_parcel='max_cape'
        )

    # If surface-level fields not provided, use lowest level values
    if p_sfc is None:
        k_start = 1  # start loop from second level
        p_sfc = p[0]
        T_sfc = T[0]
        q_sfc = q[0]
    else:
        k_start = 0  # start loop from first level

    # Make sure that surface fields are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    T_sfc = np.atleast_1d(T_sfc)
    q_sfc = np.atleast_1d(q_sfc)

    # Make sure that EIL base and top are at least 1D
    p_eib = np.atleast_1d(p_eib)
    p_eit = np.atleast_1d(p_eit)

    # Note the number of levels
    n_lev = p.shape[0]

    # Check that the EIL base is not below the surface
    eib_below_sfc = (p_eib > p_sfc)
    if np.any(eib_below_sfc):
        n_pts = np.count_nonzero(eib_below_sfc)
        raise ValueError(f'Effective inflow base below surface at {n_pts} points')

    # Check that the EIL top is not above the top level
    eit_above_top = (p_eit < p[-1])
    if np.any(eit_above_top):
        n_pts = np.count_nonzero(eit_above_top)
        raise ValueError(f'Effective inflow top above top level at {n_pts} points')

    # Note the pressure at the mid-point of the EIL
    p_mid = 0.5 * (p_eib + p_eit)

    # Create arrays to store potential temperature and specific humidity
    # of EFF parcel
    thp_eff = np.zeros_like(p_sfc)
    qp_eff = np.zeros_like(p_sfc)

    # Initialise level 2 fields at surface
    p2 = p_sfc.copy()
    T2 = T_sfc.copy()
    q2 = q_sfc.copy()

    # Compute potential temperature at level 2
    th2 = potential_temperature(p2, T2, q2)

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 fields
        p1 = p2
        T1 = T2
        q1 = q2
        th1 = th2

        # Find level 2 points above the surface
        above_sfc = (p[k] < p_sfc)
        below_sfc = np.logical_not(above_sfc)
        if np.all(below_sfc):
            # if all points are below the surface, skip this level
            continue

        # Find level 2 points above the EIL base
        above_eib = (p[k] < p_eib)
        below_eib = np.logical_not(above_eib)
        if np.all(below_eib):
            # if all points are below the EIL base, skip this level
            continue

        # Find level 1 points at or above the EIL top
        above_eit = (p1 <= p_eit)
        below_eit = np.logical_not(above_eit)
        if np.all(above_eit):
            # if all points are above the EIL top, break out of loop
            break

        # Update level 2 fields
        p2 = np.where(above_sfc, p[k], p_sfc)
        T2 = np.where(above_sfc, T[k], T_sfc)
        q2 = np.where(above_sfc, q[k], q_sfc)

        # If crossing the EIL base, reset level 2 as the EIL base
        cross_eib = (p1 > p_eib) & (p2 < p_eib)
        if np.any(cross_eib):

            # Interpolate to get environmental temperature and specific 
            # humidity at the EIL base
            weight = np.log(p_eib[cross_eib] / p1[cross_eib]) / \
                np.log(p2[cross_eib] / p1[cross_eib])
            T2[cross_eib] = (1 - weight) * T1[cross_eib] + \
                weight * T2[cross_eib]
            q2[cross_eib] = (1 - weight) * q1[cross_eib] + \
                weight * q2[cross_eib]

            # Use EIL base pressure
            p2[cross_eib] = p_eib[cross_eib]

        # If crossing the EIL top, reset level 2 as the EIL top
        cross_eit = (p1 > p_eit) & (p2 < p_eit)
        if np.any(cross_eit):

            # Interpolate to get environmental temperature and specific 
            # humidity at the EIL top
            weight = np.log(p_eit[cross_eit] / p1[cross_eit]) / \
                np.log(p2[cross_eit] / p1[cross_eit])
            T2[cross_eit] = (1 - weight) * T1[cross_eit] + \
                weight * T2[cross_eit]
            q2[cross_eit] = (1 - weight) * q1[cross_eit] + \
                weight * q2[cross_eit]

            # Use EIL top pressure
            p2[cross_eit] = p_eit[cross_eit]

        # Compute the potential temperature at level 2
        th2 = potential_temperature(p2, T2, q2)

        # Update the EFF parcel averages
        thp_eff[above_eib & below_eit] += 0.5 * \
            (th1[above_eib & below_eit] + th2[above_eib & below_eit]) * \
            (p1[above_eib & below_eit] - p2[above_eib & below_eit])
        qp_eff[above_eib & below_eit] += 0.5 * \
            (q1[above_eib & below_eit] + q2[above_eib & below_eit]) * \
            (p1[above_eib & below_eit] - p2[above_eib & below_eit])

        #print(p1, T1, q1, th1, p2, T2, q2, th2, thp_eff, qp_eff)

    # Divide averages by EIL depth to get final values
    thp_eff = thp_eff / (p_eib - p_eit)
    qp_eff = qp_eff / (p_eib - p_eit)

    # Compute corresponding temperature at the surface
    Tp_eff = follow_dry_adiabat(100000., p_mid, thp_eff, qp_eff)

    #print(p_mid, thp_eff, Tp_eff, qp_eff)

    if len(p_sfc) == 1 and output_scalars:
        p_mid = p_mid.item()
        Tp_eff = Tp_eff.item()
        qp_eff = qp_eff.item()

    return p_mid, Tp_eff, qp_eff


def effective_parcel_ascent(p, T, q, p_sfc=None, T_sfc=None, q_sfc=None,
                            p_eib=None, p_eit=None, vertical_axis=0,
                            return_parcel_properties=False, **kwargs):
    """
    Performs an effective (EFF) parcel ascent and returns the resulting
    convective available potential energy (CAPE) and convective inhibition
    (CIN), along with the lifting condensation level (LCL), level of free
    convection (LFC), and equilibrium level (EL). The EFF parcel is defined by
    the mass-weighted average potential temperature and specific humidity,
    computed over the effective inflow layer (EIL), together with the pressure
    at the mid-point of the EIL. If the EIL is not supplied, it is derived by
    calling the most-unstable parcel function.

    Args:
        p (ndarray): pressure profile(s) (Pa)
        T (ndarray): temperature profile(s) (K)
        q (ndarray): specific humidity profile(s) (kg/kg)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        T_sfc (float or ndarray, optional): surface temperature (K)
        q_sfc (float or ndarray, optional): surface specific humidity (kg/kg)
        p_eib (float or ndarray, optional): effective inflow base pressure (Pa)
        p_eit (float or ndarray, optional): effective inflow top pressure (Pa)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)
        return_parcel_properties (bool, optional): flag indicating whether to
            return parcel temperature and specific humidity (default is False)
        **kwargs: additional keyword arguments passed to parcel_ascent

    Returns:
        CAPE (float or ndarray): convective available potential energy (J/kg)
        CIN (float or ndarray): convective inhibition (J/kg)
        LPL (float or ndarray): lifted parcel level (Pa)
        LCL (float or ndarray): lifting condensation level (Pa)
        LFC (float or ndarray): level of free convection (Pa)
        EL (float or ndarray): equilibrium level (Pa)

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

    if p_eib is None or p_eit is None:

        # Get pressure at the base and top of the EIL
        _, _, _, _, _, _, p_eib, p_eit = most_unstable_parcel_ascent(
            p, T, q, p_sfc=p_sfc, T_sfc=T_sfc, q_sfc=q_sfc, 
            mu_parcel='max_cape'
        )

    # Get the LPL pressure, temperature, and specific humidity
    p_lpl, Tp_lpl, qp_lpl = effective_parcel(
        p, T, q, p_sfc=p_sfc, T_sfc=T_sfc, q_sfc=q_sfc,
        p_eib=p_eib, p_eit=p_eit
    )

    # Note the LPL
    LPL = p_lpl

    # Call code to perform parcel ascent
    CAPE, CIN, LCL, LFC, EL = parcel_ascent(
        p, T, q, p_lpl, Tp_lpl, qp_lpl,
        p_sfc=p_sfc, T_sfc=T_sfc, q_sfc=q_sfc,
        output_scalars=False, **kwargs
    )

    if len(CAPE) == 1:
        # convert outputs to scalars
        CAPE = CAPE[0]
        CIN = CIN[0]
        LPL = LPL[0]
        LCL = LCL[0]
        LFC = LFC[0]
        EL = EL[0]

    if return_parcel_properties:
        return CAPE, CIN, LPL, LCL, LFC, EL, Tp_lpl, qp_lpl
    else:
        return CAPE, CIN, LPL, LCL, LFC, EL


def lifted_index(pi, pf, Ti, Tf, qi, qf=None, phase='liquid',
                 polynomial=True, explicit=False, dp=500.0,
                 use_virtual_temperature=False):
    """
    Calculates the lifted index (LI), defined as the difference between the
    environmental temperature at a specified pressure level and the
    temperature a parcel lifted dry adiabatically to saturation and then
    pseudoadiabatically to the specified level. The parcel and environmental
    virtual temperatures can optionally be used instead of the temperatures.

    Args:
        pi (float or ndarray): initial pressure (Pa)
        pf (float or ndarray): final pressure (Pa)
        Ti (float or ndarray): initial temperature (K)
        Tf (float or ndarray): final temperature (K)
        qi (float or ndarray): initial specific humidity (kg/kg)
        qf (float or ndarray, optional): final specific humidity (kg/kg)
            (only required if using virtual temperature; default is None)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        polynomial (bool, optional): flag indicating whether to use polynomial
            fits to pseudoadiabats (default is True)
        explicit (bool, optional): flag indicating whether to use explicit
            integration of lapse rate equation (default is False)
        dp (float, optional): pressure increment for integration of lapse rate
            equation (default is 500 Pa = 5 hPa)
        use_virtual_temperature (bool, optional): use virtual temperature,
            instead of temperature, to compute LI (default is False)

    Returns:
        LI (float or ndarray): lifted index (K)

    """

    # Convert scalar inputs to arrays, if required
    pi = np.atleast_1d(pi)
    pf = np.atleast_1d(pf)
    Ti = np.atleast_1d(Ti)
    Tf = np.atleast_1d(Tf)
    qi = np.atleast_1d(qi)
    if qf is not None:
        qf = np.atleast_1d(qf)
    if len(Ti) > 1:
        # multiple initial temperature values
        if len(pi) == 1:
            # single initial pressure value
            pi = np.full_like(Ti, pi)
        if len(pf) == 1:
            # single final pressure value
            pf = np.full_like(Ti, pf)

    # Check that array shapes match
    if pi.shape != pf.shape:
        raise ValueError('''Initial and final pressure arrays must have
                         identical shape''',  pi.shape, pf.shape)
    if pi.shape != Ti.shape:
        raise ValueError('''Initial pressure and temperature arrays must have
                         identical shape''', pi.shape, Ti.shape)
    if Ti.shape != qi.shape:
        raise ValueError('''Initial temperature and specific humidity arrays
                         must have identical shape''', Ti.shape, qi.shape)
    if pf.shape != Tf.shape:
        raise ValueError('''Final pressure and temperature arrays must have
                         identical shape''', pf.shape, Tf.shape)
    if qf is not None:
        if Tf.shape != qf.shape:
            raise ValueError('''Final temperature and specific humidity arrays
                             must have identical shape''', Tf.shape, qf.shape)

    # Check that final pressure is less than initial pressure
    if np.any(pi < pf):
        raise ValueError('Final pressure must be less than initial pressure')

    # Compute the LCL/LDL/LSL pressure and parcel temperature (hereafter, we
    # refer to all of these as the LCL for simplicity)
    if phase == 'liquid':
        p_lcl, Tp_lcl = lifting_condensation_level(pi, Ti, qi)
    elif phase == 'ice':
        p_lcl, Tp_lcl = lifting_deposition_level(pi, Ti, qi)
    elif phase == 'mixed':
        p_lcl, Tp_lcl = lifting_saturation_level(pi, Ti, qi)
    else:
        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    # Create arrays to store the final parcel temperature and specific
    # humidity
    Tpf = np.zeros_like(pi)
    qpf = np.zeros_like(pi)

    # Find points where the final level is above the LCL
    above_lcl = (p_lcl > pf)
    if np.any(above_lcl):

        # Follow a pseudoadiabat to get parcel temperature
        Tpf[above_lcl] = follow_moist_adiabat(
            p_lcl[above_lcl], pf[above_lcl], Tp_lcl[above_lcl],
            phase=phase, pseudo=True,
            polynomial=polynomial, explicit=explicit, dp=dp
            )

        # Specific humidity is equal to its value at saturation
        omega = ice_fraction(Tf[above_lcl])
        qpf[above_lcl] = saturation_specific_humidity(
            pf[above_lcl], Tf[above_lcl],
            phase=phase, omega=omega
            )

    # Find points where the final level is below the LCL
    below_lcl = (p_lcl <= pf)
    if np.any(below_lcl):

        # Follow a dry adiabat to get parcel temperature
        Tpf[below_lcl] = follow_dry_adiabat(
            pi[below_lcl], pf[below_lcl], Ti[below_lcl], qi[below_lcl]
            )

        # Specific humidity is equal to its initial value
        qpf[below_lcl] = qi[below_lcl]

    # Computed the lifted index
    if use_virtual_temperature:
        if qf is None:
            raise ValueError('''Final specific humidity must be specified if
                             using virtual temperature''')
        LI = virtual_temperature(Tf, qf) - virtual_temperature(Tpf, qpf)
    else:
        LI = Tf - Tpf

    # Mask points where final pressure exceeds initial pressure
    LI[pf > pi] = np.nan

    if len(LI) == 1:
        return LI.item()
    else:
        return LI
