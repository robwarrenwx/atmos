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


def parcel_ascent(p, T, q, p_lpl, Tp_lpl, qp_lpl, k_lpl=0, vertical_axis=0,
                  output_scalars=True, which_lfc='first', which_el='last',
                  count_cape_below_lcl=False, count_cin_below_lcl=True,
                  count_cin_above_lfc=True, phase='liquid', pseudo=True,
                  polynomial=True, explicit=False, dp=500.0,
                  return_profiles=False):
    """
    Performs a parcel ascent from a specified lifted parcel level (LPL) and 
    returns the resulting convective available potential energy (CAPE) and
    convective inhibition (CIN), together with the lifting condensation level
    (LCL), level of free convection (LFC), and equilibrium level (EL). In the
    case of multiple layers of positive buoyancy, the final LFC and EL are
    selected according to 'which_lfc' and 'which_el', where
        * 'first' corresponds to the LFC/EL of the first layer of positive
          buoyancy above the LCL
        * 'maxcape' corresponds to the LFC/EL of the layer of positive buoyancy
          with the largest CAPE
        * 'last' corresponds to the LFC/EL of the last layer of positive
          buoyancy encountered
    Options also exist to count layers of positive buoyancy below the LCL
    towards CAPE and to count layers of negative buoyancy below the LCL or
    above the LFC towards CIN.

    Args:
        p (ndarray): pressure profile(s) (Pa)
        T (ndarray): temperature profile(s) (K)
        q (ndarray): specific humidity profile(s) (kg/kg)
        p_lpl (float or ndarray): lifted parcel level (LPL) pressure (Pa)
        Tp_lpl (float or ndarray): parcel temperature at the LPL (K)
        qp_lpl (float or ndarray): parcel specific humidity at the LPL (kg/kg)
        k_lpl (int, optional): index of vertical axis corresponding to the LPL
            (default is 0)
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
        pseudo (bool): flag indicating whether to perform pseudoadiabatic
            parcel ascent (default is True)
        polynomial (bool, optional): flag indicating whether to use polynomial
            fits to pseudoadiabats (default is True)
        explicit (bool, optional): flag indicating whether to use explicit
            integration of lapse rate equation (default is False)
        dp (float, optional): pressure increment for integration of lapse rate
            equation (default is 500 Pa = 5 hPa)
        return_profiles (bool, optional): flag indicating whether to return
            profiles of pressure, parcel temperature, and parcel specific
            humidity (default is False)

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

    # Make sure that LPL quantities are at least 1D
    p_lpl = np.atleast_1d(p_lpl)
    Tp_lpl = np.atleast_1d(Tp_lpl)
    qp_lpl = np.atleast_1d(qp_lpl)

    # Ensure that all quantities have the same type
    p = p.astype(p_lpl.dtype)
    T = T.astype(p_lpl.dtype)
    q = q.astype(p_lpl.dtype)
    Tp_lpl = Tp_lpl.astype(p_lpl.dtype)
    qp_lpl = qp_lpl.astype(p_lpl.dtype)

    # Check that array dimensions are compatible
    if p.shape != T.shape or T.shape != q.shape:
        raise ValueError(f"""Incompatible profile arrays: {p.shape},
                         {T.shape}, {q.shape}""")
    if p_lpl.shape != Tp_lpl.shape or Tp_lpl.shape != qp_lpl.shape:
        raise ValueError(f"""Incompatible LPL arrays: {p_lpl.shape},
                         {Tp_lpl.shape}, {qp_lpl.shape}""")
    if p[0].shape != p_lpl.shape:
        raise ValueError(f"""Incompatible profile and LPL arrays: {p.shape},
                         {p_lpl.shape}""")

    # Check that LPL is above bottom level
    lpl_below_bot = (p_lpl > p[0])
    if np.any(lpl_below_bot):
        n_pts = np.count_nonzero(lpl_below_bot)
        raise ValueError(f'LPL below bottom level at {n_pts} points')
    
    # Check that LPL is below top level
    lpl_above_top = (p_lpl < p[-1])
    if np.any(lpl_above_top):
        n_pts = np.count_nonzero(lpl_above_top)
        raise ValueError(f'LPL above top level at {n_pts} points')

    # Check that LFC and EL options are valid
    if which_lfc not in ['first', 'last', 'maxcape']:
        raise ValueError(
            "which_lfc must be one of 'first', 'last', or 'maxcape'")
    if which_el not in ['first', 'last', 'maxcape']:
        raise ValueError(
            "which_el must be one of 'first', 'last', or 'maxcape'")
    
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

    # Note the number of levels
    n_lev = p.shape[0]

    # Compute the LCL/LDL/LSL pressure and parcel temperature (hereafter, we
    # use the name 'LCL' due to its familiarity)
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
    LCL = p_lcl  # output variable

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
    p_lfc = np.zeros_like(p_lpl)       # LFC for most recent positive area
    p_el = np.zeros_like(p_lpl)        # EL for most recent positive area

    # Create arrays for final CAPE, CIN, LFC, and EL values
    CAPE = np.zeros_like(p_lpl)
    CIN = np.full_like(p_lpl, np.nan)  # undefined where CAPE = 0
    LFC = np.full_like(p_lpl, np.nan)  # undefined where CAPE = 0
    EL = np.full_like(p_lpl, np.nan)   # undefined where CAPE = 0

    # Initialise level 2 environmental fields using LPL index
    p2 = p[k_lpl].copy()
    T2 = T[k_lpl].copy()
    q2 = q[k_lpl].copy()

    # Initialise level 2 parcel properties using LPL values
    Tp2 = Tp_lpl.copy()
    qp2 = qp_lpl.copy()

    # Compute parcel buoyancy (virtual temperature excess) at level 2
    # (note we don't need to include qt as LPL can't be above LCL)
    B2 = virtual_temperature(Tp2, qp2) - virtual_temperature(T2, q2)

    #print(p_lpl, Tp_lpl, qp_lpl)
    #print(p_lcl, Tp_lcl, qt)
    #print(Tp2, qp2, B2)

    if return_profiles:

        # Create arrays to store parcel profiles
        shape = list(p.shape)
        shape[0] = k_max+1
        shape = tuple(shape)
        pp = np.full(shape, np.nan)
        Tp = np.full(shape, np.nan)
        qp = np.full(shape, np.nan)

    # Loop over levels, accounting for addition of extra level for LCL
    for k in range(k_lpl+1, k_max+1):

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
        # (use level k-1 above LCL to account for additional level)
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

        #p2 = np.where(above_lcl, p[k-1], p[k])
        #T2 = np.where(above_lcl, T[k-1], T[k])
        #q2 = np.where(above_lcl, q[k-1], q[k])

        # If all points are below the LPL we can skip this level
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

            # Recompute parcel buoyancy (virtual temperature excess) at level 1
            # (note we don't need to include qt as LPL can't be above LCL)
            B1 = virtual_temperature(Tp1, qp1) - virtual_temperature(T1, q1)

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

        # Check that pressure is not increasing
        p2_gt_p1 = (p2 > p1)
        if np.any(p2_gt_p1):
            n_pts = np.count_nonzero(p2_gt_p1)
            raise ValueError(f'Pressure increasing at {n_pts} points')

        # Compute the log of p1 and p2
        lnp1 = np.log(p1)
        lnp2 = np.log(p2)

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

        if return_profiles:

            # Store parcel properties at this level
            pp[k] = p2
            Tp[k] = Tp2
            qp[k] = qp2

            p1_is_lpl = (p1 == p_lpl)
            if np.any(p1_is_lpl):
                pp[k-1][p1_is_lpl] = p1[p1_is_lpl]
                Tp[k-1][p1_is_lpl] = Tp1[p1_is_lpl]
                qp[k-1][p1_is_lpl] = qp1[p1_is_lpl]

        # Update parcel buoyancy (virtual temperature difference)
        B2 = virtual_temperature(Tp2, qp2, qt=qt) - virtual_temperature(T2, q2)

        # Initialise mask indicating where positive area is complete
        done = np.zeros_like(p2).astype(bool)

        # Find points where parcel is within negative area
        neg_to_neg = (B1 <= 0.0) & (B2 <= 0.0)
        if np.any(neg_to_neg):

            # Update the negative area
            neg_area[neg_to_neg] -= Rd * 0.5 * \
                (B1[neg_to_neg] + B2[neg_to_neg]) * \
                    (lnp1[neg_to_neg] - lnp2[neg_to_neg])

        # Find points where parcel is crossing from negative to positive area
        neg_to_pos = (B1 <= 0.0) & (B2 > 0.0)
        if np.any(neg_to_pos):

            #print('Crossing LFC', p1, p2, B1, B2)

            # Interpolate to get pressure at crossing level
            lnpx = np.zeros_like(p2)
            lnpx[neg_to_pos] = (B2[neg_to_pos] * lnp1[neg_to_pos] -
                                B1[neg_to_pos] * lnp2[neg_to_pos]) / \
                                    (B2[neg_to_pos] - B1[neg_to_pos])

            # Update negative and positive areas
            neg_area[neg_to_pos] -= Rd * 0.5 * B1[neg_to_pos] * \
                (lnp1[neg_to_pos] - lnpx[neg_to_pos])
            pos_area[neg_to_pos] += Rd * 0.5 * B2[neg_to_pos] * \
                (lnpx[neg_to_pos] - lnp2[neg_to_pos])
            
            # Update total CIN
            if count_cin_below_lcl:
                # update if above LPL
                cin_total[neg_to_pos & above_lpl] += neg_area[neg_to_pos & above_lpl]
            else:
                # update if above LCL
                cin_total[neg_to_pos & above_lcl] += neg_area[neg_to_pos & above_lcl]

            # Set LFC if above LCL
            p_lfc[neg_to_pos & above_lcl] = np.exp(lnpx[neg_to_pos & above_lcl])

            # Reset the negative area to zero
            neg_area[neg_to_pos] = 0.0

        # Find where parcel is within positive area
        pos_to_pos = (B1 > 0.0) & (B2 > 0.0)
        if np.any(pos_to_pos):

            # Update the positive area
            pos_area[pos_to_pos] += Rd * 0.5 * \
                (B1[pos_to_pos] + B2[pos_to_pos]) * \
                    (lnp1[pos_to_pos] - lnp2[pos_to_pos])

        # Find points where parcel is crossing from positive to negative area
        pos_to_neg = (B1 > 0.0) & (B2 <= 0.0)
        if np.any(pos_to_neg):

            #print('Crossing EL', p1, p2, B1, B2)

            # Interpolate to get pressure at crossing level
            lnpx = np.zeros_like(p2)
            lnpx[pos_to_neg] = (B2[pos_to_neg] * lnp1[pos_to_neg] -
                                B1[pos_to_neg] * lnp2[pos_to_neg]) / \
                                    (B2[pos_to_neg] - B1[pos_to_neg])
            
            # Update positive and negative areas
            pos_area[pos_to_neg] += Rd * 0.5 * B1[pos_to_neg] * \
                (lnp1[pos_to_neg] - lnpx[pos_to_neg])
            neg_area[pos_to_neg] -= Rd * 0.5 * B2[pos_to_neg] * \
                (lnpx[pos_to_neg] - lnp2[pos_to_neg])
            
            # Update layer and total CAPE, set EL, and mark positive area as
            # complete
            if count_cape_below_lcl:
                # update if above LPL
                cape_layer[pos_to_neg & above_lpl] = pos_area[pos_to_neg & above_lpl]
                cape_total[pos_to_neg & above_lpl] += pos_area[pos_to_neg & above_lpl]
                p_el[pos_to_neg & above_lpl] = np.exp(lnpx[pos_to_neg & above_lpl])
                done[pos_to_neg & above_lpl] = True
            else:
                # update if above LCL
                cape_layer[pos_to_neg & above_lcl] = pos_area[pos_to_neg & above_lcl]
                cape_total[pos_to_neg & above_lcl] += pos_area[pos_to_neg & above_lcl]
                p_el[pos_to_neg & above_lcl] = np.exp(lnpx[pos_to_neg & above_lcl])
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
        # (use p1 so that this also works where LCL = LPL)
        pos_at_lcl = (p1 == p_lcl) & (B1 > 0.0)
        if np.any(pos_at_lcl):
            p_lfc[pos_at_lcl] = p_lcl[pos_at_lcl]

        # If positively buoyant at top level then set as EL, update layer and
        # total CAPE, and set positive area as complete
        pos_at_top = (p2 == p[-1]) & (B2 > 0.0)
        if np.any(pos_at_top):
            n_pts = np.count_nonzero(pos_at_top)
            print(f'WARNING: Positive buoyancy at top level for {n_pts} points')
            p_el[pos_at_top] = p2[pos_at_top]
            cape_layer[pos_at_top] = pos_area[pos_at_top]
            cape_total[pos_at_top] += pos_area[pos_at_top]
            done[pos_at_top] = True

        #print(k, p2, T2, q2, Tp2, qp2, qt, B2)
        #print(k, pos_area, neg_area, cape_layer, cape_total, cape_max, cin_total)

        if np.any(done):

            # Update maximum CAPE
            is_max = (cape_layer > cape_max)
            if np.any(is_max):
                cape_max[done & is_max] = cape_layer[done & is_max]

            # Set final CAPE, CIN, LFC and EL based on which_lfc and which_el
            if which_lfc == 'first':

                # Use LFC for first positive area
                is_first = (CAPE == 0.0)
                LFC[done & is_first] = p_lfc[done & is_first]

                if not count_cin_above_lfc:

                    # Use CIN up to first positive area
                    CIN[done & is_first] = cin_total[done & is_first]

                if which_el == 'first':

                    # Use EL for first positive area
                    EL[done & is_first] = p_el[done & is_first]

                    # Use total CAPE up to first positive area
                    CAPE[done & is_first] = cape_total[done & is_first]

                    if count_cin_above_lfc:

                        # Use total CIN up to largest positive area
                        CIN[done & is_first] = cin_total[done & is_first]

                elif which_el == 'maxcape':

                    # Use EL for largest positive area
                    EL[done & is_max] = p_el[done & is_max]

                    # Use total CAPE up to largest positive area
                    CAPE[done & is_max] = cape_total[done & is_max]

                    if count_cin_above_lfc:

                        # Use total CIN up to largest positive area
                        CIN[done & is_max] = cin_total[done & is_max]

                else:

                    # Use EL for last positive area
                    EL[done] = p_el[done]

                    # Use total CAPE up to last positive area
                    CAPE[done] = cape_total[done]

                    if count_cin_above_lfc:

                        # Use CIN up to last positive area
                        CIN[done] = cin_total[done]

            elif which_lfc == 'maxcape':

                # Use LFC for largest positive area
                LFC[done & is_max] = p_lfc[done & is_max]

                if not count_cin_above_lfc:

                    # Use CIN up to largest positive area
                    CIN[done & is_max] = cin_total[done & is_max]

                if which_el == 'maxcape':

                    # Use EL for largest positive area
                    EL[done & is_max] = p_el[done & is_max]

                    # Use CAPE for largest positive area
                    CAPE[done & is_max] = cape_layer[done & is_max]

                    if count_cin_above_lfc:

                        # Use total CIN up to largest positive area
                        CIN[done & is_max] = cin_total[done & is_max]

                else:

                    # Use EL for last positive area
                    EL[done] = p_el[done]

                    # Use total CAPE from largest positive area upwards
                    CAPE[done & is_max] = 0.0  # reset if max-CAPE layer
                    CAPE[done] += cape_layer[done]

                    if count_cin_above_lfc:

                        # Use CIN up to last positive area
                        CIN[done] = cin_total[done]

            else:

                # Use LFC for last positive area
                LFC[done] = p_lfc[done]

                # Use total CIN up to last positive area
                CIN[done] = cin_total[done]

                # Use EL for last positive area
                EL[done] = p_el[done]

                # Use CAPE for last positive area
                CAPE[done] = cape_layer[done]

    if len(p_lpl) == 1 and output_scalars:
        # convert outputs to scalars
        CAPE = CAPE[0]
        CIN = CIN[0]
        LCL = LCL[0]
        LFC = LFC[0]
        EL = EL[0]

    if return_profiles:
        return CAPE, CIN, LCL, LFC, EL, pp, Tp, qp
    else:
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

    # If surface-level fields not provided, use lowest level values
    if p_sfc is None:
        p_sfc = p[0]
        T_sfc = T[0]
        q_sfc = q[0]

    # Call code to perform parcel ascent
    CAPE, CIN, LCL, LFC, EL = parcel_ascent(p, T, q, p_sfc, T_sfc, q_sfc,
                                            **kwargs)
    
    return CAPE, CIN, LCL, LFC, EL
    

def mixed_layer_parcel_ascent(p, T, q, p_sfc=None, T_sfc=None, q_sfc=None, 
                              mixed_layer_depth=5000.0, vertical_axis=0, 
                              return_parcel_props=False, **kwargs):
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
        return_parcel_props (bool, optional): flag indicating whether to return
            parcel temperature and specific humidity)
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

    # Determine the pressure at the ML top
    p_mlt = p_sfc - mixed_layer_depth

    # Check that the ML top is not above the top level
    mlt_above_top = (p_mlt < p[-1])
    if np.any(mlt_above_top):
        n_pts = np.count_nonzero(mlt_above_top)
        raise ValueError(f'Mixed-layer top above top level at {n_pts} points')

    # Create arrays to store potential temperature and specific humidity
    # average over the ML
    th_avg = np.zeros_like(p_sfc)
    q_avg = np.zeros_like(p_sfc)

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

        # Find points above and below the surface
        above_sfc = (p1 <= p_sfc)
        below_sfc = np.logical_not(above_sfc)
        if np.all(below_sfc):
            # if all points are below the surface we can skip this level
            continue

        # Find points above the ML top
        above_mlt = (p1 <= p_mlt)
        below_mlt = np.logical_not(above_mlt)
        if np.all(above_mlt):
            # if all points are above the ML top we can break out of loop
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
        th_avg[above_sfc & below_mlt] += 0.5 * \
            (th1[above_sfc & below_mlt] + th2[above_sfc & below_mlt]) * \
            (p1[above_sfc & below_mlt] - p2[above_sfc & below_mlt])
        q_avg[above_sfc & below_mlt] += 0.5 * \
            (q1[above_sfc & below_mlt] + q2[above_sfc & below_mlt]) * \
            (p1[above_sfc & below_mlt] - p2[above_sfc & below_mlt])
        
        #print(p1, T1, q1, th1, p2, T2, q2, th2, th_avg, q_avg)

    # Divide averages by ML depth to get final values
    th_avg = th_avg / mixed_layer_depth
    q_avg = q_avg / mixed_layer_depth

    # Compute corresponding average temperature at the surface
    T_avg = follow_dry_adiabat(100000., p_sfc, th_avg, q_avg)

    #print(p_sfc, th_avg, T_avg, q_avg)

    # Call code to perform parcel ascent
    CAPE, CIN, LCL, LFC, EL = parcel_ascent(p, T, q, p_sfc, T_avg, q_avg,
                                            **kwargs)

    if return_parcel_props:
        return CAPE, CIN, LCL, LFC, EL, T_avg, q_avg
    else:
        return CAPE, CIN, LCL, LFC, EL


def most_unstable_parcel_ascent(p, T, q, p_sfc=None, T_sfc=None, q_sfc=None,
                                mu_parcel='max_wbpt', min_pressure=50000.0,
                                eil_min_cape=100.0, eil_max_cin=250.0,
                                vertical_axis=0, return_parcel_props=False,
                                **kwargs):
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
        return_parcel_props (bool, optional): flag indicating whether to return
            parcel temperature and specific humidity)
        **kwargs: additional keyword arguments passed to parcel_ascent

    Returns:
        CAPE (float or ndarray): convective available potential energy (J/kg)
        CIN (float or ndarray): convective inhibition (J/kg)
        LPL (float or ndarray): lifted parcel level (Pa)
        LCL (float or ndarray): lifting condensation level (Pa)
        LFC (float or ndarray): level of free convection (Pa)
        EL (float or ndarray): equilibrium level (Pa)
        inflow_base (float or ndarray): EIL base (Pa)
        inflow_top (float or ndarray): EIL top (Pa)

    """

    # Check that MU parcel option is valid
    if mu_parcel not in ['max_wbpt', 'max_cape']:
        raise ValueError("mu_parcel must be either 'max_wbpt' or 'max_cape'")

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

    if mu_parcel == 'max_cape':

        # Perform parcel ascent from surface
        cape, cin, lcl, lfc, el = parcel_ascent(p, T, q, p_sfc, T_sfc, q_sfc,
                                                output_scalars=False, **kwargs)
    
        # Reset values where the surface is above the minimum pressure level
        above_min = (p_sfc < min_pressure)
        cape[above_min] = 0.0
        cin[above_min] = np.nan
        lcl[above_min] = np.nan
        lfc[above_min] = np.nan
        el[above_min] = np.nan

        #print('sfc', p_sfc, T_sfc, q_sfc, cape, cin, lcl, lfc, el)

        # Initialise final CAPE, CIN, LCL, LFC, and EL arrays
        CAPE = cape
        CIN = cin
        LCL = lcl
        LFC = lfc
        EL = el

        # Initialise arrays for EIL base and top
        inflow_base = np.full_like(p_sfc, np.nan)
        inflow_top = np.full_like(p_sfc, np.nan)

        # Check if surface is part of EIL
        in_eil = (cape >= eil_min_cape) & (cin <= eil_max_cin)
        inflow_base[in_eil] = p_sfc[in_eil]

        # Initialise the LPL pressure
        p_lpl = p_sfc.copy()
        LPL = p_lpl

        # Loop over levels
        for k in range(k_start, n_lev):

            # Note the pressure and EIL mask from previous level
            in_eil_prev = in_eil
            p_lpl_prev = p_lpl

            # Find points above and below the surface
            above_sfc = (p[k] <= p_sfc)
            below_sfc = np.logical_not(above_sfc)
            if np.all(below_sfc):
                # if all points are below the surface we can skip this level
                continue

            # Find points above the minimum pressure level
            above_min = (p[k] < min_pressure)
            if np.all(above_min):
                # if all points are above the minimum pressure level we can
                # break out of loop
                break

            # Set lifted parcel level (LPL) fields
            p_lpl = np.where(above_sfc, p[k], p_sfc)
            Tp_lpl = np.where(above_sfc, T[k], T_sfc)
            qp_lpl = np.where(above_sfc, q[k], q_sfc)

            # Perform parcel ascent from the LPL
            cape, cin, lcl, lfc, el = parcel_ascent(
                p, T, q, p_lpl, Tp_lpl, qp_lpl, k_lpl=k, output_scalars=False,
                **kwargs
                )

            # Reset values where LPL is above the minimum pressure level
            # (this should only happen where p_sfc < min_pressure)
            cape[above_min] = 0.0
            cin[above_min] = np.nan
            lcl[above_min] = np.nan
            lfc[above_min] = np.nan
            el[above_min] = np.nan

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
            is_base = in_eil & np.isnan(inflow_base)
            inflow_base[is_base] = p_lpl[is_base]
            is_top = in_eil_prev & np.logical_not(in_eil) & np.isnan(inflow_top)
            inflow_top[is_top] = p_lpl_prev[is_top]

            # Reset EIL base and top where layer comprises only a single level
            base_eq_top = (inflow_base == inflow_top)
            inflow_base[base_eq_top] = np.nan
            inflow_top[base_eq_top] = np.nan
            
        if len(p_sfc) == 1:
            # convert outputs to scalars
            CAPE = CAPE[0]
            CIN = CIN[0]
            LPL = LPL[0]
            LCL = LCL[0]
            LFC = LFC[0]
            EL = EL[0]
            inflow_base = inflow_base[0]
            inflow_top = inflow_top[0]

        if return_parcel_props:
            return (CAPE, CIN, LPL, LCL, LFC, EL, inflow_base, inflow_top,
                    Tp_lpl, qp_lpl)
        else:
            return CAPE, CIN, LPL, LCL, LFC, EL, inflow_base, inflow_top

    else:

        # Get phase and polynomial flag from kwargs
        phase = kwargs.get('phase', 'liquid')
        polynomial = kwargs.get('polynomial', True)

        # Compute WBPT at the surface
        thw_sfc = wet_bulb_potential_temperature(p_sfc, T_sfc, q_sfc,
                                                 phase=phase,
                                                 polynomial=polynomial)
        thw_sfc = np.atleast_1d(thw_sfc)

        # Initialise the maximum WBPT
        thw_max = thw_sfc

        # Initialise the lifted parcel level fields using surface values
        p_lpl = p_sfc.copy()
        Tp_lpl = T_sfc.copy()
        qp_lpl = q_sfc.copy()

        #print('sfc', thw_sfc, p_lpl, Tp_lpl, qp_lpl, thw_max)

        # Loop over levels
        for k in range(k_start, n_lev):

            # Find points below the surface
            below_sfc = (p[k] > p_sfc)
            if np.all(below_sfc): 
                # if all points are below the surface we can skip this level
                continue

            # Find points above the minimum pressure level
            above_min = (p[k] < min_pressure)
            if np.all(above_min):
                # if all points are above the minimum pressure level we can
                # break out of the loop
                break

            # Compute WBPT
            thw = wet_bulb_potential_temperature(p[k], T[k], q[k],
                                                 phase=phase,
                                                 polynomial=polynomial)
            thw = np.atleast_1d(thw)

            # For points below the surface or above the minimum pressure level,
            # replace WBPT with the value at the surface
            thw[above_min | below_sfc] = thw_sfc[above_min | below_sfc]

            # Update LPL fields and maximum WBPT
            is_max = (thw > thw_max)
            p_lpl[is_max] = p[k][is_max]
            Tp_lpl[is_max] = T[k][is_max]
            qp_lpl[is_max] = q[k][is_max]
            thw_max[is_max] = thw[is_max]

            #print(k, thw, p_lpl, Tp_lpl, qp_lpl, thw, thw_max)

        #print(p_lpl, Tp_lpl, qp_lpl)

        # Note the LPL
        LPL = p_lpl

        # Perform parcel ascent from the LPL
        CAPE, CIN, LCL, LFC, EL = parcel_ascent(p, T, q, p_lpl, Tp_lpl, qp_lpl,
                                                output_scalars=False, **kwargs)

        # Reset values where the LPL is above the minimum pressure level
        # (this should only happen if p_sfc < min_pressure)
        above_min = (p_lpl < min_pressure)
        CAPE[above_min] = 0.0
        CIN[above_min] = np.nan
        LPL[above_min] = np.nan
        LCL[above_min] = np.nan
        LFC[above_min] = np.nan
        EL[above_min] = np.nan

        if len(p_sfc) == 1:
            # convert outputs to scalars
            CAPE = CAPE[0]
            CIN = CIN[0]
            LPL = LPL[0]
            LCL = LCL[0]
            LFC = LFC[0]
            EL = EL[0]

        if return_parcel_props:
            return CAPE, CIN, LPL, LCL, LFC, EL, Tp_lpl, qp_lpl
        else:
            return CAPE, CIN, LPL, LCL, LFC, EL
