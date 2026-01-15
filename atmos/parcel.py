import numpy as np
import warnings
from atmos import pseudoadiabat
from atmos.constant import Rd
from atmos.utils import (
    interpolate_scalar_to_pressure_level,
    pressure_layer_mean_scalar,
    pressure_layer_maxmin_scalar
)
from atmos.thermo import (
    saturation_specific_humidity,
    virtual_temperature,
    potential_temperature,
    lifting_condensation_level,
    lifting_deposition_level,
    lifting_saturation_level,
    ice_fraction,
    follow_dry_adiabat,
    follow_moist_adiabat,
    pseudo_wet_bulb_temperature,
    wet_bulb_potential_temperature
)


def parcel_ascent(p, T, q, p_lpl, Tp_lpl, qp_lpl, k_lpl=None, p_sfc=None,
                  T_sfc=None, q_sfc=None, vertical_axis=0, output_scalars=True,
                  phase='liquid', pseudo=True, polynomial=True, explicit=False,
                  dp=500.0, which_lfc='first', which_el='last',
                  count_cape_below_lcl=False, count_cin_below_lcl=True,
                  count_cape_below_lfc=False, count_cin_above_lfc=True):
    """
    Performs a parcel ascent from a specified lifted parcel level (LPL) and 
    returns the resulting convective available potential energy (CAPE) and
    convective inhibition (CIN), together with the lifting condensation level
    (LCL), level of free convection (LFC), level of maximum buoyancy (LMB) and
    equilibrium level (EL). In the case of multiple layers of positive
    buoyancy, the final LFC and EL are selected according to 'which_lfc' and
    'which_el', where
        * 'first' corresponds to the first (lowest) layer of positive buoyancy
        * 'maxcape' corresponds to the layer of positive buoyancy with the
          largest CAPE
        * 'last' corresponds to the last (highest) layer of positive buoyancy
    Options also exist to count layers of positive buoyancy below the LCL or
    below the LFC towards CAPE and to count layers of negative buoyancy below
    the LCL or above the LFC towards CIN.

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
        which_lfc (str, optional): choice for LFC (valid options are 'first', 
            'last', or 'maxcape'; default is 'first')
        which_el (str, optional): choice for EL (valid options are 'first', 
            'last', or 'maxcape'; default is 'last')
        count_cape_below_lcl (bool, optional): flag indicating whether to
            count positive areas below the LCL towards CAPE (default is False)
        count_cin_below_lcl (bool, optional): flag indicating whether to
            count negative areas below the LCL towards CIN (default is True)
        count_cape_below_lfc (bool, optional): float indicating whether to
            count positive areas below the LFC towards CAPE (default is False)
        count_cin_above_lfc (bool, optional): flag indicating whether to 
            count negative areas above the LFC towards CIN (default is True)

    Returns:
        CAPE (float or ndarray): convective available potential energy (J/kg)
        CIN (float or ndarray): convective inhibition (J/kg)
        LCL (float or ndarray): lifting condensation level (Pa)
        LFC (float or ndarray): level of free convection (Pa)
        LMB (float or ndarray): level of maximum buoyancy (Pa)
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
            """
            Incompatible selection for which_lfc and which_el. For
            which_lfc='maxcape', which_el must be set as 'maxcape'
            or 'last'.
            """
        )
    if (which_lfc == 'last' and which_el != 'last'):
        raise ValueError(
            """
            Incompatible selection for which_lfc and which_el. For 
            which_lfc='last', which_el must be set as 'last'.
            """
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
        warnings.warn(f'LPL is below {bottom} at {n_pts} points')
    
    # Check that LPL is not above top level
    lpl_above_top = (p_lpl < p[-1])
    if np.any(lpl_above_top):
        n_pts = np.count_nonzero(lpl_above_top)
        warnings.warn(f'LPL is above top level at {n_pts} points')

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
        warnings.warn(f'LCL above top level for {n_pts} points')
        k_stop = n_lev-1  # stop loop a level early
    else:
        k_stop = n_lev

    # Create arrays for negative and positive areas
    neg_area = np.zeros_like(p_lpl)
    pos_area = np.zeros_like(p_lpl)

    # Create arrays for temporary CAPE, CIN, LFC, and EL values
    cape = np.zeros_like(p_lpl)        # CAPE for most recent positive area
    cape_max = np.zeros_like(p_lpl)    # CAPE for largest positive area
    cin_total = np.zeros_like(p_lpl)   # total CIN across all negative areas
    p_lfc = np.zeros_like(p_lpl)       # LFC for most recent positive area
    p_lmb = np.zeros_like(p_lpl)       # LMB for most recent positive area
    p_el = np.zeros_like(p_lpl)        # EL for most recent positive area

    # Create arrays for final CAPE, CIN, LFC, and EL values
    CAPE = np.zeros_like(p_lpl)
    CIN = np.full_like(p_lpl, np.nan)  # undefined where CAPE = 0
    LFC = np.full_like(p_lpl, np.nan)  # undefined where CAPE = 0
    LMB = np.full_like(p_lpl, np.nan)  # undefined where CAPE = 0
    EL = np.full_like(p_lpl, np.nan)   # undefined where CAPE = 0

    # Initialise level 2 environmental fields
    if k_lpl is None:
        p2 = p_sfc.copy()
        T2 = T_sfc.copy()
        q2 = q_sfc.copy()
    else:
        k_start = k_lpl + 1
        lpl_above_sfc = (p[k_lpl] <= p_sfc)
        p2 = np.where(lpl_above_sfc, p[k_lpl], p_sfc)
        T2 = np.where(lpl_above_sfc, T[k_lpl], T_sfc)
        q2 = np.where(lpl_above_sfc, q[k_lpl], q_sfc)

    # Initialise level 2 parcel properties using LPL values
    Tp2 = Tp_lpl.copy()
    qp2 = qp_lpl.copy()

    # Initialise parcel buoyancy (virtual temperature excess) at level 2
    B2 = virtual_temperature(Tp2, qp2) - virtual_temperature(T2, q2)

    # Initialise the maximum buoyancy and corresponding pressure
    p2_above_sfc = (p2 <= p_sfc)
    B_max = np.where(p2_above_sfc, B2, 0.0)
    p_max = np.where(p2_above_sfc, p2, np.nan)

    #print(p_lpl, Tp_lpl, qp_lpl)
    #print(p_lcl, Tp_lcl)
    #print(Tp2, qp2, B2)

    # Loop over levels, accounting for addition of extra level for LCL
    for k in range(k_start, k_stop+1):

        # Update level 1 fields
        p1 = p2.copy()
        T1 = T2.copy()
        q1 = q2.copy()
        Tp1 = Tp2.copy()
        qp1 = qp2.copy()
        B1 = B2.copy()

        # If at or above 100 hPa with negative buoyancy, break out of loop
        if np.all((p1 <= 10000.0) & (B1 < 0.0)):
            break

        # Find level 1 points below and above the LCL
        p1_below_lcl = (p1 > p_lcl)
        p1_above_lcl = np.logical_not(p1_below_lcl)

        # Set level 2 environmental fields
        if np.any(p1_below_lcl):
            # use level k below LCL
            p2[p1_below_lcl] = p[k][p1_below_lcl]
            T2[p1_below_lcl] = T[k][p1_below_lcl]
            q2[p1_below_lcl] = q[k][p1_below_lcl]
        if np.any(p1_above_lcl):
            # use level k-1 above LCL to account for additional level
            p2[p1_above_lcl] = p[k-1][p1_above_lcl]
            T2[p1_above_lcl] = T[k-1][p1_above_lcl]
            q2[p1_above_lcl] = q[k-1][p1_above_lcl]

        # Reset level 2 environmental fields to surface values where
        # level 2 is below the surface
        p2_below_sfc = (p2 > p_sfc)
        p2 = np.where(p2_below_sfc, p_sfc, p2)
        T2 = np.where(p2_below_sfc, T_sfc, T2)
        q2 = np.where(p2_below_sfc, q_sfc, q2)

        # If all level 2 points are below the surface, skip this level
        if np.all(p2_below_sfc):
            continue

        # If crossing the LPL, reset level 1 as the LPL
        cross_lpl = (p1 > p_lpl) & (p2 < p_lpl)
        if np.any(cross_lpl):

            #print('Crossing LPL', p1, p2, p_lpl)

            # Interpolate to get environmental temperature and specific
            # humidity at the LPL
            weight = np.log(p1[cross_lpl] / p_lpl[cross_lpl]) / \
                np.log(p1[cross_lpl] / p2[cross_lpl])
            T1[cross_lpl] = (1 - weight) * T1[cross_lpl] + \
                weight * T2[cross_lpl]
            q1[cross_lpl] = (1 - weight) * q1[cross_lpl] + \
                weight * q2[cross_lpl]
                        
            # Use LPL pressure
            p1[cross_lpl] = p_lpl[cross_lpl]

        # Find level 1 points below and above the LPL
        p1_below_lpl = (p1 > p_lpl)
        p1_above_lpl = np.logical_not(p1_below_lpl)

        # If all level 1 points are below the LPL, skip this level
        if np.all(p1_below_lpl):
            continue

        # Find level 1 points that are at the LPL
        p1_is_lpl = (p1 == p_lpl)
        if np.any(p1_is_lpl):

            # Recompute parcel buoyancy at level 1
            B1[p1_is_lpl] = virtual_temperature(
                Tp1[p1_is_lpl], qp1[p1_is_lpl]
            ) - virtual_temperature(
                T1[p1_is_lpl], q1[p1_is_lpl]
            )

            # Update the maximum buoyancy and corresponding pressure
            B1_is_max = (B1 > B_max)
            if np.any(B1_is_max):
                if count_cape_below_lcl:
                    update = B1_is_max & p1_is_lpl
                else:
                    update = B1_is_max & p1_is_lpl & p1_above_lcl  # LPL = LCL
                B_max[update] = B1[update]
                p_max[update] = p1[update]

        # If crossing the LCL, reset level 2 as the LCL
        cross_lcl = (p1 > p_lcl) & (p2 < p_lcl)
        if np.any(cross_lcl):

            #print('Crossing LCL', p1, p2, p_lcl)

            # Interpolate to get environmental temperature and specific 
            # humidity at the LCL
            weight = np.log(p1[cross_lcl] / p_lcl[cross_lcl]) / \
                np.log(p1[cross_lcl] / p2[cross_lcl])
            T2[cross_lcl] = (1 - weight) * T1[cross_lcl] + \
                weight * T2[cross_lcl]
            q2[cross_lcl] = (1 - weight) * q1[cross_lcl] + \
                weight * q2[cross_lcl]

            # Use LCL pressure
            p2[cross_lcl] = p_lcl[cross_lcl]

        # Find points undergoing dry (unsaturated) and wet (saturated) ascent
        dry_ascent = p1_above_lpl & p1_below_lcl
        wet_ascent = p1_above_lcl  # since LCL is always at or above the LPL

        # Set parcel temperature and specific humidity at level 2
        if np.any(dry_ascent):

            # Follow a dry adiabat to get parcel temperature
            Tp2[dry_ascent] = follow_dry_adiabat(
                p1[dry_ascent], p2[dry_ascent], Tp1[dry_ascent],
                qp1[dry_ascent]
            )
 
            # Specific humidity is conserved
            qp2[dry_ascent] = qp1[dry_ascent]

        if np.any(wet_ascent):

            if pseudo:

                # Follow a pseudoadiabat to get parcel temperature
                Tp2[wet_ascent] = follow_moist_adiabat(
                    p1[wet_ascent], p2[wet_ascent], Tp1[wet_ascent],
                    phase=phase, pseudo=True, polynomial=polynomial,
                    explicit=explicit, dp=dp
                )

                # Specific humidity is equal to its value at saturation
                omega = ice_fraction(Tp2[wet_ascent])
                qp2[wet_ascent] = saturation_specific_humidity(
                    p2[wet_ascent], Tp2[wet_ascent],
                    phase=phase, omega=omega
                )

            else:

                # Follow a saturated adiabat to get parcel temperature
                Tp2[wet_ascent] = follow_moist_adiabat(
                    p1[wet_ascent], p2[wet_ascent], Tp1[wet_ascent],
                    qt=qt[wet_ascent], phase=phase, pseudo=False,
                    polynomial=polynomial, explicit=explicit, dp=dp
                )

                # Specific humidity is equal to its value at saturation
                omega = ice_fraction(Tp2[wet_ascent])
                qp2[wet_ascent] = saturation_specific_humidity(
                    p2[wet_ascent], Tp2[wet_ascent], qt=qt[wet_ascent],
                    phase=phase, omega=omega
                )

        # Compute parcel buoyancy at level 2
        B2 = virtual_temperature(Tp2, qp2, qt=qt) - virtual_temperature(T2, q2)

        # Update the maximum buoyancy and corresponding pressure
        B2_is_max = (B2 > B_max)
        if np.any(B2_is_max):
            if count_cape_below_lcl:
                update = B2_is_max & p1_above_lpl
            else:
                update = B2_is_max & p1_above_lcl
            B_max[update] = B2[update]
            p_max[update] = p2[update]

        # Initialise mask indicating where positive area is complete
        done = np.zeros_like(p2).astype(bool)

        # Find points where parcel is within negative area
        neg_to_neg = (B1 <= 0.0) & (B2 <= 0.0)
        if np.any(neg_to_neg):

            #print('In negative area', p1, p2, B1, B2)

            # Update negative area
            neg_area[neg_to_neg] -= Rd * 0.5 * \
                (B1[neg_to_neg] + B2[neg_to_neg]) * \
                np.log(p1[neg_to_neg] / p2[neg_to_neg])

        # Find points where parcel is crossing from negative to positive area
        neg_to_pos = (B1 <= 0.0) & (B2 > 0.0)
        if np.any(neg_to_pos):

            #print('Crossing from negative to positive area', p1, p2, B1, B2)

            # Interpolate to get pressure at crossing level
            px = np.zeros_like(p2)
            weight = B1[neg_to_pos] / (B1[neg_to_pos] - B2[neg_to_pos])
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
                update = neg_to_pos & p1_above_lpl
            else:
                # update if above LCL
                update = neg_to_pos & p1_above_lcl
            cin_total[update] += neg_area[update]

            # Update LFC
            if count_cape_below_lcl:
                # update if above LPL
                update = neg_to_pos & p1_above_lpl
            else:
                # update if above LCL
                update = neg_to_pos & p1_above_lcl
            p_lfc[update] = px[update]

            # Reset the negative area to zero
            neg_area[neg_to_pos] = 0.0

        # Find points where parcel is within positive area
        pos_to_pos = (B1 > 0.0) & (B2 > 0.0)
        if np.any(pos_to_pos):

            #print('In positive area', p1, p2, B1, B2)

            # Update positive area
            pos_area[pos_to_pos] += Rd * 0.5 * \
                (B1[pos_to_pos] + B2[pos_to_pos]) * \
                np.log(p1[pos_to_pos] / p2[pos_to_pos])

        # Find points where parcel is crossing from positive to negative area
        pos_to_neg = (B1 > 0.0) & (B2 <= 0.0)
        if np.any(pos_to_neg):

            #print('Crossing from positive to negative area', p1, p2, B1, B2)

            # Interpolate to get pressure at crossing level
            px = np.zeros_like(p2)
            weight = B1[pos_to_neg] / (B1[pos_to_neg] - B2[pos_to_neg])
            px[pos_to_neg] = p1[pos_to_neg] ** (1 - weight) * \
                p2[pos_to_neg] ** weight

            # Update positive and negative areas
            pos_area[pos_to_neg] += Rd * 0.5 * B1[pos_to_neg] * \
                np.log(p1[pos_to_neg] / px[pos_to_neg])
            neg_area[pos_to_neg] -= Rd * 0.5 * B2[pos_to_neg] * \
                np.log(px[pos_to_neg] / p2[pos_to_neg])
            
            # Update CAPE, LMB, and EL, and mark positive area as complete
            if count_cape_below_lcl:
                # update if above LPL
                update = pos_to_neg & p1_above_lpl
            else:
                # update if above LCL
                update = pos_to_neg & p1_above_lcl
            cape[update] = pos_area[update]
            p_lmb[update] = p_max[update]
            p_el[update] = px[update]
            done[update] = True

            # Reset the positive area to zero
            pos_area[pos_to_neg] = 0.0

        # Reset negative areas that shouldn't be counted
        if count_cin_below_lcl:
            neg_area[p1_below_lpl] = 0.0
        else:
            neg_area[p1_below_lcl] = 0.0

        # Reset positive areas that shouldn't be counted
        if count_cape_below_lcl:
            pos_area[p1_below_lpl] = 0.0
        else:
            pos_area[p1_below_lcl] = 0.0

        if count_cape_below_lcl:

            # If positively buoyant at the LPL then set LFC = LPL
            pos_at_lpl = (p1 == p_lpl) & (B1 > 0.0)
            if np.any(pos_at_lpl):
                p_lfc[pos_at_lpl] = p_lpl[pos_at_lpl]

        else:

            # If positively buoyant at the LCL then set LFC = LCL
            pos_at_lcl = (p1 == p_lcl) & (B1 > 0.0)
            if np.any(pos_at_lcl):
                p_lfc[pos_at_lcl] = p_lcl[pos_at_lcl]

        # If positively buoyant at the top level then update CAPE, LMB, and EL,
        # and mark positive area as complete
        pos_at_top = (p2 == p[-1]) & (B2 > 0.0)
        if np.any(pos_at_top):
            n_pts = np.count_nonzero(pos_at_top)
            warnings.warn(f'Positive buoyancy at top level for {n_pts} points')
            cape[pos_at_top] = pos_area[pos_at_top]
            p_lmb[pos_at_top] = p_max[pos_at_top]
            p_el[pos_at_top] = p2[pos_at_top]
            done[pos_at_top] = True

        #print(k, p2, T2, q2, Tp2, qp2, qt)
        #print(k, p1, p2, B1, B2, pos_area, neg_area)
        #print(k, pos_area, neg_area, cape, cape_max, cin_total)
        #print(k, p_lfc, p_lmb, p_el)

        if np.any(done):

            # Note if this is the first positive area
            is_first = (CAPE == 0.0)

            # Note if this is the "maxcape" positive area
            is_max = (cape > cape_max)
            if np.any(is_max):
                # update the maximum CAPE
                cape_max[done & is_max] = cape[done & is_max]

            # Create masks for updating output arrays based on which_lfc and which_el
            if which_lfc == 'first':
                update_lfc = done & is_first
            elif which_lfc == 'maxcape':
                update_lfc = done & is_max
            else:
                update_lfc = done
            if which_el == 'first':
                update_el = done & is_first
            elif which_el == 'maxcape':
                update_el = done & is_max
            else:
                update_el = done

            # Update CAPE
            if not count_cape_below_lfc:
                # reset CAPE where the LFC has been updated
                CAPE[update_lfc] = 0.0
            CAPE[update_el] += cape[update_el]

            # Update CIN
            if count_cin_above_lfc:
                CIN[update_el] = cin_total[update_el]
            else:
                CIN[update_lfc] = cin_total[update_lfc]

            # Update the LFC, LMB, and EL
            LFC[update_lfc] = p_lfc[update_lfc]
            LMB[update_el] = p_lmb[update_el]
            EL[update_el] = p_el[update_el]

    if output_scalars:
        # convert outputs to scalars
        CAPE = CAPE.squeeze()
        CIN = CIN.squeeze()
        LCL = LCL.squeeze()
        LFC = LFC.squeeze()
        LMB = LMB.squeeze()
        EL = EL.squeeze()

    return CAPE, CIN, LCL, LFC, LMB, EL


def surface_based_parcel(p, T, q, p_sfc=None, T_sfc=None, q_sfc=None,
                         vertical_axis=0, **kwargs):
    """
    Performs a surface-based (SB) parcel ascent and returns the resulting
    convective available potential energy (CAPE) and convective inhibition
    (CIN), along with the lifting condensation level (LCL), level of free
    convection (LFC), level of maximum buoyancy (LMB), and equilibrium level
    (EL). The SB parcel is defined using the lowest level in the profile or
    the surface pressure, temperature, and specific humidity, if supplied.

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
        LMB (float or ndarray): level of maximum buoyancy (Pa)
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
    CAPE, CIN, LCL, LFC, LMB, EL = parcel_ascent(
        p, T, q, p_lpl, Tp_lpl, qp_lpl,
        p_sfc=p_sfc, T_sfc=T_sfc, q_sfc=q_sfc,
        **kwargs
    )
    
    return CAPE, CIN, LCL, LFC, LMB, EL


def mixed_layer_parcel(p, T, q, p_sfc=None, T_sfc=None, q_sfc=None,
                       mixed_layer_depth=5000.0, vertical_axis=0,
                       return_parcel_properties=False, **kwargs):
    """
    Performs a mixed-layer (ML) parcel ascent and returns the resulting
    convective available potential energy (CAPE) and convective inhibition
    (CIN), along with the lifting condensation level (LCL), level of free
    convection (LFC), level of maximum buoyancy (LMB), and equilibrium level
    (EL). The ML parcel is defined using the mass-weighted average potential
    temperature and specific humidity, computed over a specified mixed-layer
    depth, together with the surface pressure.

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
        LMB (float or ndarray): level of maximum buoyancy (Pa)
        EL (float or ndarray): equilibrium level (Pa)

    """

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        p = np.moveaxis(p, vertical_axis, 0)
        T = np.moveaxis(T, vertical_axis, 0)
        q = np.moveaxis(q, vertical_axis, 0)

    # Compute potential temperature
    th = potential_temperature(p, T, q)
    if p_sfc is None:
        th_sfc = None
    else:
        th_sfc = potential_temperature(p_sfc, T_sfc, q_sfc)

    # Set pressure at bottom and top of mixed layer
    if p_sfc is None:
        p_bot = p[0]
        p_top = p[0] - mixed_layer_depth
    else:
        p_bot = p_sfc
        p_top = p_sfc - mixed_layer_depth

    # Average potential temperature and specific humidity across mixed layer
    th_avg = pressure_layer_mean_scalar(p, th, p_bot, p_top,
                                        p_sfc=p_sfc, s_sfc=th_sfc)
    q_avg = pressure_layer_mean_scalar(p, q, p_bot, p_top,
                                       p_sfc=p_sfc, s_sfc=q_sfc)

    # Compute corresponding temperature at the surface
    T_avg = follow_dry_adiabat(100000., p_bot, th_avg, q_avg)

    # Set initial parcel temperature and specific humidity
    Tpi, qpi = T_avg, q_avg

    # Call code to perform parcel ascent
    CAPE, CIN, LCL, LFC, LMB, EL = parcel_ascent(
        p, T, q, p_bot, Tpi, qpi,
        p_sfc=p_sfc, T_sfc=T_sfc, q_sfc=q_sfc,
        **kwargs
    )

    if return_parcel_properties:
        return CAPE, CIN, LCL, LFC, LMB, EL, Tpi, qpi
    else:
        return CAPE, CIN, LCL, LFC, LMB, EL


def most_unstable_parcel(p, T, q, p_sfc=None, T_sfc=None, q_sfc=None,
                         thw=None, thw_sfc=None, p_bot=None, p_top=None,
                         most_unstable_method='max_wbpt',
                         eil_min_cape=100.0, eil_max_cin=250.0,
                         vertical_axis=0, return_parcel_properties=False,
                         **kwargs):
    """
    Performs a most-unstable (MU) parcel ascent and returns the resulting
    convective available potential energy (CAPE) and convective inhibition
    (CIN), along with the lifted parcel level (LPL), lifting condensation level
    (LCL), level of free convection (LFC), level of maximum buoyancy (LMB) and
    equilibrium level (EL). By default, the MU parcel is defined using the
    maximum wet-bulb potential temperature (most_unstable_method='max_wbpt')
    within a specified layer (by default the lowest 300 hPa above the surface).
    Alternatively, it can be defined by launching parcels from every level in
    this layer and identifying the one that results in the maximum CAPE
    (most_unstable_method='max_cape'). In this case, the function also returns
    the base and top of the effective inflow layer (EIL). Following Thompson
    et al. (2007), the EIL is defined as the first contiguous layer (comprising
    at least two levels) where CAPE >= 100 J/kg and CIN <= 250 J/kg. Note that
    the 'max_cape' calculation is significantly slower than the 'max_wbpt'
    calculation, particularly when the input data has high vertical resolution.

    Args:
        p (ndarray): pressure profile(s) (Pa)
        T (ndarray): temperature profile(s) (K)
        q (ndarray): specific humidity profile(s) (kg/kg)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        T_sfc (float or ndarray, optional): surface temperature (K)
        q_sfc (float or ndarray, optional): surface specific humidity (kg/kg)
        thw (float or ndarray, optional): wet-bulb potential temperature (K)
        thw_sfc (float or ndarray, optional): surface wet-bulb potential
            temperature (K)
        p_bot (float, optional): pressure at bottom of layer (Pa) (default is
            None, in which case the surface is used)
        p_top (float, optional): pressure at top of layer (Pa) (default is
            None, in which case the level 300 hPa above the surface is used)
        most_unstable_method (str, optional): method for defining the most
            unstable parcel (valid options are 'max_wbpt' or 'max_cape';
            default is 'max_wbpt')
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
        LMB (float or ndarray): level of maximum buoyancy (Pa)
        EL (float or ndarray): equilibrium level (Pa)
        EIB (float or ndarray): effective inflow base (Pa)
        EIT (float or ndarray): effective inflow top (Pa)
        Tpi (float or ndarray): initial parcel temperature (K)
        qpi (float or ndarray): initial parcel specific humidity (g/kg)

    """

    # Check that MU parcel option is valid
    if most_unstable_method not in ['max_wbpt', 'max_cape']:
        raise ValueError("""
        most_unstable_method must be either 'max_wbpt' or 'max_cape'
        """)

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

    # Set the bottom of the layer, if not specified
    if p_bot is None:
        if p_sfc is None:
            p_bot = p[0]  # lowest level
        else:
            p_bot = p_sfc  # surface

    # Set the top of the layer, if not specified
    if p_top is None:
        if p_sfc is None:
            p_top = p[0] - 30000.0  # 300 hPa above lowest level
        else:
            p_top = p_sfc - 30000.0  # 300 hPa above surface

    # Note the number of levels
    n_lev = p.shape[0]

    if most_unstable_method == 'max_cape':

        # Set initial LPL as the surface
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

        # Perform parcel ascent
        cape, cin, p_lcl, p_lfc, p_lmb, p_el = parcel_ascent(
            p, T, q, p_lpl, Tp_lpl, qp_lpl,
            p_sfc=p_sfc, T_sfc=T_sfc, q_sfc=q_sfc,
            output_scalars=False, **kwargs
        )
    
        # Reset parcel parameters where the LPL is outside the layer
        outside_layer = (p_lpl > p_bot) | (p_lpl < p_top)
        cape[outside_layer] = 0.0
        cin[outside_layer] = np.nan
        p_lfc[outside_layer] = np.nan
        p_lmb[outside_layer] = np.nan
        p_el[outside_layer] = np.nan

        #print('sfc', p_lpl, p_lcl, p_lfc, p_el, cape, cin)

        # Initialise the output arrays
        CAPE = cape
        CIN = cin
        LPL = p_lpl
        LCL = p_lcl
        LFC = p_lfc
        LMB = p_lmb
        EL = p_el
        Tpi = Tp_lpl
        qpi = qp_lpl

        # Initialise arrays for the effective inflow base (EIB) and top (EIT)
        EIB = np.full_like(cape, np.nan)
        EIT = np.full_like(cape, np.nan)

        # Check if CAPE and CIN meet the criteria for the EIL
        in_eil = (cape >= eil_min_cape) & (cin <= eil_max_cin)

        # Loop over levels
        for k in range(k_start, n_lev):

            # Note LPL and EIL mask from the previous level
            in_eil_prev = in_eil.copy()
            p_lpl_prev = p_lpl.copy()

            # Find points below the layer bottom
            below_bot = (p[k] > p_bot)
            if np.all(below_bot):
                # skip this level
                continue

            # Find points above the layer top
            above_top = (p[k] < p_top)
            if np.all(above_top):
                # break out of loop
                break

            # Set the LPL and parcel temperature and specific humidity
            p_lpl = p[k]
            Tp_lpl = T[k]
            qp_lpl = q[k]

            # Perform parcel ascent
            cape, cin, p_lcl, p_lfc, p_lmb, p_el = parcel_ascent(
                p, T, q, p_lpl, Tp_lpl, qp_lpl, k_lpl=k,
                p_sfc=p_sfc, T_sfc=T_sfc, q_sfc=q_sfc,
                output_scalars=False, **kwargs
            )

            #print(k, p_lpl, p_lcl, p_lfc, p_lmb, p_el, cape, cin)

            # Update output arrays where CAPE is maximised and this level is
            # within the layer
            is_max = (cape > CAPE) & np.logical_not(below_bot) & \
                np.logical_not(above_top)
            if np.any(is_max):
                CAPE[is_max] = cape[is_max]
                CIN[is_max] = cin[is_max]
                LPL[is_max] = p_lpl[is_max]
                LCL[is_max] = p_lcl[is_max]
                LFC[is_max] = p_lfc[is_max]
                LMB[is_max] = p_lmb[is_max]
                EL[is_max] = p_el[is_max]
                Tpi[is_max] = Tp_lpl[is_max]
                qpi[is_max] = qp_lpl[is_max]

            # Check if CAPE and CIN meet the criteria for the EIL
            in_eil = (cape >= eil_min_cape) & (cin <= eil_max_cin)

            # Update the EIB and EIT
            is_eib = in_eil & in_eil_prev & np.isnan(EIB)
            EIB[is_eib] = p_lpl_prev[is_eib]
            is_eit = np.logical_not(in_eil) & in_eil_prev & \
                np.isfinite(EIB) & np.isnan(EIT)
            EIT[is_eit] = p_lpl_prev[is_eit]

        # Set the EIT for any incomplete EILs
        is_eit = in_eil & in_eil_prev & np.isfinite(EIB) & np.isnan(EIT)
        EIT[is_eit] = p_lpl[is_eit]
            
        # convert outputs to scalars
        CAPE = CAPE.squeeze()
        CIN = CIN.squeeze()
        LPL = LPL.squeeze()
        LCL = LCL.squeeze()
        LFC = LFC.squeeze()
        LMB = LMB.squeeze()
        EL = EL.squeeze()
        EIB = EIB.squeeze()
        EIT = EIT.squeeze()
        Tpi = Tpi.squeeze()
        qpi = qpi.squeeze()

        if return_parcel_properties:
            return CAPE, CIN, LPL, LCL, LFC, LMB, EL, EIB, EIT, Tpi, qpi
        else:
            return CAPE, CIN, LPL, LCL, LFC, LMB, EL, EIB, EIT

    else:  # most_unstable_method = 'max_wbpt'

        if thw is None:

            # Get phase and polynomial flag from kwargs
            phase = kwargs.get('phase', 'liquid')
            polynomial = kwargs.get('polynomial', True)
            explicit = kwargs.get('explicit', False)
            dp = kwargs.get('dp', 500.0)

            # Compute wet-bulb potential temperature (WBPT)
            thw = wet_bulb_potential_temperature(
                p, T, q, phase=phase, polynomial=polynomial,
                explicit=explicit, dp=dp
            )
            if p_sfc is None:
                thw_sfc = None
            else:
                thw_sfc = wet_bulb_potential_temperature(
                    p_sfc, T_sfc, q_sfc, phase=phase, polynomial=polynomial,
                    explicit=explicit, dp=dp
                )

        # Find the level corresponding to the maximum WBPT between p_bot and
        # p_top and set this as the LPL
        _, p_lpl = pressure_layer_maxmin_scalar(
            p, thw, p_bot, p_top, p_sfc=p_sfc, s_sfc=thw_sfc, statistic='max'
        )

        # Interpolate to get the parcel temperature and specific humidity at
        # the LPL
        Tp_lpl = interpolate_scalar_to_pressure_level(
            p, T, p_lpl, p_sfc=p_sfc, s_sfc=T_sfc,
        )
        qp_lpl = interpolate_scalar_to_pressure_level(
            p, q, p_lpl, p_sfc=p_sfc, s_sfc=q_sfc,
        )

        # Note the LPL and initial parcel temperature and specific humidity
        LPL, Tpi, qpi = p_lpl, Tp_lpl, qp_lpl

        # Perform parcel ascent from the LPL
        CAPE, CIN, LCL, LFC, LMB, EL = parcel_ascent(
            p, T, q, p_lpl, Tp_lpl, qp_lpl,
            p_sfc=p_sfc, T_sfc=T_sfc, q_sfc=q_sfc,
            **kwargs
        )

        if return_parcel_properties:
            return CAPE, CIN, LPL, LCL, LFC, LMB, EL, Tpi, qpi
        else:
            return CAPE, CIN, LPL, LCL, LFC, LMB, EL


def effective_parcel(p, T, q, p_sfc=None, T_sfc=None, q_sfc=None,
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
        LMB (float or ndarray): level of maximum buoyancy (Pa)
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
        _, _, _, _, _, _, _, p_eib, p_eit = most_unstable_parcel(
            p, T, q, p_sfc=p_sfc, T_sfc=T_sfc, q_sfc=q_sfc, 
            mu_parcel='max_cape'
        )

    # Note the pressure at the mid-point of the EIL
    p_mid = 0.5 * (p_eib + p_eit)

    # Compute potential temperature
    th = potential_temperature(p, T, q)
    if p_sfc is None:
        th_sfc = None
    else:
        th_sfc = potential_temperature(p_sfc, T_sfc, q_sfc)

    # Average potential temperature and specific humidity across EIL
    th_avg = pressure_layer_mean_scalar(p, th, p_eib, p_eit,
                                        p_sfc=p_sfc, s_sfc=th_sfc)
    q_avg = pressure_layer_mean_scalar(p, q, p_eib, p_eit,
                                       p_sfc=p_sfc, s_sfc=q_sfc)

    # Compute corresponding temperature at the mid-point of the EIL
    T_avg = follow_dry_adiabat(100000., p_mid, th_avg, q_avg)

    # Set LPL pressure and initial parcel temperature and specific humidity
    LPL, Tpi, qpi = p_mid, T_avg, q_avg

    # Call code to perform parcel ascent
    CAPE, CIN, LCL, LFC, LMB, EL = parcel_ascent(
        p, T, q, LPL, Tpi, qpi,
        p_sfc=p_sfc, T_sfc=T_sfc, q_sfc=q_sfc,
        **kwargs
    )

    if return_parcel_properties:
        return CAPE, CIN, LPL, LCL, LFC, LMB, EL, Tpi, qpi
    else:
        return CAPE, CIN, LPL, LCL, LFC, LMB, EL


def parcel_descent(p, T, q, p_dpl, Tp_dpl, k_dpl=None,
                   p_sfc=None, T_sfc=None, q_sfc=None,
                   vertical_axis=0, output_scalars=True,
                   phase='liquid', polynomial=True,
                   explicit=False, dp=500.0):
    """
    Performs a parcel descent from a specified downdraft parcel level (DPL) 
    and returns the resulting downdraft convective available potential energy
    (DCAPE) and downdraft convective inhibition (DCIN), along with the
    downdraft equilibrium level (DEL) and final downdraft parcel temperature.
    It is assumed that there will only be one (if any) DEL present in the
    profile. If multiple DELs exist, the last (lowest) one will be returned.

    Args:
        p (ndarray): pressure profile(s) (Pa)
        T (ndarray): temperature profile(s) (K)
        q (ndarray): specific humidity profile(s) (kg/kg)
        p_dpl (float or ndarray): DPL pressure (Pa)
        Tp_dpl (float or ndarray): parcel temperature at the DPL (K)
        k_dpl (int, optional): index of vertical axis corresponding to the DPL
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        T_sfc (float or ndarray, optional): surface temperature (K)
        q_sfc (float or ndarray, optional): surface specific humidity (kg/kg)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)
        output_scalars (bool, optional): flag indicating whether to convert
            output arrays to scalars if input profiles are 1D (default is True)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        polynomial (bool, optional): flag indicating whether to use polynomial
            fits to pseudoadiabats (default is True)
        explicit (bool, optional): flag indicating whether to use explicit
            integration of lapse rate equation (default is False)
        dp (float, optional): pressure increment for integration of lapse rate
            equation (default is 500 Pa = 5 hPa)

    Returns:
        DCAPE (float or ndarray): downdraft convective available potential
            energy (J/kg)
        DCIN (float or ndarray): downdraft convective inhibition (J/kg)
        DEL (float or ndarray): downdraft equilibrium level (Pa)
        Tpf (float or ndarray): final downdraft parcel temperature (K)

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

    # Make sure that DPL quantities are at least 1D
    p_dpl = np.atleast_1d(p_dpl)
    Tp_dpl = np.atleast_1d(Tp_dpl)

    # If surface-level fields not provided, use lowest level values
    if p_sfc is None:
        bottom = 'lowest level'
        k_stop = 0  # stop loop at first level
        p_sfc = p[0]
        T_sfc = T[0]
        q_sfc = q[0]
    else:
        bottom = 'surface'
        k_stop = -1  # stop loop below first level

    # Make sure that surface fields are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    T_sfc = np.atleast_1d(T_sfc)
    q_sfc = np.atleast_1d(q_sfc)

    # Ensure that all thermodynamic variables have the same type
    p = p.astype(p_dpl.dtype)
    T = T.astype(p_dpl.dtype)
    q = q.astype(p_dpl.dtype)
    Tp_dpl = Tp_dpl.astype(p_dpl.dtype)
    p_sfc = p_sfc.astype(p_dpl.dtype)
    T_sfc = T_sfc.astype(p_dpl.dtype)
    q_sfc = q_sfc.astype(p_dpl.dtype)

    # Check that array dimensions are compatible
    if p.shape != T.shape or T.shape != q.shape:
        raise ValueError(f"""Incompatible profile arrays: 
                         {p.shape}, {T.shape}, {q.shape}""")
    if p_dpl.shape != Tp_dpl.shape:
        raise ValueError(f"""Incompatible DPL arrays: 
                         {p_dpl.shape}, {Tp_dpl.shape}""")
    if p_sfc.shape != T_sfc.shape or T_sfc.shape != q_sfc.shape:
        raise ValueError(f"""Incompatible surface arrays: 
                         {p_sfc.shape}, {T_sfc.shape}, {q_sfc.shape}""")
    if p[0].shape != p_dpl.shape:
        raise ValueError(f"""Incompatible profile and DPL arrays: 
                         {p.shape}, {p_dpl.shape}""")
    if p[0].shape != p_sfc.shape:
        raise ValueError(f"""Incompatible profile and surface arrays: 
                         {p.shape}, {p_sfc.shape}""")

    # Check that DPL is not below the surface
    dpl_below_sfc = (p_dpl > p_sfc)
    if np.any(dpl_below_sfc):
        n_pts = np.count_nonzero(dpl_below_sfc)
        raise ValueError(f'DPL below {bottom} at {n_pts} points')
    
    # Check that DPL is not above top level
    dpl_above_top = (p_dpl < p[-1])
    if np.any(dpl_above_top):
        n_pts = np.count_nonzero(dpl_above_top)
        raise ValueError(f'DPL above top level at {n_pts} points')

    # Note the number of levels
    n_lev = p.shape[0]

    # Compute the parcel specific humidity at the DPL
    omega = ice_fraction(Tp_dpl)
    qp_dpl = saturation_specific_humidity(p_dpl, Tp_dpl, phase=phase,
                                          omega=omega)

    # Create arrays for DCAPE, DCIN, and the DEL
    DCAPE = np.zeros_like(p_dpl)
    DCIN = np.zeros_like(p_dpl)
    DEL = np.full_like(p_dpl, np.nan)  # undefined where DCIN = 0

    # Initialise level 2 environmental fields
    if k_dpl is None:
        k_start = n_lev - 2
        p2 = p[-1]
        T2 = T[-1]
        q2 = q[-1]
    else:
        k_start = k_dpl - 1
        p2 = p[k_dpl]
        T2 = T[k_dpl]
        q2 = q[k_dpl]

    # Initialise level 2 parcel properties using DPL values
    Tp2 = Tp_dpl.copy()
    qp2 = qp_dpl.copy()

    # Initialise parcel buoyancy (virtual temperature excess) at level 2
    B2 = virtual_temperature(Tp2, qp2) - virtual_temperature(T2, q2)

    #print(p_dpl, Tp_dpl, qp_dpl)
    #print(p2, T2, q2, Tp2, qp2, B2)

    # Loop downward over levels
    for k in range(k_start, k_stop-1, -1):

        # Update level 1 fields
        p1 = p2.copy()
        T1 = T2.copy()
        q1 = q2.copy()
        Tp1 = Tp2.copy()
        qp1 = qp2.copy()
        B1 = B2.copy()

        # If all points are below the surface, break out of loop
        p1_below_sfc = (p1 >= p_sfc)
        if np.all(p1_below_sfc):
            break

        # Set level 2 environmental fields
        if k > k_stop:
            pk_above_sfc = (p[k] < p_sfc)
            p2 = np.where(pk_above_sfc, p[k], p_sfc)
            T2 = np.where(pk_above_sfc, T[k], T_sfc)
            q2 = np.where(pk_above_sfc, q[k], q_sfc)
        else:
            p2 = p_sfc
            T2 = T_sfc
            q2 = q_sfc

        # Find level 2 points below and above the DPL
        p2_below_dpl = (p2 > p_dpl)
        p2_above_dpl = np.logical_not(p2_below_dpl)

        # If all level 2 points are at/above the DPL, skip this level
        if np.all(p2_above_dpl):
            continue

        # If crossing the DPL, reset level 1 as the DPL
        cross_dpl = (p1 < p_dpl) & (p2 > p_dpl)
        if np.any(cross_dpl):

            #print('Crossing DPL', p1, p2, p_dpl)

            # Interpolate to get environmental temperature and specific
            # humidity at the DPL
            weight = np.log(p_dpl[cross_dpl] / p1[cross_dpl]) / \
                np.log(p2[cross_dpl] / p1[cross_dpl])
            T1[cross_dpl] = (1 - weight) * T1[cross_dpl] + \
                weight * T2[cross_dpl]
            q1[cross_dpl] = (1 - weight) * q1[cross_dpl] + \
                weight * q2[cross_dpl]
                        
            # Use DPL pressure
            p1[cross_dpl] = p_dpl[cross_dpl]

        # Find level 1 points that are at the LPL
        p1_is_dpl = (p1 == p_dpl)
        if np.any(p1_is_dpl):

            # Recompute parcel buoyancy at level 1
            B1[p1_is_dpl] = virtual_temperature(
                Tp1[p1_is_dpl], qp1[p1_is_dpl]
            ) - virtual_temperature(
                T1[p1_is_dpl], q1[p1_is_dpl]
            )

        # Follow a pseudoadiabat to get parcel temperature
        Tp2[p2_below_dpl] = follow_moist_adiabat(
            p1[p2_below_dpl], p2[p2_below_dpl], Tp1[p2_below_dpl], phase=phase,
            pseudo=True, polynomial=polynomial, explicit=explicit, dp=dp
        )

        # Set parcel specific humidity equal to its value at saturation
        omega = ice_fraction(Tp2[p2_below_dpl])
        qp2[p2_below_dpl] = saturation_specific_humidity(
            p2[p2_below_dpl], Tp2[p2_below_dpl], phase=phase, omega=omega
        )

        # Compute parcel buoyancy at level 2
        B2 = virtual_temperature(Tp2, qp2) - virtual_temperature(T2, q2)

        # Find points where parcel is within negative area
        neg_to_neg = (B1 <= 0.0) & (B2 <= 0.0)
        if np.any(neg_to_neg):

            #print('In negative area', p1, p2, B1, B2)

            # Update DCAPE
            DCAPE[neg_to_neg] -= Rd * 0.5 * \
                (B1[neg_to_neg] + B2[neg_to_neg]) * \
                np.log(p2[neg_to_neg] / p1[neg_to_neg])

        # Find points where parcel is crossing from negative to positive area
        neg_to_pos = (B1 <= 0.0) & (B2 > 0.0)
        if np.any(neg_to_pos):

            #print('Crossing from negative to positive area', p1, p2, B1, B2)

            # Interpolate to get pressure at crossing level
            px = np.zeros_like(p2)
            weight = B1[neg_to_pos] / (B1[neg_to_pos] - B2[neg_to_pos])
            px[neg_to_pos] = p1[neg_to_pos] ** (1 - weight) * \
                p2[neg_to_pos] ** weight

            # Set the DEL
            DEL[neg_to_pos] = px[neg_to_pos]

            # Update DCAPE and DCIN
            DCAPE[neg_to_pos] -= Rd * 0.5 * B1[neg_to_pos] * \
                np.log(px[neg_to_pos] / p1[neg_to_pos])
            DCIN[neg_to_pos] += Rd * 0.5 * B2[neg_to_pos] * \
                np.log(p2[neg_to_pos] / px[neg_to_pos])

        # Find where parcel is within positive area
        pos_to_pos = (B1 > 0.0) & (B2 > 0.0)
        if np.any(pos_to_pos):

            #print('In positive area', p1, p2, B1, B2)

            # Update DCIN
            DCIN[pos_to_pos] += Rd * 0.5 * \
                (B1[pos_to_pos] + B2[pos_to_pos]) * \
                np.log(p2[pos_to_pos] / p1[pos_to_pos])

        # Find points where parcel is crossing from positive to negative area
        pos_to_neg = (B1 > 0.0) & (B2 <= 0.0)
        if np.any(pos_to_neg):

            #print('Crossing from positive to negative area', p1, p2, B1, B2)

            # Interpolate to get pressure at crossing level
            px = np.zeros_like(p2)
            weight = B1[pos_to_neg] / (B1[pos_to_neg] - B2[pos_to_neg])
            px[pos_to_neg] = p1[pos_to_neg] ** (1 - weight) * \
                p2[pos_to_neg] ** weight

            # Update DCAPE and DCIN
            DCIN[pos_to_neg] += Rd * 0.5 * B1[pos_to_neg] * \
                np.log(px[pos_to_neg] / p1[pos_to_neg])
            DCAPE[pos_to_neg] -= Rd * 0.5 * B2[pos_to_neg] * \
                np.log(p2[pos_to_neg] / px[pos_to_neg])

        # If positively buoyant at DPL then set DEL = DPL
        # (this can happen due to small errors in Tw calculation)
        pos_at_dpl = (p1 == p_dpl) & (B1 > 0.0)
        if np.any(pos_at_dpl):
            DEL[pos_at_dpl] = p_dpl[pos_at_dpl]

        # Reset DCAPE, DCIN, and the DEL where p2 is above the DPL
        DCAPE[p2_above_dpl] = 0.0
        DCIN[p2_above_dpl] = 0.0
        DEL[p2_above_dpl] = np.nan

        #print(k, p2, T2, q2, Tp2, qp2)
        #print(k, p1, p2, B1, B2, DCAPE, DCIN)

    # Note the final downdraft parcel temperature
    Tpf = Tp2

    if output_scalars:
        # convert outputs to scalars
        DCAPE = DCAPE.squeeze()
        DCIN = DCIN.squeeze()
        DEL = DEL.squeeze()
        Tpf = Tpf.squeeze()

    return DCAPE, DCIN, DEL, Tpf


def downdraft_parcel(p, T, q, p_sfc=None, T_sfc=None, q_sfc=None,
                     thw=None, thw_sfc=None, p_bot=None, p_top=None,
                     vertical_axis=0, return_parcel_properties=False,
                     **kwargs):
    """
    Performs a downdraft parcel descent and returns the resulting downdraft
    convective available potential energy (DCAPE) and downdraft convective
    inhibition (DCIN), along with the downdraft parcel level (DPL) and
    downdraft equilibrium level (DEL). The DPL is defined as the level with the
    lowest wet-bulb potential temperature in either the lowest 400 hPa above
    the surface (by default) or between two specified levels (p_bot and p_top).
    The parcel temperature at the DPL is set as the environmental wet-bulb
    temperature at this level. The DEL is defined as the last (lowest) level at
    which the downdraft parcel becomes positively buoyant.

    Args:
        p (ndarray): pressure profile(s) (Pa)
        T (ndarray): temperature profile(s) (K)
        q (ndarray): specific humidity profile(s) (kg/kg)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        T_sfc (float or ndarray, optional): surface temperature (K)
        q_sfc (float or ndarray, optional): surface specific humidity (kg/kg)
        thw (float or ndarray, optional): wet-bulb potential temperature (K)
        thw_sfc (float or ndarray, optional): surface wet-bulb potential
            temperature (K)
        p_bot (float or ndarray, optional): pressure of bottom of layer (Pa)
        p_top (float or ndarray, optional): pressure of top of layer (Pa)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)
        return_parcel_properties (bool, optional): flag indicating whether to
            return parcel temperature and specific humidity (default is False)
        **kwargs: additional keyword arguments passed to parcel_descent (valid
            arguments are 'phase', 'polynomial', 'explicit', and 'dp')

    Returns:
        DCAPE (float or ndarray): downdraft convective available potential
            energy (J/kg)
        DCIN (float or ndarray): downdraft convective inhibition (J/kg)
        DPL (float or ndarray): downdraft parcel level (Pa)
        DEL (float or ndarray): downdraft equilibrium level (Pa)
        Tpi (float or ndarray): initial downdraft parcel temperature (K)
        Tpf (float or ndarray): final downdraft parcel temperature (K)

    """

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        p = np.moveaxis(p, vertical_axis, 0)
        T = np.moveaxis(T, vertical_axis, 0)
        q = np.moveaxis(q, vertical_axis, 0)

    if p_bot is None:
        if p_sfc is None:
            p_bot = p[0]
        else:
            p_bot = p_sfc

    if p_top is None:
        if p_sfc is None:
            p_top = p[0] - 40000.0  # 400 hPa above lowest level
        else:
            p_top = p_sfc - 40000.0  # 400 hPa above surface

    if thw is None:

        # Compute wet-bulb potential temperature (WBPT)
        thw = wet_bulb_potential_temperature(p, T, q, **kwargs)
        if p_sfc is None:
            thw_sfc = None
        else:
            thw_sfc= wet_bulb_potential_temperature(p_sfc, T_sfc, q_sfc, **kwargs)

    # Find the level corresponding to the minimum WBPT between p_bot and p_top
    # and set this as the DPL
    thw_dpl, p_dpl = pressure_layer_maxmin_scalar(
        p, thw, p_bot, p_top, p_sfc=p_sfc, s_sfc=thw_sfc, statistic='min'
    )

    # Interpolate to get the environmental temperature and specific humidity at
    # the DPL
    T_dpl = interpolate_scalar_to_pressure_level(
        p, T, p_dpl, p_sfc=p_sfc, s_sfc=T_sfc,
    )
    q_dpl = interpolate_scalar_to_pressure_level(
        p, q, p_dpl, p_sfc=p_sfc, s_sfc=q_sfc,
    )

    # Set the parcel temperature at the DPL as the environmental wet-bulb
    # temperature at this level
    Tp_dpl = pseudo_wet_bulb_temperature(p_dpl, T_dpl, q_dpl, **kwargs)

    # Note the DPL and initial downdraft parcel temperature
    DPL, Tpi = p_dpl, Tp_dpl

    # Call code to perform parcel descent
    DCAPE, DCIN, DEL, Tpf = parcel_descent(
        p, T, q, p_dpl, Tp_dpl,
        p_sfc=p_sfc, T_sfc=T_sfc, q_sfc=q_sfc,
        **kwargs
    )

    if return_parcel_properties:
        return DCAPE, DCIN, DPL, DEL, Tpi, Tpf
    else:
        return DCAPE, DCIN, DPL, DEL


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
    if Ti.size > 1:
        # multiple initial temperature values
        if pi.size == 1:
            # single initial pressure value
            pi = np.full_like(Ti, pi)
        if pf.size == 1:
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
    Tpf = np.full_like(pi, np.nan)
    qpf = np.full_like(pi, np.nan)

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

    return LI.squeeze()
