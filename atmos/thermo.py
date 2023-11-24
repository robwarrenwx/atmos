"""
References:

Ambaum, M.H., 2020. Accurate, simple equation for saturated vapour
    pressure over water and ice. Quart. J. Roy. Met. Soc., 146, 4252-4258,
    https://doi.org/10.1002/qj.3899.

Bakhshaii, A., and R. Stull, 2013: Saturated Pseudoadiabats - A Noniterative
    Approximation.  J. Appl. Meteor. Climatol., 52, 5-15,
    https://doi.org/10.1175/JAMC-D-12-062.1.

Romps, D.M., 2017. Exact expression for the lifting condensation level.
    J. Atmos. Sci., 74, 3033-3057, https://doi.org/10.1175/JAS-D-17-0102.1.
        
Romps, D.M., 2021. Accurate expressions for the dewpoint and frost point
    derived from the Rankine-Kirchoff approximations. J. Atmos. Sci., 78,
    2113-2116, https://doi.org/10.1175/JAS-D-20-0301.1.

"""


import numpy as np
from scipy.special import lambertw
from atmos.constant import (Rd, Rv, eps, cpd, cpv, gamma, cpl, cpi,
                            T0, es0, Lv0, Lf0, Ls0, Tliq, Tice)
import atmos.pseudoadiabat as pseudoadiabat


def effective_gas_constant(q):
    """
    Computes effective gas constant for moist air.

    Args:
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        Rm (float or ndarray): effective gas constant (J/kg/K)

    """
    Rm = (1 - q) * Rd + q * Rv

    return Rm


def effective_specific_heat(q):
    """
    Computes effective isobaric specific heat for moist air (J/kg/K).

    Args:
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        cpm (float or ndarray): effective isobaric specific heat (J/kg/K)

    """
    cpm = (1 - q) * cpd + q * cpv

    return cpm


def latent_heat_of_vaporisation(T):
    """
    Computes latent heat of vaporisation for a given temperature.

    Args:
        T (float or ndarray): temperature (K)

    Returns:
        Lv (float or ndarray): latent heat of vaporisation (J/kg)

    """
    Lv = Lv0 - (cpl - cpv) * (T - T0)

    return Lv


def latent_heat_of_freezing(T):
    """
    Computes latent heat of freezing for a given temperature.

    Args:
        T (float or ndarray): temperature (K)

    Returns:
        Lf (float or ndarray): latent heat of freezing (J/kg)

    """
    Lf = Lf0 - (cpi - cpl) * (T - T0)

    return Lf


def latent_heat_of_sublimation(T):
    """
    Computes latent heat of sublimation for a given temperature.

    Args:
        T (float or ndarray): temperature (K)

    Returns:
        Ls (float or ndarray): latent heat of sublimation (J/kg)

    """
    Ls = Ls0 - (cpi - cpv) * (T - T0)

    return Ls


def air_density(p, T, q):
    """
    Computes density of air using the ideal gas equation.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        rho (float or ndarray): air density (kg/m3)

    """
    Rm = effective_gas_constant(q)
    rho = p / (Rm * T)

    return rho


def dry_air_density(p, T, q):
    """
    Computes density of dry air using the ideal gas equation.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        rhod (float or ndarray): dry air density (kg/m3)

    """
    rhod = (1 - q) * air_density(p, T, q)

    return rhod


def virtual_temperature(T, q):
    """
    Computes virtual temperature.

    Args:
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        Tv (float or ndarray): virtual temperature (K)

    """
    Tv = T * (1 + (1/eps - 1) * q)

    return Tv


def mixing_ratio(q):
    """
    Computes water vapour mixing ratio from specific humidity.

    Args:
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """
    r = q / (1 - q)

    return r


def vapour_pressure(p, q):
    """
    Computes vapour pressure from pressure and specific humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    e = p * q / ((1 - eps) * q + eps)

    return e


def saturation_vapour_pressure(T, phase='liquid', omega=0.0):
    """
    Computes saturation vapour pressure (SVP) over liquid, ice, or mixed-phase
    water for a given temperature using equations from Ambaum (2020).

    Args:
        T (float or ndarray): temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        es (float or ndarray): saturation vapour pressure (Pa)

    """
   
    if phase == 'liquid':
        
        # Compute latent heat of vaporisation
        Lv = latent_heat_of_vaporisation(T)

        # Compute SVP over liquid water (Ambaum 2020, Eq. 13)
        es = es0 * np.power((T0 / T), ((cpl - cpv) / Rv)) * \
            np.exp((Lv0 / (Rv * T0)) - (Lv / (Rv * T)))
        
    elif phase == 'ice':
        
        # Compute latent heat of sublimation
        Ls = latent_heat_of_sublimation(T)

        # Compute SVP over ice (Ambaum 2020, Eq. 17)
        es = es0 * np.power((T0 / T), ((cpi - cpv) / Rv)) * \
            np.exp((Ls0 / (Rv * T0)) - (Ls / (Rv * T)))
        
    elif phase == 'mixed':
        
        # Compute mixed-phase specific heat
        cpx = (1 - omega) * cpl + omega * cpi
        
        # Compute mixed-phase latent heat at the triple point
        Lx0 = (1 - omega) * Lv0 + omega * Ls0

        # Compute mixed-phase latent heat
        Lx = Lx0 - (cpx - cpv) * (T - T0)
        
        # Compute mixed-phase SVP
        es = es0 * np.power((T0 / T), ((cpx - cpv) / Rv)) * \
            np.exp((Lx0 / (Rv * T0)) - (Lx / (Rv * T)))
        
    else:

        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    return es


def saturation_specific_humidity(p, T, phase='liquid', omega=0.0):
    """
    Computes saturation specific humidity from pressure and temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        qs (float or ndarray): saturation specific humidity (kg/kg)

    """
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    qs = eps * es / (p - (1 - eps) * es)

    return qs


def saturation_mixing_ratio(p, T, phase='liquid', omega=0.0):
    """
    Computes saturation mixing ratio from pressure and temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        rs (float or ndarray): saturation mixing ratio (kg/kg)

    """
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    rs = eps * es / (p - es)

    return rs


def relative_humidity(p, T, q, phase='liquid', omega=0.0):
    """
    Computes relative humidity with respect to specified phase from pressure, 
    temperature, and specific humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation 
            (default is 0.0)
        
    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    e = vapour_pressure(p, q)
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    RH = e / es

    return RH


def dewpoint_temperature(p, T, q):
    """
    Computes dewpoint temperature from pressure, temperature, and specific
    humidity using equations from Romps (2021).

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """

    # Compute relative humidity over liquid water
    RH = relative_humidity(p, T, q, phase='liquid')
    #RH = np.minimum(RH, 1.0)  # limit RH to 100 %

    # Set constant (Romps 2021, Eq. 6)
    c = (Lv0 - (cpv - cpl) * T0) / ((cpv - cpl) * T)

    # Compute dewpoint temperature (Romps 2021, Eq. 5)
    fn = np.power(RH, (Rv / (cpl - cpv))) * c * np.exp(c)
    W = lambertw(fn, k=-1).real
    Td = c * (1 / W) * T
    
    # Ensure that Td does not exceed T
    Td = np.minimum(Td, T)

    return Td


def frost_point_temperature(p, T, q):
    """
    Computes frost-point temperature from pressure, temperature, and specific
    humidity using equations from Romps (2021).

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        Tf (float or ndarray): frost-point temperature (K)

    """    
    # Compute relative humidity over ice
    RH = relative_humidity(p, T, q, phase='ice')
    #RH = np.minimum(RH, 1.0)  # limit RH to 100 %

    # Set constant (Romps 2021, Eq. 8)
    c = (Ls0 - (cpv - cpi) * T0) / ((cpv - cpi) * T)

    # Compute frost-point temperature (Romps 2021, Eq. 7)
    fn = np.power(RH, (Rv / (cpi - cpv))) * c * np.exp(c)
    W = lambertw(fn, k=-1).real  # -1 branch because cpi > cpv
    Tf = c * (1 / W) * T
    
    # Ensure that Tf does not exceed T
    Tf = np.minimum(Tf, T)

    return Tf


def saturation_point_temperature(p, T, q, converged=0.001):
    """
    Computes saturation-point temperature from pressure, temperature, and 
    specific humidity using equations similar to Romps (2021).

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        converged (float, optional): target precision for saturation-point
            temperature (default is 0.001 K)

    Returns:
        Ts (float or ndarray): saturation-point temperature (K)

    """

    # Intialise the saturation point temperature as the temperature
    Ts = T.copy()

    # Iterate to convergence
    count = 0
    delta = np.full_like(T, 10)
    while np.max(delta) > converged:

        # Update the previous Ts value
        Ts_prev = Ts

        # Compute the ice fraction
        omega = ice_fraction(Ts)

        # Compute the mixed-phase relative humidity
        RH = relative_humidity(p, T, q, phase='mixed', omega=omega)
        #RH = np.minimum(RH, 1.0)  # limit RH to 100 %

        # Compute mixed-phase specific heat
        cpx = (1 - omega) * cpl + omega * cpi

        # Compute mixed-phase latent heat at the triple point
        Lx0 = (1 - omega) * Lv0 + omega * Ls0

        # Set constant (cf. Romps 2021, Eq. 6 and 8)
        c = (Lx0 - (cpv - cpx) * T0) / ((cpv - cpx) * T)

        # Compute saturation-point temperature (cf. Romps 2021, Eq. 5 and 7)
        fn = np.power(RH, (Rv / (cpx - cpv))) * c * np.exp(c)
        W = lambertw(fn, k=-1).real
        Ts = c * (1 / W) * T
   
        # Check if solution has converged
        delta = np.abs(Ts - Ts_prev)
        count += 1
        if count > 20:
            print("T_lsl not converged after 20 iterations")
            break

    # Ensure that Ts does not exceed T
    Ts = np.minimum(Ts, T)

    return Ts


def lifting_condensation_level(p, T, q):
    """
    Computes pressure and temperature at the lifted condensation level (LCL)
    using equations from Romps (2017).

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        p_lcl (float or ndarray): pressure at the LCL (Pa)
        T_lcl (float or ndarray): temperature at the LCL (K)

    """
    
    # Compute effective gas constant and specific heat
    Rm = effective_gas_constant(q)
    cpm = effective_specific_heat(q)

    # Compute relative humidity with respect to liquid water
    RH = relative_humidity(p, T, q, phase='liquid')
    #RH = np.minimum(RH, 1.0)  # limit RH to 100 %
    
    # Set constants (Romps 2017, Eq. 22d-f)
    a = cpm / Rm + (cpl - cpv) / Rv
    b = -(Lv0 + (cpl - cpv) * T0) / (Rv * T)
    c = b / a

    # Compute temperature at the LCL (Romps 2017, Eq. 22a)
    fn = np.power(RH, (1 / a)) * c * np.exp(c)
    W = lambertw(fn, k=-1).real
    T_lcl = c * (1 / W) * T
    
    # Compute pressure at the LCL (Romps 2017, Eq. 22b)
    p_lcl = p * np.power((T_lcl / T), (cpm / Rm))
    
    # Ensure that LCL temperature and pressure do not exceed initial values
    T_lcl = np.minimum(T_lcl, T)
    p_lcl = np.minimum(p_lcl, p)
    
    return p_lcl, T_lcl


def lifting_deposition_level(p, T, q):
    """
    Computes pressure and temperature at the lifting deposition level (LDL)
    using equations from Romps (2017).

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        p_ldl (float or ndarray): pressure at the LDL (Pa)
        T_ldl (float or ndarray): temperature at the LDL (K)

    """

    # Compute effective gas constant and specific heat
    Rm = effective_gas_constant(q)
    cpm = effective_specific_heat(q)

    # Compute relative humidity with respect to ice
    RH = relative_humidity(p, T, q, phase='ice')
    #RH = np.minimum(RH, 1.0)  # limit RH to 100 %
   
    # Set constants (Romps 2017, Eq. 23d-f)
    a = cpm / Rm + (cpi - cpv) / Rv
    b = -(Ls0 + (cpi - cpv) * T0) / (Rv * T)
    c = b / a

    # Compute temperature at the LDL (Romps 2017, Eq. 23a)
    fn = np.power(RH, (1 / a)) * c * np.exp(c)
    W = lambertw(fn, k=-1).real
    T_ldl = c * (1 / W) * T
    
    # Compute pressure at the LDL (Romps 2017, Eq. 23b)
    p_ldl = p * np.power((T_ldl / T), (cpm / Rm))
    
    # Ensure that LDL temperature and pressure do not exceed initial values
    T_ldl = np.minimum(T_ldl, T)
    p_ldl = np.minimum(p_ldl, p)
    
    return p_ldl, T_ldl


def lifting_saturation_level(p, T, q, converged=0.001):
    """
    Computes pressure and temperature at the lifting saturation level (LSL)
    using equations similar to Romps (2017).

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        converged (float, optional): target precision for LSL temperature
            (default is 0.001 K)

    Returns:
        p_lsl (float or ndarray): pressure at the LSL (Pa)
        T_lsl (float or ndarray): temperature at the LSL (K)

    """

    # Set the initial temperature at the LSL
    T_lsl = T.copy()

    # Iterate to convergence
    count = 0
    delta = np.full_like(T, 10)
    while np.max(delta) > converged:

        # Update the previous Tstar value
        T_lsl_prev = T_lsl

        # Compute the ice fraction
        omega = ice_fraction(T_lsl)

        # Compute effective gas constant and specific heat
        Rm = effective_gas_constant(q)
        cpm = effective_specific_heat(q)

        # Compute mixed-phase relative humidity
        RH = relative_humidity(p, T, q, phase='mixed', omega=omega)
        #RH = np.minimum(RH, 1.0)  # limit RH to 100 %

        # Compute mixed-phase specific heat
        cpx = (1 - omega) * cpl + omega * cpi

        # Compute mixed-phase latent heat at the triple point
        Lx0 = (1 - omega) * Lv0 + omega * Ls0
        
        # Set constants (cf. Romps 2017, Eq. 22d-f and 23d-f)
        a = cpm / Rm + (cpx - cpv) / Rv
        b = -(Lx0 + (cpx - cpv) * T0) / (Rv * T)
        c = b / a

        # Compute temperature at the LSL (cf. Romps 2017, Eq. 22a and 23a)
        fn = np.power(RH, (1 / a)) * c * np.exp(c)
        W = lambertw(fn, k=-1).real
        T_lsl = c * (1 / W) * T
    
        # Check if solution has converged
        delta = np.abs(T_lsl - T_lsl_prev)
        count += 1
        if count > 20:
            print("T_lsl not converged after 20 iterations")
            break

    # Compute pressure at the LSL (cf. Romps 2017, Eq. 22b and 23b)
    p_lsl = p * np.power((T_lsl / T), (cpm / Rm))
    
    # Ensure that LSL temperature and pressure do not exceed initial values
    T_lsl = np.minimum(T_lsl, T)
    p_lsl = np.minimum(p_lsl, p)
    
    return p_lsl, T_lsl


def ice_fraction(Tstar):
    """
    Computes ice fraction given temperature at saturation.

    Args:
        Tstar (float or ndarray): temperature at saturation (K)

    Returns:
        omega (float or ndarray): ice fraction

    """

    Tstar = np.atleast_1d(Tstar)

    omega = 0.5 * (1 - np.cos(np.pi * ((Tliq - Tstar) / (Tliq - Tice))))
    omega[Tstar <= Tice] = 1.0
    omega[Tstar >= Tliq] = 0.0

    if len(Tstar) == 1:
        Tstar = Tstar[0]

    return omega


def ice_fraction_derivative(Tstar):
    """
    Computes derivative of ice fraction with respect to temperature at
    saturation.
    
    Args:
        Tstar (float or ndarray): temperature at saturation (K)

    Returns:
        domega_dTstar (float or ndarray): derivative of ice fraction (K^-1)
       
    """

    Tstar = np.atleast_1d(Tstar)

    domega_dTstar = -0.5 * (np.pi / (Tliq - Tice)) * \
            np.sin(np.pi * ((Tliq - Tstar) / (Tliq - Tice)))
    domega_dTstar[(Tstar <= Tice) | (Tstar >= Tliq)] = 0.0

    if len(Tstar) == 1:
        domega_dTstar = domega_dTstar[0]

    return domega_dTstar


def ice_fraction_at_saturation(p, T, q, saturation='isobaric', converged=0.001):
    """
    Computes ice fraction at saturation for specified saturation process.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        saturation (str, optional): saturation process (valid options are
            'isobaric' or 'adiabatic'; default is 'isobaric')
        converged (float, optional): target precision for temperature at
            saturation (default is 0.001 K)

    Returns:
        omega (float or ndarray): ice fraction at saturation

    """

    if saturation == 'isobaric':

        # Compute saturation-point temperature
        Tstar = saturation_point_temperature(p, T, q, converged=converged)

    elif saturation == 'adiabatic':

        # Compute lifting saturation level (LSL) temperature
        _, Tstar = lifting_saturation_level(p, T, q, converged=converged)

    else:

        raise ValueError("saturation must be 'isobaric' or 'adiabatic'")

    # Compute the ice fraction
    omega = ice_fraction(Tstar)

    return omega


def dry_adiabatic_lapse_rate(p, T, q):
    """
    Computes dry adiabatic lapse rate in pressure coordinates.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        dT_dp (float or ndarray): dry adiabatic lapse rate (K/Pa)

    """

    # Compute factor b (Bakhshaii and Stull 2013, Eq. 2)
    b = (1 - q + q / eps) / (1 - q + q / gamma)

    # Compute dry adiabatic lapse rate (Bakshaii and Stull 2013, Eq. 3)
    dT_dp = b * (Rd * T) / (cpd * p)

    return dT_dp


def pseudoadiabatic_lapse_rate(p, T, phase='liquid'):
    """
    Computes pseudoadiabatic lapse rate in pressure coordinates.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')

    Returns:
        dT_dp (float or ndarray): pseudoadiabatic lapse rate (K/Pa)

    """

    if phase == 'liquid':

        # Compute saturation specific humidity with respect to liquid water
        qs = saturation_specific_humidity(p, T, phase='liquid')

        # Compute latent heat of vaporisation
        Lv = latent_heat_of_vaporisation(T)

        # Compute factor b (Bakhshaii and Stull 2013, Eq. 2)
        b = (1 - qs + qs / eps) / (1 - qs + qs / gamma)

        # Compute liquid pseudoadiabatic lapse rate (Bakshaii and Stull 2013,
        # Eq. 8)
        dT_dp = (b / p) * (Rd * T + Lv * qs) / \
            (cpd + b * (Lv**2 * qs) / (Rv * T**2))

    elif phase == 'ice':

        # Compute saturation specific humidity with respect to ice
        qs = saturation_specific_humidity(p, T, phase='ice')

        # Compute latent heat of sublimation
        Ls = latent_heat_of_sublimation(T)

        # Compute factor b (Bakhshaii and Stull 2013, Eq. 2)
        b = (1 - qs + qs / eps) / (1 - qs + qs / gamma)

        # Compute ice pseudoadiabatic lapse rate (cf. Bakshaii and Stull 2013,
        # Eq. 8)
        dT_dp = (b / p) * (Rd * T + Ls * qs) / \
            (cpd + b * (Ls**2 * qs) / (Rv * T**2))

    elif phase == 'mixed':

        # Compute ice fraction, omega
        omega = ice_fraction(T)

        # Compute the derivative of omega with respect to temperature
        domega_dT = ice_fraction_derivative(T)

        # Compute mixed-phase saturation specific humidity
        qs = saturation_specific_humidity(p, T, phase='mixed', omega=omega)

        # Compute factor b (Bakhshaii and Stull 2013, Eq. 2)
        b = (1 - qs + qs / eps) / (1 - qs + qs / gamma)

        # Compute mixed-phase latent heat
        cpx = (1 - omega) * cpl + omega * cpi
        Lx0 = (1 - omega) * Lv0 + omega * Ls0
        Lx = Lx0 - (cpx - cpv) * (T - T0)

        # Compute saturation vapour pressues over liquid and ice
        esl = saturation_vapour_pressure(T, phase='liquid')
        esi = saturation_vapour_pressure(T, phase='ice')

        # Compute mixed-phase pseudoadiabatic lapse rate
        dT_dp = (b / p) * (Rd * T + Lx * qs) / \
            (cpd + b * (Lx**2 * qs) / (Rv * T**2) +
             b * Lx * qs * np.log(esi / esl) * domega_dT)

    return dT_dp


def follow_dry_adiabat(pi, pf, Ti, q):
    """
    Computes parcel temperature following a dry adiabat.

    Args:
        pi (float or ndarray): initial pressure (Pa)
        pf (float or ndarray): final pressure (Pa)
        Ti (float or ndarray): initial temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        Tf (float or ndarray): final temperature (K)

    """

    # Set effective gas constant and specific heat
    Rm = effective_gas_constant(q)
    cpm = effective_specific_heat(q)

    # Compute new temperature
    Tf = Ti * np.power((pf / pi), (Rm / cpm))

    return Tf


def follow_pseudoadiabat(pi, pf, Ti, phase='liquid', method='polynomial',
                         pinc=500.0, converged=0.001):
    """
    Computes parcel temperature following a pseudoadiabat. By default, uses
    polynomial fits to pseudoadiabats for fast calculations. Option to use
    slower iterative calculation.

    Args:
        pi (float or ndarray): initial pressure (Pa)
        pf (float or ndarray): final pressure (Pa)
        Ti (float or ndarray): initial temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        method (str, optional): method for computing parcel temperature (valid
            options are 'polynomial' or 'iterative')
        pinc (float, optional): pressure increment for iterative calculation
            (default is 500 Pa = 5 hPa)
        converged (float, optional): target precision for iterative solution
            (default is 0.001 K)

    Returns:
        Tf (float or ndarray): final temperature (K)

    """

    if method == 'polynomial':

        # Compute the wet-bulb potential temperature of the pseudoadiabat
        # that passes through (pi, Ti)
        thw = pseudoadiabat.wbpt(pi, Ti)

        # Compute the temperature on this pseudoadiabat at pf
        Tf = pseudoadiabat.temp(pf, thw)

    elif method == 'iterative':

        pi = np.atleast_1d(pi)
        pf = np.atleast_1d(pf)
        Ti = np.atleast_1d(Ti)

        # Set the pressure increment based on whether the parcel is ascending
        # or descending
        pinc = np.abs(pinc)  # make sure pinc is positive
        ascending = (pf < pi)
        descending = np.logical_not(ascending)
        dp = np.where(ascending, -pinc, pinc)

        # Initialise the pressure and temperature at level 2
        p2 = np.copy(pi)
        T2 = np.copy(Ti)

        #print(np.min(p2), np.max(p2))
        #print(np.count_nonzero(ascending), np.count_nonzero(descending))
        #print(np.min(dp), np.max(dp))

        # Loop over pressure increments
        while np.max(np.abs(p2 - pf)) > 0.0:

            # Set level 1 values
            p1 = p2.copy()
            T1 = T2.copy()

            # Update the pressure
            p2 = p1 + dp

            # Make sure we haven't overshot final pressure level
            if np.any(ascending):
                p2[ascending] = np.maximum(p2[ascending], pf[ascending])
            if np.any(descending):
                p2[descending] = np.minimum(p2[descending], pf[descending])

            #print(np.min(p2), np.max(p2))

            # Compute the layer-mean pressure
            pbar = np.sqrt(p1 * p2)  # average of log(p)
    
            # Initialise the temperature at level 2
            T2 = T1

            # Iterate to convergence
            delta = np.full_like(p2, 10)
            count = 0
            while np.max(delta) > converged:

                # Compute the layer-mean temperature
                Tbar = 0.5 * (T1 + T2)

                # Compute the lapse rate
                dT_dp = pseudoadiabatic_lapse_rate(pbar, Tbar, phase=phase)

                # Update the level 2 temperature
                T2_new = T1 + dT_dp * (p2 - p1)

                # Check if the solution has converged
                delta = np.abs(T2_new - T2)
                count += 1
                if count == 20:
                    # should converge in just a couple of iterations, provided
                    # pinc is not too large
                    print('Not converged after 20 iterations')
                    break

                # Update the level 2 temperature
                T2 = T2_new.copy()

            #print(count)

        # Set the final temperature
        Tf = T2

    else:

        raise ValueError("method must be one of 'polynomial' or 'iterative'")

    return Tf


def potential_temperature(p, T, q=0.0):
    """
    Computes potential temperature, optionally including moisture
    contribution to dry adiabatic lapse rate.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray, optional): specific humidity (kg/kg)

    Returns:
        th (float or ndarray): potential temperature (K)

    """

    # Follow a dry adiabat to 1000hPa reference
    th = follow_dry_adiabat(p, 100000., T, q)

    return th


def adiabatic_wet_bulb_temperature(p, T, q, phase='liquid',
                                   pseudoadiabat_method='polynomial'):
    """
    Computes adiabatic wet-bulb temperature.

    Adiabatic (or pseudo) wet-bulb temperature is the temperature of a parcel
    of air lifted adiabatically to saturation and then brought
    (pseudo)adiabatically at saturation back to its original pressure. It is
    always less than the isobaric wet-bulb temperature.

    See https://glossary.ametsoc.org/wiki/Wet-bulb_temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        pseudoadiabat_method (str, optional): method for performing pseudo-
            adiabatic descent (valid options are 'polynomial' or 'iterative')

    Returns:
        Tw (float or ndarray): adiabatic wet-bulb temperature (K)

    """

    if phase == 'liquid':

        # Get pressure and temperature at the LCL
        p_lcl, T_lcl = lifting_condensation_level(p, T, q)

        # Follow a pseudoadiabat from the LCL to the original pressure
        Tw = follow_pseudoadiabat(p_lcl, p, T_lcl, phase='liquid',
                                  method=pseudoadiabat_method)

    elif phase == 'ice':

        # Get pressure and temperature at the LDL
        p_ldl, T_ldl = lifting_deposition_level(p, T, q)

        # Follow a pseudoadiabat from the LDL to the original pressure
        Tw = follow_pseudoadiabat(p_ldl, p, T_ldl, phase='ice',
                                  method=pseudoadiabat_method)

    elif phase == 'mixed':

        # Get pressure and temperature at the LSL
        p_lsl, T_lsl = lifting_saturation_level(p, T, q)

        # Follow a pseudoadiabat from the LSL to the original pressure
        Tw = follow_pseudoadiabat(p_lsl, p, T_lsl, phase='mixed',
                                  method=pseudoadiabat_method)

    return Tw


def isobaric_wet_bulb_temperature(p, T, q, phase='liquid', converged=0.001):
    """
    Computes isobaric wet-bulb temperature.

    Isobaric wet-bulb temperature is the temperature of a parcel of air cooled
    isobarically to saturation via the evaporation of water into it, with all
    latent heat supplied by the parcel. It is always greater than the adiabatic
    wet-bulb temperature. Isobaric wet-bulb temperature is similar (but not
    identical) to the quantity measured by a wet-bulb thermometer. 

    See https://glossary.ametsoc.org/wiki/Wet-bulb_temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        converged (float, optional): target precision for iterative solution
            (default is 0.001 K)

    Returns:
        Tw (float or ndarray): isobaric wet-bulb temperature (K)

    """

    # Compute dewpoint temperature
    Td = dewpoint_temperature(p, T, q)

    # Initialise Tw as mean of T and Td
    Tw = (T + Td) / 2

    # Compute the latent heat at temperature T
    if phase == 'liquid':
        Lv_T = latent_heat_of_vaporisation(T)
    elif phase == 'ice':
        Ls_T = latent_heat_of_sublimation(T)
    elif phase == 'mixed':
        omega_T = ice_fraction(T)
        cpx_T = (1 - omega_T) * cpl + omega_T * cpi
        Lx0_T = (1 - omega_T) * Lv0 + omega_T * Ls0
        Lx_T = Lx0_T - (cpx_T - cpv) * (T - T0)

    # Iterate to convergence
    delta = np.full_like(T, 10.)
    count = 0
    while np.max(delta) > converged:

        # Update the previous Tw value
        Tw_prev = Tw

        if phase == 'liquid':

            # Compute the latent heat of vaporisation at Tw
            Lv_Tw = latent_heat_of_vaporisation(Tw)

            # Compute saturation specific humidity at Tw
            qs_Tw = saturation_specific_humidity(p, Tw, phase='liquid')

            # Compute the effective specific heat at qs(Tw)
            cpm_qs_Tw = effective_specific_heat(qs_Tw)

            # Compute the derivative of qs with respect to Tw
            dqs_dTw = qs_Tw * (1 + qs_Tw / eps - qs_Tw) * Lv_Tw / (Rv * Tw**2)

            # Compute f and f'
            f = cpm_qs_Tw * (T - Tw) - Lv_T * (qs_Tw - q)
            fprime = ((cpv - cpd) * (T - Tw) - Lv_T) * dqs_dTw - cpm_qs_Tw

        elif phase == 'ice':

            # Compute the latent heat of sublimation at Tw
            Ls_Tw = latent_heat_of_sublimation(Tw)

            # Compute saturation specific humidity at Tw
            qs_Tw = saturation_specific_humidity(p, Tw, phase='ice')

            # Compute the effective specific heat at qs(Tw)
            cpm_qs_Tw = effective_specific_heat(qs_Tw)

            # Compute the derivative of qs with respect to Tw
            dqs_dTw = qs_Tw * (1 + qs_Tw / eps - qs_Tw) * Ls_Tw / (Rv * Tw**2)

            # Compute f and f'
            f = cpm_qs_Tw * (T - Tw) - Ls_T * (qs_Tw - q)
            fprime = ((cpv - cpd) * (T - Tw) - Ls_T) * dqs_dTw - cpm_qs_Tw

        elif phase == 'mixed':

            # Compute the ice fraction and its derivative at Tw
            omega_Tw = ice_fraction(Tw)
            domega_dTw = ice_fraction_derivative(Tw)

            # Compute the mixed-phase latent heat at Tw
            cpx_Tw = (1 - omega_Tw) * cpl + omega_Tw * cpi
            Lx0_Tw = (1 - omega_Tw) * Lv0 + omega_Tw * Ls0
            Lx_Tw = Lx0_Tw - (cpx_Tw - cpv) * (Tw - T0)

            # Compute saturation specific humidity at Tw
            qs_Tw = saturation_specific_humidity(p, Tw, phase=phase,
                                                 omega=omega_Tw)

            # Compute the effective specific heat at qs(Tw)
            cpm_qs_Tw = effective_specific_heat(qs_Tw)

            # Compute the saturation vapour pressues over liquid and ice at Tw
            esl_Tw = saturation_vapour_pressure(T, phase='liquid')
            esi_Tw = saturation_vapour_pressure(T, phase='ice')

            # Compute the derivative of qs with respect to Tw
            dqs_dTw = qs_Tw * (1 + qs_Tw / eps - qs_Tw) * \
                (Lx_Tw / (Rv * Tw**2) + np.log(esi_Tw / esl_Tw) * domega_dTw)

            # Compute f and f'
            f = cpm_qs_Tw * (T - Tw) - Lx_T * (qs_Tw - q)
            fprime = ((cpv - cpd) * (T - Tw) - Lx_T) * dqs_dTw - cpm_qs_Tw
       
        # Update Tw using Newton's method
        Tw = Tw - f / fprime

        # Check for convergence
        delta = np.abs(Tw - Tw_prev)
        count += 1
        if count > 20:
            print("Tw not converged after 20 iterations")
            break

    return Tw


def wet_bulb_temperature(p, T, q, saturation='adiabatic', phase='liquid',
                         pseudoadiabat_method='polynomial', converged=0.001):
    """
    Computes wet-bulb temperature for specified saturation process.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        saturation (str, optional): saturation process (valid options are
            'isobaric' or 'adiabatic'; default is 'adiabatic')
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        pseudoadiabat_method (str, optional): method for performing pseudo-
            adiabatic descent in calculation of adiabatic Tw (valid options are
            'polynomial' or 'iterative')
        converged (float, optional): target precision for iterative solution
            of isobaric Tw (default is 0.001 K)

    Returns:
        Tw: wet-bulb temperature (K)

    """

    if saturation == 'adiabatic':
        Tw = adiabatic_wet_bulb_temperature(p, T, q, phase=phase,
                                            pseudoadiabat_method=\
                                                pseudoadiabat_method)
    elif saturation == 'isobaric':
        Tw = isobaric_wet_bulb_temperature(p, T, q, phase=phase,
                                           converged=converged)
    else:
        raise ValueError("saturation must be one of 'isobaric' or 'adiabatic'")

    return Tw


def wet_bulb_potential_temperature(p, T, q):
    """
    Computes wet-bulb potential temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        thw (float or ndarray): wet-bulb potential temperature (K)

    """

    # Get pressure and parcel temperature at the LCL
    p_lcl, Tp_lcl = lifting_condensation_level(p, T, q)

    # Compute the wet-bulb potential temperature of the pseudoadiabat
    # that passes through (p_lcl, Tp_lcl)
    thw = pseudoadiabat.wbpt(p_lcl, Tp_lcl)

    return thw


def saturated_wet_bulb_potential_temperature(p, T):
    """
    Computes saturation wet-bulb potential temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)

    Returns:
        thws (float or ndarray): wet-bulb potential temperature (K)

    """

    # Compute the wet-bulb potential temperature of the pseudoadiabat
    # that passes through (p, T)
    thws = pseudoadiabat.wbpt(p, T)

    return thws
