"""
Functions for calculating the following thermodynamic variables:
* effective gas constant, Rm
* effective specific heat, cpm
* latent heat of vaporisation, Lv
* latent heat of freezing, Lf
* latent heat of sublimation, Ls
* mixed-phase latent heat, Lx
* air density, rho
* dry air density, rhod
* virtual temperature, Tv
* mixing ratio, r
* vapour pressure, e
* saturation vapour pressure, es
* saturation specific humidity, qs
* saturation mixing ratio, rs
* relative humidity, RH
* dewpoint temperature, Td
* frost-point temperature, Tf
* saturation-point temperature, Ts
* lifting condensation level temperature, T_lsl, and pressure, p_lsl
* lifting deposition level temperature, T_ldl, and pressure, p_ldl
* lifting saturation level temperature, T_lsl, and pressure, T_lsl
* dry, pseudo, and saturated adiabatic pressure lapse rates
* parcel temperature following dry, pseudo, and saturated adiabats
* pseudo and isobaric wet-bulb temperatures, Tw
* dry potential temperature, thd
* moist potential temperature, thm
* virtual potential temperature, thv
* equivalent potential temperature, theq
* ice-liquid water potential temperature, thil
* wet-bulb potential temperature, thw
* saturated wet-bulb potential temperature, thws

References:
* Ambaum, M. H., 2020: Accurate, simple equation for saturated vapour
    pressure over water and ice. Quart. J. Roy. Met. Soc., 146, 4252-4258,
    https://doi.org/10.1002/qj.3899.
* Bryan, G. H., and J. M. Fristch, 2004: A reevaluation of ice-liquid water
    potential temperature. Mon. Wea. Rev., 132, 2421-2431,
    https://doi.org/10.1175/1520-0493(2004)132<2421:AROIWP>2.0.CO;2.
* Knox, J. A., D. S. Nevius, and P. N. Knox, 2017: Two simple and accurate
    approximations for wet-bulb temperature in moist conditions, with
    forecasting applications. Bull. Amer. Meteor. Soc., 98, 1897-1906,
    https://doi.org/10.1175/BAMS-D-16-0246.1.
* Romps, D. M., 2017: Exact expression for the lifting condensation level.
    J. Atmos. Sci., 74, 3033-3057, https://doi.org/10.1175/JAS-D-17-0102.1.
* Romps, D. M., 2021: Accurate expressions for the dewpoint and frost point
    derived from the Rankine-Kirchoff approximations. J. Atmos. Sci., 78,
    2113-2116, https://doi.org/10.1175/JAS-D-20-0301.1.

"""


import numpy as np
from scipy.special import lambertw
from atmos.constant import (g, Rd, Rv, eps, cpd, cpv, cpl, cpi, p_ref,
                            T0, es0, Lv0, Lf0, Ls0, T_liq, T_ice)
import atmos.pseudoadiabat as pseudoadiabat

# Precision for iterative temperature calculations (K)
precision = 0.001

# Maximum number of iterations for iterative calculations
max_n_iter = 20


def effective_gas_constant(q, qt=None):
    """
    Computes effective gas constant for moist air.

    Args:
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)

    Returns:
        Rm (float or ndarray): effective gas constant (J/kg/K)

    """
    if qt is None:
        Rm = (1 - q) * Rd + q * Rv
    else:
        Rm = (1 - qt) * Rd + q * Rv

    return Rm


def effective_specific_heat(q, qt=None, omega=0.0):
    """
    Computes effective isobaric specific heat for moist air.

    Args:
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        omega (float or ndarray, optional): ice fraction

    Returns:
        cpm (float or ndarray): effective isobaric specific heat (J/kg/K)

    """
    if qt is None:
        cpm = (1 - q) * cpd + q * cpv
    else:
        ql = (1 - omega) * (qt - q)  # liquid water mass fraction
        qi = omega * (qt - q)        # ice water mass fraction
        cpm = (1 - qt) * cpd + q * cpv + ql * cpl + qi * cpi

    return cpm


def latent_heat_of_vaporisation(T):
    """
    Computes latent heat of vaporisation for a given temperature.

    Args:
        T (float or ndarray): temperature (K)

    Returns:
        Lv (float or ndarray): latent heat of vaporisation (J/kg)

    """
    Lv = Lv0 + (cpv - cpl) * (T - T0)

    return Lv


def latent_heat_of_freezing(T):
    """
    Computes latent heat of freezing for a given temperature.

    Args:
        T (float or ndarray): temperature (K)

    Returns:
        Lf (float or ndarray): latent heat of freezing (J/kg)

    """
    Lf = Lf0 + (cpl - cpi) * (T - T0)

    return Lf


def latent_heat_of_sublimation(T):
    """
    Computes latent heat of sublimation for a given temperature.

    Args:
        T (float or ndarray): temperature (K)

    Returns:
        Ls (float or ndarray): latent heat of sublimation (J/kg)

    """
    Ls = Ls0 + (cpv - cpi) * (T - T0)

    return Ls


def mixed_phase_latent_heat(T, omega):
    """
    Computes mixed-phase latent heat for a given temperature and ice fraction.

    Args:
        T (float or ndarray): temperature (K)
        omega (float or ndarray): ice fraction

    Returns:
        Ls (float or ndarray): latent heat of sublimation (J/kg)

    """
    cpx = (1 - omega) * cpl + omega * cpi
    Lx0 = (1 - omega) * Lv0 + omega * Ls0  # = Lv0 + omega * Lf0
    Lx = Lx0 + (cpv - cpx) * (T - T0)  # = (1 - omega) * Lv + omega * Ls
                                       # = Lv + omega * Lf

    return Lx


def air_density(p, T, q, qt=None):
    """
    Computes density of air using the ideal gas equation.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)

    Returns:
        rho (float or ndarray): air density (kg/m3)

    """
    Rm = effective_gas_constant(q, qt=qt)
    rho = p / (Rm * T)

    return rho


def dry_air_density(p, T, q, qt=None):
    """
    Computes density of dry air using the ideal gas equation.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)

    Returns:
        rhod (float or ndarray): dry air density (kg/m3)

    """
    rho = air_density(p, T, q, qt=qt)
    if qt is None:
        rhod = (1 - q) * rho
    else:
        rhod = (1 - qt) * rho

    return rhod


def virtual_temperature(T, q, qt=None):
    """
    Computes virtual (or density) temperature.

    Args:
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)

    Returns:
        Tv (float or ndarray): virtual (or density) temperature (K)

    """
    if qt is None:
        Tv = T * (1 - q + q / eps)  # virtual temperature
    else:
        Tv = T * (1 - qt + q / eps)  # density temperature

    return Tv


def mixing_ratio(q, qt=None):
    """
    Computes mixing ratio from specific humidity.

    Args:
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """
    if qt is None:
        r = q / (1 - q)
    else:
        r = q / (1 - qt)

    return r


def vapour_pressure(p, q, qt=None):
    """
    Computes vapour pressure from pressure and specific humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    if qt is None:
        e = p * q / (eps * (1 - q) + q)
    else:
        e = p * q / (eps * (1 - qt) + q)

    return e


def saturation_vapour_pressure(T, phase='liquid', omega=0.0):
    """
    Computes saturation vapour pressure (SVP) for a given temperature using
    equations from Ambaum (2020).

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


def saturation_specific_humidity(p, T, qt=None, phase='liquid', omega=0.0):
    """
    Computes saturation specific humidity from pressure and temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        qs (float or ndarray): saturation specific humidity (kg/kg)

    """
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    if qt is None:
        qs = eps * es / (p - (1 - eps) * es)
    else:
        qs  = (1 - qt) * eps * es / (p - es)

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


def relative_humidity(p, T, q, qt=None, phase='liquid', omega=0.0):
    """
    Computes relative humidity with respect to specified phase from pressure, 
    temperature, and specific humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation 
            (default is 0.0)
        
    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    e = vapour_pressure(p, q, qt=qt)
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    RH = e / es

    return RH


def _dewpoint_temperature_from_relative_humidity(T, RH):
    """
    Computes dewpoint temperature from temperature and relative humidity over
    liquid water using equations from Romps (2021).

    Args:
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """

    # Set constant (Romps 2021, Eq. 6)
    c = (Lv0 - (cpv - cpl) * T0) / ((cpv - cpl) * T)

    # Compute dewpoint temperature (Romps 2021, Eq. 5)
    fn = np.power(RH, (Rv / (cpl - cpv))) * c * np.exp(c)
    W = lambertw(fn, k=-1).real
    Td = c * (1 / W) * T

    return Td


def dewpoint_temperature(p, T, q):
    """
    Computes dewpoint temperature from pressure, temperature, and specific
    humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """

    # Compute relative humidity over liquid water
    RH = relative_humidity(p, T, q, phase='liquid')
    
    # Compute dewpoint temperature
    Td = _dewpoint_temperature_from_relative_humidity(T, RH)

    return Td


def _frost_point_temperature_from_relative_humidity(T, RH):
    """
    Computes frost-point temperature from temperature and relative humidity
    over ice using equations from Romps (2021).

    Args:
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)

    Returns:
        Tf (float or ndarray): frost-point temperature (K)

    """

    # Set constant (Romps 2021, Eq. 8)
    c = (Ls0 - (cpv - cpi) * T0) / ((cpv - cpi) * T)

    # Compute frost-point temperature (Romps 2021, Eq. 7)
    fn = np.power(RH, (Rv / (cpi - cpv))) * c * np.exp(c)
    W = lambertw(fn, k=-1).real  # -1 branch because cpi > cpv
    Tf = c * (1 / W) * T

    return Tf


def frost_point_temperature(p, T, q):
    """
    Computes frost-point temperature from pressure, temperature, and specific
    humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        Tf (float or ndarray): frost-point temperature (K)

    """    
    # Compute relative humidity over ice
    RH = relative_humidity(p, T, q, phase='ice')

    # Compute frost-point temperature
    Tf = _frost_point_temperature_from_relative_humidity(T, RH)

    return Tf


def _saturation_point_temperature_from_relative_humidity(T, RH, omega):
    """
    Computes saturation-point temperature from temperature, mixed-phase
    relative humidity, and ice fraction at saturation using equations similar
    to Romps (2021).

    Args:
        T (float or ndarray): temperature (K)
        RH (float or ndarray): mixed-phase relative humidity (fraction)
        omega (float or ndarray): ice fraction at saturation

    Returns:
        Ts (float or ndarray): saturation-point temperature (K)

    """

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

    return Ts


def saturation_point_temperature(p, T, q):
    """
    Computes saturation-point temperature from pressure, temperature, and 
    specific humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        Ts (float or ndarray): saturation-point temperature (K)

    """

    # Intialise the saturation point temperature as the temperature
    Ts = T

    # Iterate to convergence
    converged = False
    count = 0
    while not converged:

        # Update the previous Ts value
        Ts_prev = Ts

        # Compute the ice fraction
        omega = ice_fraction(Ts)

        # Compute mixed-phase relative humidity
        RH = relative_humidity(p, T, q, phase='mixed', omega=omega)

        # Compute saturation-point temperature
        Ts = _saturation_point_temperature_from_relative_humidity(T, RH, omega)
   
        # Check if solution has converged
        if np.nanmax(np.abs(Ts - Ts_prev)) < precision:
            converged = True
        else:
            count += 1
            if count == max_n_iter:
                print(f"Ts not converged after {max_n_iter} iterations")
                break

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


def lifting_saturation_level(p, T, q):
    """
    Computes pressure and temperature at the lifting saturation level (LSL)
    using equations similar to Romps (2017).

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        p_lsl (float or ndarray): pressure at the LSL (Pa)
        T_lsl (float or ndarray): temperature at the LSL (K)

    """

    # Compute effective gas constant and specific heat
    Rm = effective_gas_constant(q)
    cpm = effective_specific_heat(q)

    # Initialise the LSL temperature as the temperature
    T_lsl = T

    # Iterate to convergence
    converged = False
    count = 0
    while not converged:

        # Update the previous LSL temperature value
        T_lsl_prev = T_lsl

        # Compute the ice fraction
        omega = ice_fraction(T_lsl)

        # Compute mixed-phase relative humidity
        RH = relative_humidity(p, T, q, phase='mixed', omega=omega)

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
        W = lambertw(fn, k=-1).real  # -1 branch because cpx > cpv
        T_lsl = c * (1 / W) * T
    
        # Check if solution has converged
        if np.max(np.abs(T_lsl - T_lsl_prev)) < precision:
            converged  = True
        else:
            count += 1
            if count == max_n_iter:
                print(f"T_lsl not converged after {max_n_iter} iterations")
                break

    # Compute pressure at the LSL (cf. Romps 2017, Eq. 22b and 23b)
    p_lsl = p * np.power((T_lsl / T), (cpm / Rm))
    
    # Ensure that LSL temperature and pressure do not exceed initial values
    T_lsl = np.minimum(T_lsl, T)
    p_lsl = np.minimum(p_lsl, p)
    
    return p_lsl, T_lsl


def ice_fraction(Tstar, phase='mixed'):
    """
    Computes ice fraction given temperature at saturation.

    Args:
        Tstar (float or ndarray): temperature at saturation (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'mixed')

    Returns:
        omega (float or ndarray): ice fraction

    """

    Tstar = np.atleast_1d(Tstar)

    if phase == 'liquid':
        omega = np.zeros_like(Tstar)  # ice fraction is zero
    elif phase == 'ice':
        omega = np.ones_like(Tstar)  # ice fraction is one
    elif phase == 'mixed':
        omega = 0.5 * (1 - np.cos(np.pi * ((T_liq - Tstar) / (T_liq - T_ice))))
        omega[Tstar <= T_ice] = 1.0
        omega[Tstar >= T_liq] = 0.0
    else:
        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    if len(Tstar) == 1:
        omega = omega[0]

    return omega


def ice_fraction_derivative(Tstar, phase='mixed'):
    """
    Computes derivative of ice fraction with respect to temperature at
    saturation.
    
    Args:
        Tstar (float or ndarray): temperature at saturation (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'mixed')

    Returns:
        domega_dTstar (float or ndarray): derivative of ice fraction (K^-1)
       
    """

    Tstar = np.atleast_1d(Tstar)

    if phase == 'liquid' or phase == 'ice':
        domega_dTstar = np.zeros_like(Tstar)  # derivative is zero
    elif phase == 'mixed':
        domega_dTstar = -0.5 * (np.pi / (T_liq - T_ice)) * \
                np.sin(np.pi * ((T_liq - Tstar) / (T_liq - T_ice)))
        domega_dTstar[(Tstar <= T_ice) | (Tstar >= T_liq)] = 0.0
    else:
        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    if len(Tstar) == 1:
        domega_dTstar = domega_dTstar[0]

    return domega_dTstar


def ice_fraction_at_saturation(p, T, q, phase='mixed', saturation='isobaric'):
    """
    Computes ice fraction at saturation for specified saturation process.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'mixed')
        saturation (str, optional): saturation process (valid options are
            'isobaric' or 'adiabatic'; default is 'isobaric')

    Returns:
        omega (float or ndarray): ice fraction at saturation

    """

    if phase == 'mixed':

        if saturation == 'isobaric':

            # Compute saturation-point temperature
            Tstar = saturation_point_temperature(p, T, q)

        elif saturation == 'adiabatic':

            # Compute lifting saturation level (LSL) temperature
            _, Tstar = lifting_saturation_level(p, T, q)

        else:

            raise ValueError("saturation must be 'isobaric' or 'adiabatic'")

    else:

        # Value of Tstar is irrelevant if phase='liquid' or phase='ice'
        Tstar = T

    # Compute the ice fraction
    omega = ice_fraction(Tstar, phase=phase)

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

    # Compute the effective gas constant
    Rm = effective_gas_constant(q)

    # Compute the effective specific heat
    cpm = effective_specific_heat(q)

    # Compute dry adiabatic lapse rate
    dT_dp = (1 / p) * (Rm * T / cpm)

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

    # Set the ice fraction
    omega = ice_fraction(T, phase=phase)

    # Compute saturation specific humidity
    qs = saturation_specific_humidity(p, T, phase=phase, omega=omega)

    # Compute Q term
    Q = qs * (1 - qs + qs / eps)

    # Compute the effective gas constant
    Rm = effective_gas_constant(qs)

    # Compute the effective specific heat
    cpm = effective_specific_heat(qs)

    if phase == 'liquid':

        # Compute latent heat of vaporisation
        Lv = latent_heat_of_vaporisation(T)

        # Compute liquid pseudoadiabatic lapse rate
        dT_dp = (1 / p) * (Rm * T + Lv * Q) / \
            (cpm + (Lv**2 * Q) / (Rv * T**2))

    elif phase == 'ice':

        # Compute latent heat of sublimation
        Ls = latent_heat_of_sublimation(T)

        # Compute ice pseudoadiabatic lapse rate
        dT_dp = (1 / p) * (Rm * T + Ls * Q) / \
            (cpm + (Ls**2 * Q) / (Rv * T**2))

    elif phase == 'mixed':

        # Compute the derivative of omega with respect to temperature
        domega_dT = ice_fraction_derivative(T)

        # Compute mixed-phase latent heat
        Lx = mixed_phase_latent_heat(T, omega)

        # Compute saturation vapour pressues over liquid and ice
        esl = saturation_vapour_pressure(T, phase='liquid')
        esi = saturation_vapour_pressure(T, phase='ice')

        # Compute mixed-phase pseudoadiabatic lapse rate
        dT_dp = (1 / p) * (Rm * T + Lx * Q) / \
            (cpm + (Lx**2 * Q) / (Rv * T**2) +
             Lx * Q * np.log(esi / esl) * domega_dT)

    return dT_dp


def saturated_adiabatic_lapse_rate(p, T, qt, phase='liquid'):
    """
    Computes saturated adiabatic lapse rate in pressure coordinates.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        qt (float or ndarray): total water mass fraction (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')

    Returns:
        dT_dp (float or ndarray): saturated adiabatic lapse rate (K/Pa)

    """

    # Set the ice fraction
    omega = ice_fraction(T, phase=phase)

    # Compute saturation specific humidity
    qs = saturation_specific_humidity(p, T, qt=qt, phase=phase, omega=omega)

    # Compute Q term
    Q = qs * (1 - qt + qs / eps) / (1 - qt)

    # Compute the effective gas constant
    Rm = effective_gas_constant(qs, qt=qt)

    # Compute the effective specific heat
    cpm = effective_specific_heat(qs, qt=qt, omega=omega)

    if phase == 'liquid':

        # Compute latent heat of vaporisation
        Lv = latent_heat_of_vaporisation(T)

        # Compute liquid-only saturated adiabatic lapse rate
        dT_dp = (1 / p) * (Rm * T + Lv * Q) / \
            (cpm + (Lv**2 * Q) / (Rv * T**2))

    elif phase == 'ice':

        # Compute latent heat of sublimation
        Ls = latent_heat_of_sublimation(T)

        # Compute ice-only saturated adiabatic lapse rate
        dT_dp = (1 / p) * (Rm * T + Ls * Q) / \
            (cpm + (Ls**2 * Q) / (Rv * T**2))

    elif phase == 'mixed':

        # Compute the derivative of omega with respect to temperature
        domega_dT = ice_fraction_derivative(T)

        # Compute the mixed-phase latent heat
        Lx = mixed_phase_latent_heat(T, omega)

        # Compute the latent heat of freezing
        Lf = latent_heat_of_freezing(T)

        # Compute saturation vapour pressues over liquid and ice
        esl = saturation_vapour_pressure(T, phase='liquid')
        esi = saturation_vapour_pressure(T, phase='ice')

        # Compute mixed-phase saturated adiabatic lapse rate
        dT_dp = (1 / p) * (Rm * T + Lx * Q) / \
            (cpm + (Lx**2 * Q) / (Rv * T**2) +
             (Lx * Q * np.log(esi / esl) - Lf * (qt - qs)) * domega_dT)

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


def follow_moist_adiabat(pi, pf, Ti, qt=None, phase='liquid', pseudo=True,
                         polynomial=True, explicit=False, dp=500.0):
    """
    Computes parcel temperature following a saturated adiabat or pseudoadiabat.
    For descending parcels, a pseudoadiabat is always used. By default,
    pseudoadiabatic calculations use polynomial fits for fast calculations, but
    can optionally use direct integration. Saturated adiabatic ascent can only
    be performed using direct integration (for now). At present, polynomial
    fits are only available for liquid pseudoadiabats.

    Args:
        pi (float or ndarray): initial pressure (Pa)
        pf (float or ndarray): final pressure (Pa)
        Ti (float or ndarray): initial temperature (K)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None; required for saturated adiabatic ascent)
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

    Returns:
        Tf (float or ndarray): final temperature (K)

    """

    if pseudo and polynomial:

        if phase != 'liquid':
            raise ValueError(
                """Polynomial fits have yet to be created for ice and 
                mixed-phase pseudoadiabats. Calculations must be performed 
                using direct integration by setting polynomial=False."""
                )

        # Compute the wet-bulb potential temperature of the pseudoadiabat
        # that passes through (pi, Ti)
        thw = pseudoadiabat.wbpt(pi, Ti)

        # Compute the temperature on this pseudoadiabat at pf
        Tf = pseudoadiabat.temp(pf, thw)

    else:

        if not pseudo and qt is None:
            raise ValueError('qt is required for saturated adiabatic ascent')

        pi = np.atleast_1d(pi)
        pf = np.atleast_1d(pf)
        Ti = np.atleast_1d(Ti)

        if len(Ti) > 1:
            # multiple initial temperature values
            if len(pi) == 1:
                # single initial pressure value
                pi = np.full_like(Ti, pi)
            if len(pf) == 1:
                # single final pressure value
                pf = np.full_like(Ti, pf)

        # Set the pressure increment based on whether the parcel is ascending
        # or descending
        dp = np.abs(dp)  # make sure pressure increment is positive
        ascending = (pf < pi)
        descending = np.logical_not(ascending)
        dp = np.where(ascending, -dp, dp)

        # Initialise the pressure and temperature at level 2
        p2 = np.copy(pi)
        T2 = np.copy(Ti)

        # Create an array to store the lapse rate
        dT_dp = np.zeros_like(p2)

        #print(np.min(p2), np.max(p2))
        #print(np.count_nonzero(ascending), np.count_nonzero(descending))
        #print(np.min(dp), np.max(dp))

        # Loop over pressure increments
        while np.nanmax(np.abs(p2 - pf)) > 0.0:

            # Set level 1 values
            p1 = p2
            T1 = T2

            # Update the pressure at level 2
            p2 = p1 + dp

            # Make sure we haven't overshot final pressure level
            if np.any(ascending):
                p2[ascending] = np.maximum(p2[ascending], pf[ascending])
            if np.any(descending):
                p2[descending] = np.minimum(p2[descending], pf[descending])

            #print(np.min(p2), np.max(p2))

            # Compute the pressure at the layer mid-point (in log-space)
            pmid = np.sqrt(p1 * p2)  # = exp(0.5 * (log(p1) + log(p2)))

            # Get initial estimate for the temperature at level 2 by following
            # a dry adiabat (ignoring the contribution of moisture)
            dT_dp = dry_adiabatic_lapse_rate(p1, T1, 0.0)
            #T2 = T1 + dT_dp * (p2 - p1)
            T2 = T1 + pmid * dT_dp * np.log(p2 / p1)  # pmid * dT/dp = dT/dlnp

            # Iterate to get the new temperature at level 2
            converged = False
            count = 0
            while not converged:

                # Update the previous level 2 temperature
                T2_prev = T2

                # Compute the temperature at the layer mid-point
                Tmid = 0.5 * (T1 + T2)

                # Compute the lapse rate for ascending parcels
                if np.any(ascending):
                    if pseudo:
                        dT_dp[ascending] = pseudoadiabatic_lapse_rate(
                            pmid[ascending], Tmid[ascending],
                            phase=phase
                            )
                    else:
                        dT_dp[ascending] = saturated_adiabatic_lapse_rate(
                            pmid[ascending], Tmid[ascending], qt[ascending],
                            phase=phase
                            )

                # Compute the lapse rate for descending parcels
                if np.any(descending):
                    dT_dp[descending] = pseudoadiabatic_lapse_rate(
                        pmid[descending], Tmid[descending],
                        phase=phase
                        )

                # Update the level 2 temperature
                #T2 = T1 + dT_dp * (p2 - p1)
                T2 = T1 + pmid * dT_dp * np.log(p2 / p1)  # pmid * dT/dp = dT/dlnp

                # Check if the solution has converged
                if np.nanmax(np.abs(T2 - T2_prev)) < precision:
                    converged = True
                else:
                    count += 1
                    if count == max_n_iter:
                        # should converge in just a couple of iterations
                        # provided pinc is not too large
                        print(f'Not converged after {max_n_iter} iterations')
                        break

                if explicit:
                    # skip additional iterations
                    converged = True

            #print(np.min(p1), np.min(p2), count)

        # Set the final temperature
        Tf = T2

    return Tf


def pseudo_wet_bulb_temperature(p, T, q, phase='liquid', polynomial=True,
                                explicit=False):
    """
    Computes pseudo wet-bulb temperature.

    Pseudo wet-bulb temperature is the temperature of a parcel of air lifted
    adiabatically to saturation and then brought pseudoadiabatically at
    saturation back to its original pressure. It is always less than the
    isobaric wet-bulb temperature.

    See https://glossary.ametsoc.org/wiki/Wet-bulb_temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        polynomial (bool, optional): flag indicating whether to use polynomial
            fits to pseudoadiabats (default is True)
        explicit (bool, optional): flag indicating whether to use explicit
            integration of lapse rate equation (default is False)

    Returns:
        Tw (float or ndarray): pseudo wet-bulb temperature (K)

    """

    if phase == 'liquid':

        # Get pressure and temperature at the LCL
        p_lcl, T_lcl = lifting_condensation_level(p, T, q)

        # Follow a pseudoadiabat from the LCL to the original pressure
        Tw = follow_moist_adiabat(p_lcl, p, T_lcl, phase='liquid', pseudo=True,
                                  polynomial=polynomial, explicit=explicit)

    elif phase == 'ice':

        # Get pressure and temperature at the LDL
        p_ldl, T_ldl = lifting_deposition_level(p, T, q)

        # Follow a pseudoadiabat from the LDL to the original pressure
        Tw = follow_moist_adiabat(p_ldl, p, T_ldl, phase='ice', pseudo=True,
                                  polynomial=polynomial, explicit=explicit)

    elif phase == 'mixed':

        # Get pressure and temperature at the LSL
        p_lsl, T_lsl = lifting_saturation_level(p, T, q)

        # Follow a pseudoadiabat from the LSL to the original pressure
        Tw = follow_moist_adiabat(p_lsl, p, T_lsl, phase='mixed', pseudo=True,
                                  polynomial=polynomial, explicit=explicit)

    else:

        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    if not np.isscalar(Tw) and len(Tw) == 1:
        Tw = Tw.item()

    return Tw


def isobaric_wet_bulb_temperature(p, T, q, phase='liquid'):
    """
    Computes isobaric wet-bulb temperature.

    Isobaric wet-bulb temperature is the temperature of a parcel of air cooled
    isobarically to saturation via the evaporation of water into it, with all
    latent heat supplied by the parcel. It is always greater than the pseudo
    wet-bulb temperature. Isobaric wet-bulb temperature is similar (but not
    identical) to the quantity measured by a wet-bulb thermometer. 

    See https://glossary.ametsoc.org/wiki/Wet-bulb_temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')

    Returns:
        Tw (float or ndarray): isobaric wet-bulb temperature (K)

    """

    # Compute dewpoint temperature
    Td = dewpoint_temperature(p, T, q)

    # Initialise Tw using the "one-third rule" (Knox et al. 2017)
    Tw = T - (1 / 3) * (T - Td)

    # Compute the latent heat at temperature T
    if phase == 'liquid':
        Lv_T = latent_heat_of_vaporisation(T)
    elif phase == 'ice':
        Ls_T = latent_heat_of_sublimation(T)
    elif phase == 'mixed':
        omega_T = ice_fraction(T)
        Lx_T = mixed_phase_latent_heat(T, omega_T)
    else:
        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    # Iterate to convergence
    converged = False
    count = 0
    while not converged:

        # Update the previous Tw value
        Tw_prev = Tw

        # Compute the ice fraction at Tw
        if phase == 'liquid':
            omega_Tw = 0.0
        elif phase == 'ice':
            omega_Tw = 1.0
        elif phase == 'mixed':
            omega_Tw = ice_fraction(Tw)

        # Compute saturation specific humidity at Tw
        qs_Tw = saturation_specific_humidity(p, Tw, phase=phase,
                                             omega=omega_Tw)

        # Compute the effective specific heat at qs(Tw)
        cpm_qs_Tw = effective_specific_heat(qs_Tw)

        if phase == 'liquid':

            # Compute the latent heat of vaporisation at Tw
            Lv_Tw = latent_heat_of_vaporisation(Tw)

            # Compute the derivative of qs with respect to Tw
            dqs_dTw = qs_Tw * (1 - qs_Tw + qs_Tw / eps) * Lv_Tw / (Rv * Tw**2)

            # Compute f and f'
            f = cpm_qs_Tw * (T - Tw) - Lv_T * (qs_Tw - q)
            fprime = ((cpv - cpd) * (T - Tw) - Lv_T) * dqs_dTw - cpm_qs_Tw

        elif phase == 'ice':

            # Compute the latent heat of sublimation at Tw
            Ls_Tw = latent_heat_of_sublimation(Tw)

            # Compute the derivative of qs with respect to Tw
            dqs_dTw = qs_Tw * (1 - qs_Tw + qs_Tw / eps) * Ls_Tw / (Rv * Tw**2)

            # Compute f and f'
            f = cpm_qs_Tw * (T - Tw) - Ls_T * (qs_Tw - q)
            fprime = ((cpv - cpd) * (T - Tw) - Ls_T) * dqs_dTw - cpm_qs_Tw

        elif phase == 'mixed':

            # Compute the derivative of omega with respect to Tw
            domega_dTw = ice_fraction_derivative(Tw)

            # Compute the mixed-phase latent heat at Tw
            Lx_Tw = mixed_phase_latent_heat(Tw, omega_Tw)

            # Compute the saturation vapour pressues over liquid and ice at Tw
            esl_Tw = saturation_vapour_pressure(Tw, phase='liquid')
            esi_Tw = saturation_vapour_pressure(Tw, phase='ice')

            # Compute the derivative of qs with respect to Tw
            dqs_dTw = qs_Tw * (1 - qs_Tw + qs_Tw / eps) * \
                (Lx_Tw / (Rv * Tw**2) + np.log(esi_Tw / esl_Tw) * domega_dTw)

            # Compute f(Tw) and f'(Tw)
            f = cpm_qs_Tw * (T - Tw) - Lx_T * (qs_Tw - q)
            fprime = ((cpv - cpd) * (T - Tw) - Lx_T) * dqs_dTw - cpm_qs_Tw
       
        # Update Tw using Newton's method
        Tw = Tw - f / fprime

        # Check for convergence
        if np.nanmax(np.abs(Tw - Tw_prev)) < precision:
            converged = True
        else:
            count += 1
            if count == max_n_iter:
                print(f"Tw not converged after {max_n_iter} iterations")
                break

    return Tw


def isobaric_wet_bulb_temperature_romps(p, T, q, phase='liquid'):
    """
    Computes isobaric wet-bulb temperature using a new method developed by
    David Romps (paper currently in review), which has been implemented in the
    heatindex Python library (https://github.com/davidromps/heatindex).

    A corrigendum to Warren (2024) will be published in the near future
    containing derivations for the equations implemented in this function.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')

    Returns:
        Tw (float or ndarray): isobaric wet-bulb temperature (K)

    """

    # Compute dewpoint temperature
    Td = dewpoint_temperature(p, T, q)

    # Initialise Tw using the "one-third rule" (Knox et al. 2017)
    Tw = T - (1 / 3) * (T - Td)

    # Compute the effective specific heat
    cpm = effective_specific_heat(q)

    # Iterate to convergence
    converged = False
    count = 0
    while not converged:

        # Update the previous Tw value
        Tw_prev = Tw

        # Compute the ice fraction at Tw
        if phase == 'liquid':
            omega_Tw = 0.0
        elif phase == 'ice':
            omega_Tw = 1.0
        elif phase == 'mixed':
            omega_Tw = ice_fraction(Tw)

        # Compute saturation specific humidity at Tw
        qs_Tw = saturation_specific_humidity(p, Tw, phase=phase,
                                             omega=omega_Tw)

        if phase == 'liquid':

            # Compute the latent heat of vaporisation at Tw
            Lv_Tw = latent_heat_of_vaporisation(Tw)

            # Compute the derivative of qs with respect to Tw
            dqs_dTw = qs_Tw * (1 - qs_Tw + qs_Tw / eps) * Lv_Tw / (Rv * Tw**2)

            # Compute the derivative of Lv with respect to Tw
            dLv_dTw = (cpv - cpl)

            # Compute f and f'
            f = cpm * (T - Tw) * (1 - qs_Tw) - (qs_Tw - q) * Lv_Tw
            fprime = -(cpm * (T - Tw) + Lv_Tw) * dqs_dTw - \
                cpm * (1 - qs_Tw) - (qs_Tw - q) * dLv_dTw

        elif phase == 'ice':

            # Compute the latent heat of sublimation at Tw
            Ls_Tw = latent_heat_of_sublimation(Tw)

            # Compute the derivative of qs with respect to Tw
            dqs_dTw = qs_Tw * (1 - qs_Tw + qs_Tw / eps) * Ls_Tw / (Rv * Tw**2)

            # Compute the derivative of Ls with respect to Tw
            dLs_dTw = (cpv - cpi)

            # Compute f and f'
            f = cpm * (T - Tw) * (1 - qs_Tw) - (qs_Tw - q) * Ls(Tw)
            fprime = -(cpm * (T - Tw) + Ls_Tw) * dqs_dTw - \
                cpm * (1 - qs_Tw) - (qs_Tw - q) * dLs_dTw

        elif phase == 'mixed':

            # Compute the derivative of omega with respect to Tw
            domega_dTw = ice_fraction_derivative(Tw)

            # Compute the mixed-phase latent heat at Tw
            Lx_Tw = mixed_phase_latent_heat(Tw, omega_Tw)

            # Compute the mixed-phase isobaric specific heat
            cpx = (1 - omega_Tw) * cpl + omega_Tw * cpi

            # Compute the derivate of Lx with respect to Tw
            dLx_dTw = (cpv - cpx) + (Tw - T0) * (cpl - cpi) * domega_dTw

            # Compute the saturation vapour pressues over liquid and ice at Tw
            esl_Tw = saturation_vapour_pressure(Tw, phase='liquid')
            esi_Tw = saturation_vapour_pressure(Tw, phase='ice')

            # Compute the derivative of qs with respect to Tw
            dqs_dTw = qs_Tw * (1 - qs_Tw + qs_Tw / eps) * \
                (Lx_Tw / (Rv * Tw**2) + np.log(esi_Tw / esl_Tw) * domega_dTw)

            # Compute f(Tw) and f'(Tw)
            f = cpm * (T - Tw) * (1 - qs_Tw) - (qs_Tw - q) * Lx_Tw
            fprime = -(cpm * (T - Tw) + Lx_Tw) * dqs_dTw - \
                cpm * (1 - qs_Tw) - (qs_Tw - q) * dLx_dTw
       
        # Update Tw using Newton's method
        Tw = Tw - f / fprime

        # Check for convergence
        if np.nanmax(np.abs(Tw - Tw_prev)) < precision:
            converged = True
        else:
            count += 1
            if count == max_n_iter:
                print(f"Tw not converged after {max_n_iter} iterations")
                break

    return Tw


def wet_bulb_temperature(p, T, q, saturation='pseudo', phase='liquid',
                         polynomial=True):
    """
    Computes wet-bulb temperature for specified saturation process.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        saturation (str, optional): saturation process (valid options are
            'pseudo' or 'isobaric'; default is 'pseudo')
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        polynomial (bool, optional): flag indicating whether to use polynomial
            fits to pseudoadiabats (default is True)

    Returns:
        Tw: wet-bulb temperature (K)

    """

    if saturation == 'pseudo':
        Tw = pseudo_wet_bulb_temperature(p, T, q, phase=phase,
                                         polynomial=polynomial)
    elif saturation == 'isobaric':
        Tw = isobaric_wet_bulb_temperature(p, T, q, phase=phase)
    else:
        raise ValueError("saturation must be one of 'pseudo' or 'isobaric'")

    return Tw


def dry_potential_temperature(p, T):
    """
    Computes potential temperature of dry air.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)

    Returns:
        thd (float or ndarray): potential temperature of dry air (K)

    """

    # Compute potential temperature of dry air
    thd = T * (p_ref / p) ** (Rd / cpd)

    return thd


def moist_potential_temperature(p, T, q):
    """
    Computes potential temperature of moist air.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        thm (float or ndarray): potential temperature of moist air (K)

    """

    # Set effective gas constant and specific heat
    Rm = effective_gas_constant(q)
    cpm = effective_specific_heat(q)

    # Compute potential temperature of moist air
    thm = T * (p_ref / p) ** (Rm / cpm)

    return thm


def potential_temperature(p, T, q=None):
    """
    Computes potential temperature for dry or moist air.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray, optional): specific humidity (kg/kg) (default is
            None, in which case dry potential temperature is returned)

    Returns:
        th (float or ndarray): potential temperature (K)

    """

    if q is None:
        th = dry_potential_temperature(p, T)
    else:
        th = moist_potential_temperature(p, T, q)

    return th


def virtual_potential_temperature(p, T, q, qt=None):
    """
    Computes virtual (or density) potential temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)

    Returns:
        thv (float or ndarray): virtual potential temperature (K)

    """

    # Set effective gas constant and specific heat
    Rm = effective_gas_constant(q)
    cpm = effective_specific_heat(q)

    # Compute the virtual temperature
    Tv = virtual_temperature(T, q, qt=qt)

    # Compute virtual potential temperature
    thv = Tv * (p_ref / p) ** (Rm / cpm)

    return thv


def equivalent_potential_temperature(p, T, q, qt=None, phase='liquid',
                                     omega=0.0):
    """
    Computes equivalent potential temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction (default is 0.0)

    Returns:
        theq (float or ndarray): equivalent potential temperature (K)

    """

    if qt is None:
        qt = q

    # Compute the dry pressure
    e = vapour_pressure(p, q, qt=qt)
    pd = p - e

    # Compute the relative humidity
    RH = relative_humidity(p, T, q, qt=qt, phase=phase, omega=omega)

    # Compute cpml
    cpml = (1 - qt) * cpd + qt * cpl

    if phase == 'liquid':

        # Compute the latent heat of vaporisation
        Lv = latent_heat_of_vaporisation(T)

        # Compute the equivalent potential temperature
        theq = T * (p_ref / pd) ** ((1 - qt) * Rd / cpml) * \
            RH ** (-q * Rv / cpml) * \
            np.exp(Lv * q / (cpml * T))

    elif phase == 'ice':

        # Compute the saturation vapour pressures with respect to liquid water
        # and ice
        esl = saturation_vapour_pressure(T, phase='liquid')
        esi = saturation_vapour_pressure(T, phase='ice')

        # Compute the latent heat of sublimation
        Ls = latent_heat_of_sublimation(T)

        # Compute the latent heat of freezing
        Lf = latent_heat_of_freezing(T)

        # Compute the equivalent potential temperature
        theq = T * (p_ref / pd) ** ((1 - qt) * Rd / cpml) * \
            RH ** (-q * Rv / cpml) * \
            (esi / esl) ** (-qt * Rv / cpml) * \
            np.exp((Ls * q - Lf * qt) / (cpml * T))

    elif phase== 'mixed':

        # Compute the saturation vapour pressures with respect to liquid water
        # and ice
        esl = saturation_vapour_pressure(T, phase='liquid')
        esi = saturation_vapour_pressure(T, phase='ice')

        # Compute the mixed-phase latent heat
        Lx = mixed_phase_latent_heat(T, omega)

        # Compute the latent heat of freezing
        Lf = latent_heat_of_freezing(T)

        # Compute the equivalent potential temperature
        theq = T * (p_ref / pd) ** ((1 - qt) * Rd / cpml) * \
            RH ** (-q * Rv / cpml) * \
            (esi / esl) ** (-omega * qt * Rv / cpml) * \
            np.exp((Lx * q - omega * Lf * qt) / (cpml * T))

    else:

        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    return theq


def ice_liquid_water_potential_temperature(p, T, q, qt=None, phase='liquid',
                                           omega=0.0):
    """
    Computes ice-liquid water potential temperature using equations adapted
    from Bryan and Fritsch (2004).

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction (default is 0.0)

    Returns:
        thil (float or ndarray): ice-liquid water potential temperature (K)

    """

    if qt is None:
        qt = q

    # Compute the relative humidity with respect to liquid
    RH = relative_humidity(p, T, q, qt=qt, phase=phase, omega=omega)

    # Set constants
    Rmv = (1 - qt) * Rd + qt * Rv
    cpmv = (1 - qt) * cpd + qt * cpv

    if phase == 'liquid':

        # Compute the latent heat of vaporisation
        Lv = latent_heat_of_vaporisation(T)

        # Compute the liquid water potential temperature (c.f. Bryan and 
        # Fritsch 2004, Eq. 25)
        thil = T * (p_ref / p) ** (Rmv / cpmv) * \
            ((1 - qt + q / eps) / (1 - qt + qt / eps)) ** (Rmv / cpmv) * \
            (q / qt) ** (-qt * Rv / cpmv) * \
            RH ** ((qt - q) * Rv / cpmv) * \
            np.exp(- Lv * (qt - q) / (cpmv * T))

    elif phase == 'ice':

        # Compute the latent heat of vaporisation
        Ls = latent_heat_of_sublimation(T)

        # Compute the ice water potential temperature (c.f. Bryan and Fritsch
        # 2004, Eq. 19, 20, and 25)
        thil = T * (p_ref / p) ** (Rmv / cpmv) * \
            ((1 - qt + q / eps) / (1 - qt + qt / eps)) ** (Rmv / cpmv) * \
            (q / qt) ** (-qt * Rv / cpmv) * \
            RH ** ((qt - q) * Rv / cpmv) * \
            np.exp(- Ls * (qt - q) / (cpmv * T))

    elif phase == 'mixed':

        # Compute the mixed-phase latent heat
        Lx = mixed_phase_latent_heat(T, omega)

        # Compute the ice-liquid water potential temperature (c.f. Bryan and
        # Fritsch 2004, Eq. 19, 20, and 25)
        thil = T * (p_ref / p) ** (Rmv / cpmv) * \
            ((1 - qt + q / eps) / (1 - qt + qt / eps)) ** (Rmv / cpmv) * \
            (q / qt) ** (-qt * Rv / cpmv) * \
            RH ** ((qt - q) * Rv / cpmv) * \
            np.exp(- Lx * (qt - q) / (cpmv * T))

    else:

        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    return thil


def wet_bulb_potential_temperature(p, T, q, phase='liquid', polynomial=True):
    """
    Computes wet-bulb potential temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        polynomial (bool, optional): flag indicating whether to use polynomial
            fits to pseudoadiabats (default is True)

    Returns:
        thw (float or ndarray): wet-bulb potential temperature (K)

    """

    if phase == 'liquid':

        # Get pressure and temperature at the LCL
        p_lcl, T_lcl = lifting_condensation_level(p, T, q)

        # Follow a pseudoadiabat from the LCL to 1000 hPa
        thw = follow_moist_adiabat(p_lcl, p_ref, T_lcl, phase='liquid',
                                   pseudo=True, polynomial=polynomial)

    elif phase == 'ice':

        # Get pressure and temperature at the LDL
        p_ldl, T_ldl = lifting_deposition_level(p, T, q)

        # Follow a pseudoadiabat from the LDL to 1000 hPa
        thw = follow_moist_adiabat(p_ldl, p_ref, T_ldl, phase='ice',
                                   pseudo=True, polynomial=polynomial)

    elif phase == 'mixed':

        # Get pressure and temperature at the LSL
        p_lsl, T_lsl = lifting_saturation_level(p, T, q)

        # Follow a pseudoadiabat from the LSL to 1000 hPa
        thw = follow_moist_adiabat(p_lsl, p_ref, T_lsl, phase='mixed',
                                   pseudo=True, polynomial=polynomial)

    else:

        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    if not np.isscalar(thw) and len(thw) == 1:
        thw = thw.item()

    return thw


def saturation_wet_bulb_potential_temperature(p, T, phase='liquid',
                                              polynomial=True):
    """
    Computes saturation wet-bulb potential temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        polynomial (bool, optional): flag indicating whether to use polynomial
            fits to pseudoadiabats (default is True)

    Returns:
        thws (float or ndarray): saturation wet-bulb potential temperature (K)

    """

    # Follow a pseudoadiabat to 1000 hPa
    thws = follow_moist_adiabat(p, p_ref, T, phase=phase, pseudo=True,
                                polynomial=polynomial)

    if not np.isscalar(thws) and len(thws) == 1:
        thws = thws.item()

    return thws


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


def hydrostatic_pressure(z, T, q, p_sfc, z_sfc=None, T_sfc=None, q_sfc=None,
                         vertical_axis=0):
    """
    Computes hydrostatic pressure from profiles of height, temperature, and
    specific humidity and surface pressure using the hypsometric equation.

    Args:
        z (ndarray): height profile(s) (m)
        T (ndarray): temperature profile(s) (K)
        q (ndarray): specific humidity profile(s) (kg/kg)
        p_sfc (float or ndarray): surface pressure (Pa)
        z_sfc (float or ndarray, optional): surface height (m)
        T_sfc (float or ndarray, optional): surface temperature (K)
        q_sfc (float or ndarray, optional): surface specific humidity (kg/kg)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)

    Returns:
        P (float or ndarray): hydrostatic pressure (Pa)

    """

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        z = np.moveaxis(z, vertical_axis, 0)
        T = np.moveaxis(T, vertical_axis, 0)
        q = np.moveaxis(q, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if z.ndim == 1:
        z = np.atleast_2d(z).T  # transpose to preserve vertical axis
        T = np.atleast_2d(T).T
        q = np.atleast_2d(q).T

    # If surface-level fields not provided, use lowest level values
    if T_sfc is None:
        k_start = 1  # start loop from second level
        T_sfc = T[0]
        q_sfc = q[0]
        z_sfc = z[0]
    else:
        k_start = 0  # start loop from first level
        if z_sfc is None:
            z_sfc = np.zeros_like(p_sfc)  # assumes height AGL

    # Make sure that surface fields are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    z_sfc = np.atleast_1d(z_sfc)
    T_sfc = np.atleast_1d(T_sfc)
    q_sfc = np.atleast_1d(q_sfc)

    # Note the number of vertical levels
    n_lev = z.shape[0]

    # Create array for hydrostatic pressure
    P = np.zeros_like(z)

    # Initialise level 2 height and hydrostatic pressure
    z2 = z_sfc.copy()
    P2 = p_sfc.copy()

    # Compute level 2 virtual temperature
    Tv2 = virtual_temperature(T_sfc, q_sfc)

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 fields
        z1 = z2.copy()
        P1 = P2.copy()
        Tv1 = Tv2.copy()

        # Find points above the surface
        above_sfc = (z[k] >= z_sfc)
        if np.any(above_sfc):

            # Update level 2 height
            z2[above_sfc] = z[k][above_sfc]

            # Compute level 2 virtual temperature
            Tv2[above_sfc] = virtual_temperature(T[k][above_sfc],
                                                 q[k][above_sfc])

            # Compute level 2 hydrostatic pressure
            P2[above_sfc] = P1[above_sfc] * np.exp(
                -1 * g * (z2[above_sfc] - z1[above_sfc]) /
                (Rd * 0.5 * (Tv1[above_sfc] + Tv2[above_sfc]))
            )

            # Save hydrostatic pressure for this level
            P[k][above_sfc] = P[above_sfc]

    return P.squeeze()  # remove dimensions of length 1
