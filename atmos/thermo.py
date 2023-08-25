"""
References:

Ambaum, M.H., 2020. Accurate, simple equation for saturated vapour
    pressure over water and ice. Quart. J. Roy. Met. Soc., 146, 4252-4258.

Romps, D.M., 2017. Exact expression for the lifting condensation level.
    J. Atmos. Sci., 74, 3033-3057.
        
Romps, D.M., 2021. Accurate expressions for the dewpoint and frost point
    derived from the Rankine-Kirchoff approximations. J. Atmos. Sci., 78,
    2113-2116.

"""


import numpy as np
from scipy.special import lambertw
from atmos.constant import (Rd, Rv, eps, cpd, cpv, cpl, cpi, 
                            T0, es0, Lv0, Lf0, Ls0)
import atmos.pseudoadiabat as pseudoadiabat


def effective_gas_constant(q):
    """
    Computes effective gas constant for moist air.

    Args:
        q: specific humidity (kg/kg)

    Returns:
        Rm: effective gas constant for moist air (J/kg/K)

    """
    Rm = (1 - q) * Rd + q * Rv

    return Rm


def effective_specific_heat(q):
    """
    Computes effective isobaric specific heat for moist air (J/kg/K).

    Args:
        q: specific humidity (kg/kg)

    Returns:
        cpm: effective isobaric specific heat for moist air (J/kg/K)

    """
    cpm = (1 - q) * cpd + q * cpv

    return cpm


def latent_heat_of_vaporisation(T):
    """
    Computes latent heat of vaporisation for a given temperature.

    Args:
        T: temperature (K)

    Returns:
        Lv: latent heat of vaporisation (J/kg)

    """
    Lv = Lv0 - (cpl - cpv) * (T - T0)

    return Lv


def latent_heat_of_freezing(T):
    """
    Computes latent heat of freezing for a given temperature.

    Args:
        T: temperature (K)

    Returns:
        Lf: latent heat of freezing (J/kg)

    """
    Lf = Lf0 - (cpi - cpl) * (T - T0)

    return Lf


def latent_heat_of_sublimation(T):
    """
    Computes latent heat of sublimation for a given temperature.

    Args:
        T: temperature (K)

    Returns:
        Ls: latent heat of sublimation (J/kg)

    """
    Ls = Ls0 - (cpi - cpv) * (T - T0)

    return Ls


def air_density(p, T, q):
    """
    Computes density of air using the ideal gas equation.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)

    Returns:
        rho: air density (kg/m3)

    """
    Rm = effective_gas_constant(q)
    rho = p / (Rm * T)

    return rho


def dry_air_density(p, T, q):
    """
    Computes density of dry air using the ideal gas equation.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)

    Returns:
        rhod: dry air density (kg/m3)

    """
    rhod = (1 - q) * air_density(p, T, q)

    return rhod


def virtual_temperature(T, q):
    """
    Computes virtual temperature.

    Args:
        T: temperature (K)
        q: specific humidity (kg/kg)

    Returns:
        Tv: virtual temperature (K)

    """
    Tv = T * (1 + (1/eps - 1) * q)

    return Tv


def mixing_ratio(q):
    """
    Computes water vapour mixing ratio from specific humidity.

    Args:
        q: specific humidity (kg/kg)

    Returns:
        r: mixing ratio (kg/kg)

    """
    r = q / (1 - q)

    return r


def vapour_pressure(p, q):
    """
    Computes vapour pressure from pressure and specific humidity.

    Args:
        p: pressure (Pa)
        q: specific humidity (kg/kg)

    Returns:
        e: vapour pressure (Pa)

    """
    e = p * q / ((1 - eps) * q + eps)

    return e


def saturation_vapour_pressure(T, phase='liquid', Tl=273.15, Ti=253.15):
    """
    Computes saturation vapour pressure (SVP) over liquid, ice, or mixed-phase
    water for a given temperature using equations from Ambaum (2020)

    Args:
        T: temperature (K)
        phase (optional): condensed water phase (liquid, ice, or mixed)
        Tl (optional): temperature above which condensate is all liquid (K)
        Ti (optional): temperature below which condensate is all ice (K)

    Returns:
        es: saturation vapour pressure (Pa)

    """

    if phase not in ['liquid', 'ice', 'mixed']:
        raise ValueError("Phase must be one of 'liquid', 'solid', or 'ice'")
    
    if Ti >= Tl:
        # TODO: Allow for case where Tl == Ti
        raise ValueError('Ti must be less than Tl')
    
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
        
    else:
        
        # Compute the ice fraction
        icefrac = (Tl - T) / (Tl - Ti)
        
        # Compute mixed-phase specific heat
        cpx = (1 - icefrac) * cpl + icefrac * cpi
        
        # Compute mixed-phase latent heat
        Lx0 = (1 - icefrac) * Lv0 + icefrac * Ls0
        Lx = Lx0 + (cpv - cpx) * (T - T0)
        
        # Compute mixed-phase SVP
        es = es0 * np.power((T0 / T), ((cpx - cpv) / Rv)) * \
            np.exp((Lx0 / (Rv * T0)) - (Lx / (Rv * T)))

    return es


def saturation_specific_humidity(p, T, phase='liquid', Tl=273.15, Ti=253.15):
    """
    Computes saturation specific humidity from pressure and temperature.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        phase (optional): condensed water phase (liquid, ice, or mixed)
        Tl (optional): temperature above which condensate is all liquid (K)
        Ti (optional): temperature below which condensate is all ice (K)

    Returns:
        qs: saturation specific humidity (kg/kg)

    """
    es = saturation_vapour_pressure(T, phase=phase, Tl=Tl, Ti=Ti)
    qs = eps * es / (p - (1 - eps) * es)

    return qs


def saturation_mixing_ratio(p, T, phase='liquid', Tl=273.15, Ti=253.15):
    """
    Computes saturation mixing ratio from pressure and temperature.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        phase (optional): condensed water phase (liquid, ice, or mixed)
        Tl (optional): temperature above which condensate is all liquid (K)
        Ti (optional): temperature below which condensate is all ice (K)

    Returns:
        rs: saturation mixing ratio (kg/kg)

    """
    es = saturation_vapour_pressure(T, phase=phase, Tl=Tl, Ti=Ti)
    rs = eps * es / (p - es)

    return rs


def relative_humidity(p, T, q, phase='liquid', Tl=273.15, Ti=253.15):
    """
    Computes relative humidity from pressure, temperature, and specific 
    humidity.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)
        phase (optional): condensed water phase (liquid, ice, or mixed)
        Tl (optional): temperature above which condensate is all liquid (K)
        Ti (optional): temperature below which condensate is all ice (K)
        
    Returns:
        RH: relative humidity (fraction)

    """
    e = vapour_pressure(p, q)
    es = saturation_vapour_pressure(T, phase=phase, Tl=Tl, Ti=Ti)
    RH = e / es

    return RH


def dewpoint_temperature(p, T, q):
    """
    Computes dewpoint temperature from pressure, temperature, and specific
    humidity using equations from Romps (2021).

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)

    Returns:
        Td: dewpoint temperature (K)

    """

    # Compute relative humidity over liquid water
    RH = relative_humidity(p, T, q, phase='liquid')
    RH = np.minimum(RH, 1.0)  # limit RH to 100 %

    # Set constant (Romps 2021, Eq. 6)
    c = (Lv0 - (cpv - cpl) * T0) / ((cpv - cpl) * T)

    # Compute dewpoint temperature (Romps 2021, Eq. 5)
    fn = np.power(RH, (Rv / (cpl - cpv))) * c * np.exp(c)
    W = lambertw(fn, k=-1).real
    Td = c * (1 / W) * T
    
    # Ensure that Td does not exceed T
    Td = np.minimum(Td, T)

    return Td


def frostpoint_temperature(p, T, q):
    """
    Computes frost-point temperature from pressure, temperature, and 
    specific humidity using equations from Romps (2021).

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)

    Returns:
        Tf: frost-point temperature (K)

    """    
    # Compute relative humidity over ice
    RH = relative_humidity(p, T, q, phase='ice')
    RH = np.minimum(RH, 1.0)  # limit RH to 100 %

    # Set constant (Romps 2021, Eq. 8)
    c = (Ls0 - (cpv - cpi) * T0) / ((cpv - cpi) * T)

    # Compute dewpoint temperature (Romps 2021, Eq. 7)
    fn = np.power(RH, (Rv / (cpi - cpv))) * c * np.exp(c)
    W = lambertw(fn, k=0).real
    Tf = c * (1 / W) * T
    
    # Ensure that Tf does not exceed T
    Tf = np.minimum(Tf, T)

    return Tf


def lifting_condensation_level(p, T, q):
    """
    Computes pressure and parcel temperature at the lifted condensation
    level (LCL) using equations from Romps (2017).

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)

    Returns:
        p_lcl: pressure at the LCL (Pa)
        T_lcl: temperature at the LCL (K)

    """
    
    # Compute relative humidity with respect to liquid water
    RH = relative_humidity(p, T, q, phase='liquid')
    RH = np.minimum(RH, 1.0)  # limit RH to 100 %

    # Compute effective gas constant and specific heat
    Rm = effective_gas_constant(q)
    cpm = effective_specific_heat(q)
    
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
    Computes pressure and parcel temperature at the lifting deposition
    level (LDL) using equations from Romps (2017).

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)

    Returns:
        p_ldl: pressure at the LDL (Pa)
        T_ldl: temperature at the LDL (K)

    """
    
    # Compute relative humidity with respect to ice
    RH = relative_humidity(p, T, q, phase='ice')
    RH = np.minimum(RH, 1.0)  # limit RH to 100 %

    # Compute effective gas constant and specific heat
    Rm = effective_gas_constant(q)
    cpm = effective_specific_heat(q)
    
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


def dry_adiabat(p1, T1, p2, q=0):
    """
    Compute parcel temperature following a dry adiabat.

    Args:
        p1: initial pressure (Pa)
        T1: initial temperature (K)
        p2: final pressure (Pa)
        q (optional): specific humidity (kg/kg)

    Returns:
        T2: final temperature (K)

    """

    # Set effective gas constant and specific heat
    Rm = effective_gas_constant(q)
    cpm = effective_specific_heat(q)

    # Compute new temperature
    T2 = T1 * np.power((p2 / p1), (Rm / cpm))

    return T2


def saturated_adiabat(p1, T1, p2):
    """
    Compute parcel temperature following a saturated pseudoadiabat.

    Args:
        p1: initial pressure (Pa)
        T1: initial temperature (K)
        p2: final pressure (Pa)

    Returns:
        T2: final temperature (K)

    """

    # Compute the wet-bulb potential temperature of the pseudoadiabat
    # that passes through (p1, T1)
    thw = pseudoadiabat.wbpt(p1, T1)

    # Compute the temperature on this pseudoadiabat at p2
    T2 = pseudoadiabat.temp(p2, thw)
    
    return T2


def potential_temperature(p, T, q=0):
    """
    Computes potential temperature, optionally including moisture
    contribution to dry adiabatic lapse rate.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q (optional): specific humidity (kg/kg)

    Returns:
        th: potential temperature (K)

    """

    # Follow a dry adiabat to 1000hPa reference
    th = dry_adiabat(p, T, 100000., q)

    return th


def wetbulb_temperature(p, T, q):
    """
    Computes wet-bulb temperature.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)

    Returns:
        Tw: wet-bulb temperature (K)

    """

    # Get pressure and parcel temperature at the LCL
    p_lcl, Tp_lcl = lifting_condensation_level(p, T, q)

    # Follow a saturated adiabat from the LCL to the original pressure
    Tw = saturated_adiabat(p_lcl, Tp_lcl, p)

    return Tw


def wetbulb_potential_temperature(p, T, q):
    """
    Computes wet-bulb potential temperature.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)

    Returns:
        thw: wet-bulb potential temperature (K)

    """

    # Get pressure and parcel temperature at the LCL
    p_lcl, Tp_lcl = lifting_condensation_level(p, T, q)

    # Compute the wet-bulb potential temperature of the pseudoadiabat
    # that passes through (p_lcl, Tp_lcl)
    thw = pseudoadiabat.wbpt(p_lcl, Tp_lcl)

    return thw


def saturated_wetbulb_potential_temperature(p, T):
    """
    Computes saturation wet-bulb potential temperature.

    Args:
        p: pressure (Pa)
        T: temperature (K)

    Returns:
        thws: wet-bulb potential temperature (K)

    """

    # Compute the wet-bulb potential temperature of the pseudoadiabat
    # that passes through (p, T)
    thws = pseudoadiabat.wbpt(p, T)

    return thws
