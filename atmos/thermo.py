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


def saturation_vapour_pressure(T, phase='liquid', icefrac=None):
    """
    Computes saturation vapour pressure (SVP) over liquid, ice, or mixed-phase
    water for a given temperature using equations from Ambaum (2020).

    Args:
        T: temperature (K)
        phase (optional): condensed water phase ('liquid', 'ice', or 'mixed')
        icefrac (optional, required for mixed phase): ice fraction at saturation

    Returns:
        es: saturation vapour pressure (Pa)

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
        
        if icefrac is None:
            raise ValueError("icefrac must be provided for phase='mixed'")
        
        # Compute mixed-phase specific heat
        cpx = (1 - icefrac) * cpl + icefrac * cpi
        
        # Compute mixed-phase latent heat
        Lx0 = (1 - icefrac) * Lv0 + icefrac * Ls0
        Lx = Lx0 - (cpx - cpv) * (T - T0)
        
        # Compute mixed-phase SVP
        es = es0 * np.power((T0 / T), ((cpx - cpv) / Rv)) * \
            np.exp((Lx0 / (Rv * T0)) - (Lx / (Rv * T)))
        
    else:

        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    return es


def saturation_specific_humidity(p, T, phase='liquid', icefrac=None):
    """
    Computes saturation specific humidity from pressure and temperature.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        phase (optional): condensed water phase ('liquid', 'ice', or 'mixed')
        icefrac (optional, required for mixed phase): ice fraction at saturation

    Returns:
        qs: saturation specific humidity (kg/kg)

    """
    es = saturation_vapour_pressure(T, phase=phase, icefrac=icefrac)
    qs = eps * es / (p - (1 - eps) * es)

    return qs


def saturation_mixing_ratio(p, T, phase='liquid', icefrac=None):
    """
    Computes saturation mixing ratio from pressure and temperature.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        phase (optional): condensed water phase ('liquid', 'ice', or 'mixed')
        icefrac (optional, required for mixed phase): ice fraction at saturation

    Returns:
        rs: saturation mixing ratio (kg/kg)

    """
    es = saturation_vapour_pressure(T, phase=phase, icefrac=icefrac)
    rs = eps * es / (p - es)

    return rs


def relative_humidity(p, T, q, phase='liquid', icefrac=None):
    """
    Computes relative humidity from pressure, temperature, and specific 
    humidity.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)
        phase (optional): condensed water phase ('liquid', 'ice', or 'mixed')
        icefrac (optional, required for mixed phase): ice fraction at saturation
        
    Returns:
        RH: relative humidity with respect to specified phase (fraction)

    """
    e = vapour_pressure(p, q)
    es = saturation_vapour_pressure(T, phase=phase, icefrac=icefrac)
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


def saturation_point_temperature(p, T, q, icefrac):
    """
    Computes saturation-point temperature from pressure, temperature, and 
    specific humidity using equations similar to Romps (2021).

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)
        icefrac: ice fraction at saturation (fraction)

    Returns:
        Ts: saturation-point temperature (K)

    """

    # Compute the mixed-phase relative humidity
    RH = relative_humidity(p, T, q, phase='mixed', icefrac=icefrac)
    #RH = np.minimum(RH, 1.0)  # limit RH to 100 %

    # Compute mixed-phase specific heat
    cpx = (1 - icefrac) * cpl + icefrac * cpi
    
    # Compute mixed-phase latent heat at the triple point
    Lx0 = (1 - icefrac) * Lv0 + icefrac * Ls0

    # Set constant (cf. Romps 2021, Eq. 6 and 8)
    c = (Lx0 - (cpv - cpx) * T0) / ((cpv - cpx) * T)

    # Compute saturation-point temperature (cf. Romps 2021, Eq. 5 and 7)
    fn = np.power(RH, (Rv / (cpx - cpv))) * c * np.exp(c)
    W = lambertw(fn, k=-1).real
    Ts = c * (1 / W) * T
   
    # Ensure that Ts does not exceed T
    Ts = np.minimum(Ts, T)

    return Ts


def lifting_condensation_level(p, T, q):
    """
    Computes pressure and parcel temperature at the lifted condensation level
    (LCL) using equations from Romps (2017).

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)

    Returns:
        p_lcl: pressure at the LCL (Pa)
        T_lcl: temperature at the LCL (K)

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
    Computes pressure and parcel temperature at the lifting deposition level
    (LDL) using equations from Romps (2017).

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)

    Returns:
        p_ldl: pressure at the LDL (Pa)
        T_ldl: temperature at the LDL (K)

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


def lifting_saturation_level(p, T, q, icefrac):
    """
    Computes pressure and parcel temperature at the lifting saturation level
    (LSL) using equations similar to Romps (2017).

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)
        icefrac: ice fraction at saturation (fraction)

    Returns:
        p_lsl: pressure at the LSL (Pa)
        T_lsl: temperature at the LSL (K)

    """

    # Compute effective gas constant and specific heat
    Rm = effective_gas_constant(q)
    cpm = effective_specific_heat(q)

    # Compute mixed-phase relative humidity
    RH = relative_humidity(p, T, q, phase='mixed', icefrac=icefrac)
    #RH = np.minimum(RH, 1.0)  # limit RH to 100 %

    # Compute mixed-phase specific heat
    cpx = (1 - icefrac) * cpl + icefrac * cpi
        
    # Compute mixed-phase latent heat at the triple point
    Lx0 = (1 - icefrac) * Lv0 + icefrac * Ls0
    
    # Set constants (cf. Romps 2017, Eq. 22d-f and 23d-f)
    a = cpm / Rm + (cpx - cpv) / Rv
    b = -(Lx0 + (cpx - cpv) * T0) / (Rv * T)
    c = b / a

    # Compute temperature at the LSL (cf. Romps 2017, Eq. 22a and 23a)
    fn = np.power(RH, (1 / a)) * c * np.exp(c)
    W = lambertw(fn, k=-1).real
    T_lsl = c * (1 / W) * T
    
    # Compute pressure at the LSL (cf. Romps 2017, Eq. 22b and 23b)
    p_lsl = p * np.power((T_lsl / T), (cpm / Rm))
    
    # Ensure that LSL temperature and pressure do not exceed initial values
    T_lsl = np.minimum(T_lsl, T)
    p_lsl = np.minimum(p_lsl, p)
    
    return p_lsl, T_lsl


def ice_fraction(Tstar, Tliq=273.15, Tice=253.15):
    """
    Computes ice fraction given temperature at saturation, Tstar.
    
    Args:
        Tstar: temperature at saturation (K)
        Tliq (optional): temperature above which condensate is 100 % liquid (K)
        Tice (optional): temperature below which condensate is 100 % ice (K)

    Returns:
        icefrac: ice fraction (fraction)
       
    """
    if Tice > Tliq:
        raise ValueError('Tice must be less than or equal to Tliq')
    elif Tliq < Tice:
        icefrac = 0.5 * (1 - np.cos(np.pi * ((Tliq - Tstar) / (Tliq - Tice))))
    icefrac[Tstar <= Tice] = 1.0
    icefrac[Tstar >= Tliq] = 0.0

    return icefrac


def ice_fraction_at_saturation(p, T, q, saturation='isobaric', converged=0.001,
                               Tliq=273.15, Tice=253.15):
    """
    Computes ice fraction at saturation following specified saturation process.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)
        saturation (optional): saturation process (isobaric or adiabatic)
        converged (optional): target precision for Tstar (K)
        Tliq (optional): temperature above which condensate is 100 % liquid (K)
        Tice (optional): temperature below which condensate is 100 % ice (K)

    Returns:
        icefrac: ice fraction at saturation (fraction)

    """

    # Initialise the temperature at saturation as the actual temperature
    Tstar = T

    # Compute the initial ice fraction
    icefrac = ice_fraction(Tstar, Tliq=Tliq, Tice=Tice)

    # Iterate to convergence
    count = 0
    delta = np.full_like(Tstar, 10)
    while np.max(delta) > converged:

        # Update the previous Tstar value
        Tstar_prev = Tstar

        if saturation == 'isobaric':

            # Compute saturation-point temperature
            Tstar = saturation_point_temperature(p, T, q, icefrac)

        elif saturation == 'adiabatic':

            # Compute lifting saturation level (LSL) temperature
            _, Tstar = lifting_saturation_level(p, T, q, icefrac)

        else:

            raise ValueError("saturation must be one of 'isobaric' or 'adiabatic'")

        # Update the ice fraction
        icefrac = ice_fraction(Tstar, Tliq=Tliq, Tice=Tice)

        # Check if solution has converged
        delta = np.abs(Tstar - Tstar_prev)      
        count += 1
        if count > 20:
            print("saturation temperature not converged after 20 iterations")
            break

    return icefrac


def follow_dry_adiabat(p1, T1, p2, q=0):
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


def follow_pseudoadiabat(p1, T1, p2):
    """
    Compute parcel temperature following a pseudoadiabat.

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
    th = follow_dry_adiabat(p, T, 100000., q)

    return th


def adiabatic_wet_bulb_temperature(p, T, q):
    """
    Computes adiabatic wet-bulb temperature.

    Adiabatic wet-bulb temperature is the temperature of a parcel of air lifted 
    dry adiabatically to saturation and then brought (pseudo)adiabatically at
    saturation back to its original pressure. It is always less than the
    isobaric wet-bulb temperature.

    See https://glossary.ametsoc.org/wiki/Wet-bulb_temperature.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)

    Returns:
        Tw: adiabatic wet-bulb temperature (K)

    """

    # Get pressure and parcel temperature at the LCL
    p_lcl, Tp_lcl = lifting_condensation_level(p, T, q)

    # Follow a pseudoadiabat from the LCL to the original pressure
    Tw = follow_pseudoadiabat(p_lcl, Tp_lcl, p)

    return Tw


def isobaric_wet_bulb_temperature(p, T, q, converged=0.001):
    """
    Computes isobaric wet-bulb temperature.

    Isobaric wet-bulb temperature is the temperature of a parcel of air cooled
    isobarically to saturation via the evaporation of water into it, with all
    latent heat supplied by the parcel. It is always greater than the adiabatic
    wet-bulb temperature. Isobaric wet-bulb temperature is similar (but not
    identical) to the quantity measured by a wet-bulb thermometer. 

    See https://glossary.ametsoc.org/wiki/Wet-bulb_temperature.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)
        converged (optional): target precision for Tw (K)

    Returns:
        Tw: isobaric wet-bulb temperature (K)

    """

    # Compute effective specific heat at specific humidity q
    cpm_q = effective_specific_heat(q)

    # Compute latent heat at temperature T
    Lv_T = latent_heat_of_vaporisation(T)

    # Compute dewpoint temperature
    Td = dewpoint_temperature(p, T, q)

    # Initialise Tw as mean of T and Td
    Tw = (T + Td) / 2

    # Iterate to convergence
    delta = np.full_like(p, 10.)
    count = 0
    while np.max(delta) > converged:

        Tw_prev = Tw

        # Compute saturation specific humidity at Tw
        qs_Tw = saturation_specific_humidity(p, Tw)

        # Compute effective specific heat at qs_Tw
        cpm_qs_Tw = effective_specific_heat(qs_Tw)

        # Compute latent heat at Tw
        Lv_Tw = latent_heat_of_vaporisation(Tw)

        # Compute f and fprime
        f = (1 / (cpv - cpl)) * np.log(Lv_Tw / Lv_T) + \
            (1 / (cpv - cpd)) * np.log(cpm_qs_Tw / cpm_q)
        fprime = (1 / Lv_Tw) * (qs_Tw * (1 + qs_Tw / eps - qs_Tw) * Lv_Tw /
                                (cpm_qs_Tw * Rv * Tw**2))
        
        # Update Tw using Newton's method
        Tw = Tw - f / fprime

        # Check for convergence
        delta = np.abs(Tw - Tw_prev)
        count += 1
        if count > 20:
            print("isobaric wet-bulb temperature not converged after 20 iterations")
            break

    return Tw


def wet_bulb_temperature(p, T, q, saturation='adiabatic', converged=0.001):
    """
    Computes wet-bulb temperature following specified saturation process.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)
        saturation (optional): saturation process (isobaric or adiabatic)
        converged (optional): target precision for isobaric Tw (K)

    Returns:
        Tw: wet-bulb temperature (K)

    """

    if saturation == 'adiabatic':
        Tw = isobaric_wet_bulb_temperature(p, T, q, converged=converged)
    elif saturation == 'isobaric':
        Tw = adiabatic_wet_bulb_temperature(p, T, q)
    else:
        raise ValueError("saturation must be one of 'isobaric' or 'adiabatic'")

    return Tw


def wet_bulb_potential_temperature(p, T, q):
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


def saturated_wet_bulb_potential_temperature(p, T):
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
