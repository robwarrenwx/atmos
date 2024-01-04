"""
Functions for converting between the following moisture variables:
* specific humidity, q
* mixing ratio, r
* vapour pressure, e
* relative humidity, RH
* dewpoint temperature, Td
* frost-point temperature, Tf
* saturation-point temperature, Ts

"""

import numpy as np
from atmos.constant import eps
from atmos.thermo import (mixing_ratio, 
                          vapour_pressure,
                          relative_humidity,
                          saturation_vapour_pressure,
                          saturation_specific_humidity,
                          saturation_mixing_ratio, 
                          dewpoint_temperature,
                          frost_point_temperature,
                          saturation_point_temperature,
                          ice_fraction)


def specific_humidity_from_mixing_ratio(r):
    """
    Computes specific humidity from water vapour mixing ratio.

    Args:
        r (float or ndarray): mixing ratio (kg/kg)

    Returns:
        q (float or ndarray): specific humidity (kg/kg)

    """
    q = r / (1 + r)

    return q


def specific_humidity_from_vapour_pressure(p, e):
    """
    Computes specific humidity from pressure and vapour pressure.

    Args:
        p (float or ndarray): pressure (Pa)
        e (float or ndarray): vapour pressure (Pa)

    Returns:
        q (float or ndarray): specific humidity (kg/kg)

    """
    q = eps * e / (p - (1 - eps) * e)

    return q


def specific_humidity_from_relative_humidity(p, T, RH, phase='liquid', 
                                             omega=0.0):
    """
    Computes specific humidity from pressure, temperature, and relative 
    humidity with respect to specified phase.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)
        phase (str, optional): condensed water phase (valid options are 
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation 
            (default is 0.0)
        
    Returns:
        q (float or ndarray): specific humidity (kg/kg)

    """
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    e = RH * es
    q = specific_humidity_from_vapour_pressure(p, e)

    return q


def specific_humidity_from_dewpoint_temperature(p, Td):
    """
    Computes specific humidity from pressure and dewpoint temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        Td (float or ndarray): dewpoint temperature (K)

    Returns:
        q (float or ndarray): specific humidity (kg/kg)

    """    
    q = saturation_specific_humidity(p, Td, phase='liquid')

    return q


def specific_humidity_from_frost_point_temperature(p, Tf):
    """
    Computes specific humidity from pressure and frost-point temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        Tf (float or ndarray): frost-point temperature (K)

    Returns:
        q (float or ndarray): specific humidity (kg/kg)

    """    
    q = saturation_specific_humidity(p, Tf, phase='ice')

    return q


def specific_humidity_from_saturation_point_temperature(p, Ts, omega):
    """
    Computes specific humidity from pressure and saturation-point temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        Ts (float or ndarray): saturation-point temperature (K)
        omega (float or ndarray): ice fraction at saturation

    Returns:
        q (float or ndarray): specific humidity (kg/kg)

    """    
    q = saturation_specific_humidity(p, Ts, phase='mixed', omega=omega)

    return q


def mixing_ratio_from_specific_humidity(q):
    """
    Computes mixing ratio from specific humidity.

    Args:
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """
    r = mixing_ratio(q)

    return r


def mixing_ratio_from_vapour_pressure(p, e):
    """
    Computes mixing ratio from pressure and vapour pressure.

    Args:
        p (float or ndarray): pressure (Pa)
        e (float or ndarray): vapour pressure (Pa)

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """
    r = eps * e / (p - e)

    return r


def mixing_ratio_from_relative_humidity(p, T, RH, phase='liquid', omega=0.0):
    """
    Computes mixing ratio from pressure, temperature, and relative humidity
    with respect to specified phase.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)
        phase (str, optional): condensed water phase (valid options are 
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation 
            (default is 0.0)

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    e = RH * es
    r = mixing_ratio_from_vapour_pressure(p, e)

    return r


def mixing_ratio_from_dewpoint_temperature(p, Td):
    """
    Computes mixing ratio from pressure and dewpoint temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        Td (float or ndarray): dewpoint temperature (K)

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """    
    r = saturation_mixing_ratio(p, Td, phase='liquid')

    return r


def mixing_ratio_from_frost_point_temperature(p, Tf):
    """
    Computes mixing ratio from pressure and frost-point temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        Tf (float or ndarray): frost-point temperature (K)

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """    
    r = saturation_mixing_ratio(p, Tf, phase='ice')

    return r


def mixing_ratio_from_saturation_point_temperature(p, Ts, omega):
    """
    Computes mixing ratio from pressure and saturation-point temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        Ts (float or ndarray): saturation-point temperature (K)
        omega (float or ndarray): ice fraction at saturation

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """    
    r = saturation_mixing_ratio(p, Ts, phase='mixed', omega=omega)

    return r
    

def vapour_pressure_from_specific_humidity(p, q):
    """
    Computes vapour pressure from pressure and specific humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    e = vapour_pressure(p, q)

    return e


def vapour_pressure_from_mixing_ratio(p, r):
    """
    Computes vapour pressure from pressure and mixing ratio.

    Args:
        p (float or ndarray): pressure (Pa)
        r (float or ndarray): mixing ratio (kg/kg)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    e = p * r / (r + eps)

    return e


def vapour_pressure_from_relative_humidity(T, RH, phase='liquid', omega=0.0):
    """
    Computes vapour pressure from temperature and relative humidity with
    respect to specified phase.

    Args:
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)
        phase (str, optional): condensed water phase (valid options are 
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation 
            (default is 0.0)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    e = RH * es

    return e


def vapour_pressure_from_dewpoint_temperature(Td):
    """
    Computes vapour pressure from dewpoint temperature.

    Args:
        Td (float or ndarray): dewpoint temperature (K)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    e = saturation_vapour_pressure(Td, phase='liquid')

    return e


def vapour_pressure_from_frost_point_temperature(Tf):
    """
    Computes vapour pressure from frost-point temperature.

    Args:
        Tf (float or ndarray): frost-point temperature (K)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    e = saturation_vapour_pressure(Tf, phase='ice')

    return e


def vapour_pressure_from_saturation_point_temperature(Ts, omega):
    """
    Computes vapour pressure from saturation-point temperature.

    Args:
        Ts (float or ndarray): saturation-point temperature (K)
        omega (float or ndarray): ice fraction at saturation

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    e = saturation_vapour_pressure(Ts, phase='mixed', omega=omega)

    return e


def relative_humidity_from_specific_humidity(p, T, q, phase='liquid', 
                                             omega=0.0):
    """
    Computes relative humidity with respect to specified phase from pressure, 
    temperature, and specific humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        phase (optional): condensed water phase ('liquid', 'ice', or 'mixed')
        omega (optional): ice fraction at saturation

    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    RH = relative_humidity(p, T, q, phase=phase, omega=omega)

    return RH
    
    
def relative_humidity_from_mixing_ratio(p, T, r, phase='liquid', omega=0.0):
    """
    Computes relative humidity with respect to specified phase from pressure, 
    temperature, and mixing ratio.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        r (float or ndarray): mixing ratio (kg/kg)
        phase (optional): condensed water phase ('liquid', 'ice', or 'mixed')
        omega (float or ndarray, optional): ice fraction at saturation 
            (default is 0.0)

    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    q = specific_humidity_from_mixing_ratio(r)
    RH = relative_humidity(p, T, q, phase=phase, omega=omega)

    return RH
    
    
def relative_humidity_from_vapour_pressure(T, e, phase='liquid', omega=0.0):
    """
    Computes relative humidity with respect to specified phase from temperature
    and vapour pressure.

    Args:
        T (float or ndarray): temperature (K)
        e (float or ndarray): vapour pressure (Pa)
        phase (optional): condensed water phase ('liquid', 'ice', or 'mixed')
        omega (optional): ice fraction at saturation

    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    RH = e / es

    return RH
    
    
def relative_humidity_from_dewpoint_temperature(T, Td):
    """
    Computes relative humidity with respect to liquid water from temperature
    and dewpoint temperature.

    Args:
        T (float or ndarray): temperature (K)
        Td (float or ndarray): dewpoint temperature (K)

    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    e = saturation_vapour_pressure(Td, phase='liquid')
    es = saturation_vapour_pressure(T, phase='liquid')
    RH = e / es

    return RH
    
    
def relative_humidity_from_frost_point_temperature(T, Tf):
    """
    Computes relative humidity with respect to ice from temperature and frost-
    point temperature.

    Args:
        T (float or ndarray): temperature (K)
        Tf (float or ndarray): frost-point temperature (K)

    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    e = saturation_vapour_pressure(Tf, phase='ice')
    es = saturation_vapour_pressure(T, phase='ice')
    RH = e / es

    return RH


def relative_humidity_from_saturation_point_temperature(T, Ts, omega):
    """
    Computes mixed-phase relative humidity from temperature and saturation-
    point temperature.

    Args:
        T (float or ndarray): temperature (K)
        Ts (float or ndarray): saturation-point temperature (K)
        omega (float or ndarray): ice fraction at saturation

    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    e = saturation_vapour_pressure(Ts, phase='mixed', omega=omega)
    es = saturation_vapour_pressure(T, phase='mixed', omega=omega)
    RH = e / es

    return RH
    

def dewpoint_temperature_from_specific_humidity(p, T, q):
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
    Td = dewpoint_temperature(p, T, q)

    return Td


def dewpoint_temperature_from_mixing_ratio(p, T, r):
    """
    Computes dewpoint temperature from pressure, temperature, and mixing
    ratio.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        r (float or ndarray): mixing ratio (kg/kg)

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """
    q = specific_humidity_from_mixing_ratio(r)
    Td = dewpoint_temperature(p, T, q)

    return Td


def dewpoint_temperature_from_vapour_pressure(p, T, e):
    """
    Computes dewpoint temperature from pressure, temperature, and vapour
    pressure.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        e (float or ndarray): vapour pressure (Pa)

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """
    q = specific_humidity_from_vapour_pressure(p, e)
    Td = dewpoint_temperature(p, T, q)

    return Td
    
    
def dewpoint_temperature_from_relative_humidity(p, T, RH):
    """
    Computes dewpoint temperature from pressure, temperature, and relative
    humidity with respect to liquid water.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """
    q = specific_humidity_from_relative_humidity(p, T, RH, phase='liquid')
    Td = dewpoint_temperature(p, T, q)

    return Td


def dewpoint_temperature_from_frost_point_temperature(p, T, Tf):
    """
    Computes dewpoint temperature from pressure, temperature, and frost-point
    temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Tf (float or ndarray): frost-point temperature (K)

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """
    q = specific_humidity_from_frost_point_temperature(p, Tf)
    Td = dewpoint_temperature(p, T, q)

    return Td


def dewpoint_temperature_from_saturation_point_temperature(p, T, Ts, omega):
    """
    Computes dewpoint temperature from pressure, temperature, and saturation-
    point temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Ts (float or ndarray): saturation-point temperature (K)
        omega (float or ndarray): ice fraction at saturation

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """
    q = specific_humidity_from_saturation_point_temperature(p, Ts, omega)
    Td = dewpoint_temperature(p, T, q)

    return Td


def frost_point_temperature_from_specific_humidity(p, T, q):
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
    Tf = frost_point_temperature(p, T, q)

    return Tf


def frost_point_temperature_from_mixing_ratio(p, T, r):
    """
    Computes frost-point temperature from pressure, temperature, and mixing
    ratio.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        r (float or ndarray): mixing ratio (kg/kg)

    Returns:
        Tf (float or ndarray): frost-point temperature (K)

    """
    q = specific_humidity_from_mixing_ratio(r)
    Tf = frost_point_temperature(p, T, q)

    return Tf


def frost_point_temperature_from_vapour_pressure(p, T, e):
    """
    Computes frost-point temperature from pressure, temperature, and vapour
    pressure.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        e (float or ndarray): vapour pressure (Pa)

    Returns:
        Tf (float or ndarray): frost-point temperature (K)

    """
    q = specific_humidity_from_vapour_pressure(p, e)
    Tf = frost_point_temperature(p, T, q)

    return Tf
    
    
def frost_point_temperature_from_relative_humidity(p, T, RH):
    """
    Computes frost-point temperature from pressure, temperature, and relative
    humidity with respect to ice.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (Pa)

    Returns:
        Tf (float or ndarray): frost-point temperature (K)

    """
    q = specific_humidity_from_relative_humidity(p, T, RH, phase='ice')
    Tf = frost_point_temperature(p, T, q)

    return Tf


def frost_point_temperature_from_dewpoint_temperature(p, T, Td):
    """
    Computes frost-point temperature from pressure, temperature, and dewpoint
    temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Td (float or ndarray): dewpoint temperature (K)

    Returns:
        Tf (float or ndarray): frost-point temperature (K)

    """
    q = specific_humidity_from_dewpoint_temperature(p, Td)
    Tf = frost_point_temperature(p, T, q)

    return Tf


def frost_point_temperature_from_saturation_point_temperature(p, T, Ts, omega):
    """
    Computes frost-point temperature from pressure, temperature, and
    saturation-point temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Ts (float or ndarray): saturation-point temperature (K)
        omega (float or ndarray): ice fraction at saturation

    Returns:
        Tf (float or ndarray): frost-point temperature (K)

    """
    q = specific_humidity_from_saturation_point_temperature(p, Ts, omega)
    Tf = frost_point_temperature(p, T, q)

    return Tf


def saturation_point_temperature_from_specific_humidity(p, T, q,
                                                        converged=0.001):
    """
    Computes saturation-point temperature from pressure, temperature, and 
    specific humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        converged (float, optional): target precision for saturation-point
            temperature (default is 0.001 K)

    Returns:
        Ts (float or ndarray): saturation-point temperature (K)

    """
    Ts = saturation_point_temperature(p, T, q, converged=converged)

    return Ts


def saturation_point_temperature_from_mixing_ratio(p, T, r, converged=0.001):
    """
    Computes saturation-point temperature from pressure, temperature, and 
    mixing ratio.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        r (float or ndarray): mixing ratio (kg/kg)
        converged (float, optional): target precision for saturation-point
            temperature (default is 0.001 K)

    Returns:
        Ts (float or ndarray): saturation-point temperature (K)

    """
    q = specific_humidity_from_mixing_ratio(r)
    Ts = saturation_point_temperature(p, T, q, converged=converged)

    return Ts


def saturation_point_temperature_from_vapour_pressure(p, T, e,
                                                      converged=0.001):
    """
    Computes saturation-point temperature from pressure, temperature, and
    vapour pressure.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        e (float or ndarray): vapour pressure (Pa)
        converged (float, optional): target precision for saturation-point
            temperature (default is 0.001 K)

    Returns:
        Ts (float or ndarray): saturation-point temperature (K)

    """
    q = specific_humidity_from_vapour_pressure(p, e)
    Ts = saturation_point_temperature(p, T, q, converged=converged)

    return Ts
    
    
def saturation_point_temperature_from_relative_humidity(p, T, RH,
                                                        converged=0.001):
    """
    Computes saturation-point temperature from pressure, temperature, and
    mixed-phase relative humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)
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

        # Compute omega
        omega = ice_fraction(Ts)

        # Compute specific humidity
        q = specific_humidity_from_relative_humidity(p, T, RH, phase='mixed', 
                                                     omega=omega)

        # Compute saturation point temperature
        Ts = saturation_point_temperature(p, T, q, converged=converged)

        # Check if solution has converged
        delta = np.abs(Ts - Ts_prev)
        count += 1
        if count > 20:
            print("Ts not converged after 20 iterations")
            break

    return Ts


def saturation_point_temperature_from_dewpoint_temperature(p, T, Td,
                                                           converged=0.001):
    """
    Computes saturation-point temperature from pressure, temperature, and
    dewpoint temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Td (float or ndarray): dewsaturation-point temperature (K)
        converged (float, optional): target precision for saturation-point
            temperature (default is 0.001 K)

    Returns:
        Ts (float or ndarray): saturation-point temperature (K)

    """
    q = specific_humidity_from_dewpoint_temperature(p, Td)
    Ts = saturation_point_temperature(p, T, q, converged=converged)

    return Ts


def saturation_point_temperature_from_frost_point_temperature(p, T, Tf,
                                                              converged=0.001):
    """
    Computes saturation-point temperature from pressure, temperature, and
    frost-point temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Tf (float or ndarray): frost-point temperature (K)
        converged (float, optional): target precision for saturation-point
            temperature (default is 0.001 K)

    Returns:
        Ts (float or ndarray): saturation-point temperature (K)

    """
    q = specific_humidity_from_frost_point_temperature(p, Tf)
    Ts = saturation_point_temperature(p, T, q, converged=converged)

    return Ts


def convert_relative_humidity(T, RH_in, phase_in, phase_out, omega=0.0):
    """
    Converts relative humidity with respect to one phase to relative humidity
    with respect to another phase.

    Args:
        T (float or ndarray): temperature (K)
        RH_in (float or ndarray): input relative humidity (fraction)
        phase_in (str): input condensed water phase (valid options are 
            'liquid', 'ice', or 'mixed')
        phase_out (str): output condensed water phase (valid options are 
            'liquid', 'ice', or 'mixed')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)
    
    Returns:
        RH_out (float or ndarray): output relative humidity (fraction)

    """
    es_in = saturation_vapour_pressure(T, phase=phase_in, omega=omega)
    es_out = saturation_vapour_pressure(T, phase=phase_out, omega=omega)

    RH_out = RH_in * es_in / es_out

    return RH_out
