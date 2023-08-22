from atmos.constant import eps
from atmos.thermo import (mixing_ratio, vapour_pressure, 
                          saturation_vapour_pressure, 
                          dewpoint_temperature)


def specific_humidity_from_mixing_ratio(r):
    """
    Computes specific humidity from water vapour mixing ratio.

    Args:
        r: mixing ratio (kg/kg)

    Returns:
        q: specific humidity (kg/kg)

    """
    
    q = r / (1 + r)
    
    return q


def specific_humidity_from_vapour_pressure(p, e):
    """
    Computes specific humidity from pressure and vapour pressure.

    Args:
        p: pressure (Pa)
        e: vapour pressure (Pa)

    Returns:
        q: specific humidity (kg/kg)

    """
    q = eps * e / (p - (1 - eps) * e)
    
    return q


def specific_humidity_from_relative_humidity(p, T, RH):
    """
    Computes specific humidity from pressure, temperature, and relative 
    humidity.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        RH: relative humidity (fraction)

    Returns:
        q: specific humidity (kg/kg)

    """
    es = saturation_vapour_pressure(T)
    e = RH * es
    q = specific_humidity_from_vapour_pressure(p, e)
    
    return q


def specific_humidity_from_dewpoint_temperature(p, Td):
    """
    Computes specific humidity from pressure and dewpoint temperature.

    Args:
        p: pressure (Pa)
        Td: dewpoint temperature (K)

    Returns:
        q: specific humidity (kg/kg)

    """    
    q = sat_specific_humidity(p, Td)
    
    return q


def mixing_ratio_from_specific_humidity(q):
    """
    Computes mixing ratio from specific humidity.

    Args:
        q: specific humidity (kg/kg)

    Returns:
        r: mixing ratio (kg/kg)

    """
    r = mixing_ratio(q)
    
    return r


def mixing_ratio_from_vapour_pressure(p, e):
    """
    Computes mixing ratio from pressure and vapour pressure.

    Args:
        p: pressure (Pa)
        e: vapour pressure (Pa)

    Returns:
        r: mixing ratio (kg/kg)

    """
    r = eps * e / (p - e)
    
    return r


def mixing_ratio_from_dewpoint_temperature(p, Td):
    """
    Computes mixing ratio from pressure and dewpoint temperature.

    Args:
        p: pressure (Pa)
        Td: dewpoint temperature (K)

    Returns:
        q: specific humidity (kg/kg)

    """    
    e = saturation_vapour_pressure(Td)
    r = mixing_ratio_from_vapour_pressure(p, e)
    
    return r
    

def mixing_ratio_from_relative_humidity(p, T, RH):
    """
    Computes mixing ratio from pressure, temperature, and relative 
    humidity.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        RH: relative humidity (fraction)

    Returns:
        r: mixing ratio (kg/kg)

    """
    es = saturation_vapour_pressure(T)
    e = RH * es
    r = mixing_ratio_from_vapour_pressure(p, e)
    
    return r


def vapour_pressure_from_specific_humidity(p, q):
    """
    Computes vapour pressure from pressure and specific humidity.

    Args:
        p: pressure (Pa)
        q: specific humidity (kg/kg)

    Returns:
        e: vapour pressure (Pa)

    """
    e = vapour_pressure(p, q)

    return e


def vapour_pressure_from_mixing_ratio(p, r):
    """
    Computes vapour pressure from pressure and mixing ratio.

    Args:
        p: pressure (Pa)
        r: mixing ratio (kg/kg)

    Returns:
        e: vapour pressure (Pa)

    """
    e = p * r / (r + eps)

    return e


def vapour_pressure_from_dewpoint_temperature(Td):
    """
    Computes vapour pressure from dewpoint temperature.

    Args:
        Td: dewpoint temperature (K)

    Returns:
        e: vapour pressure (Pa)

    """
    e = saturation_vapour_pressure(Td)

    return e


def vapour_pressure_from_relative_humidity(T, RH):
    """
    Computes vapour pressure from temperature and relative humidity.

    Args:
        T: temperature (K)
        RH: relative humidity (fraction)

    Returns:
        e: vapour pressure (Pa)

    """
    es = saturation_vapour_pressure(T)
    e = RH * es

    return e


def dewpoint_temperature_from_specific_humidity(p, T, q):
    """
    Computes dewpoint temperature from pressure, temperature, and specific
    humidity.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)

    Returns:
        Td: dewpoint temperature (K)

    """
    Td = dewpoint_temperature(p, T, q)
    
    return Td


def dewpoint_temperature_from_mixing_ratio(p, T, r):
    """
    Computes dewpoint temperature from pressure, temperature, and mixing
    ratio.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        r: mixing ratio (kg/kg)

    Returns:
        Td: dewpoint temperature (K)

    """
    q = specific_humidity_from_mixing_ratio(r)
    Td = dewpoint_temperature(p, T, q)
    
    return Td


def dewpoint_temperature_from_vapour_pressure(p, T, e):
    """
    Computes dewpoint temperature from pressure, temperature, and vapour
    pressure.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        e: vapour pressure (Pa)

    Returns:
        Td: dewpoint temperature (K)

    """
    q = specific_humidity_from_vapour_pressure(p, e)
    Td = dewpoint_temperature(p, T, q)
    
    return Td
    
    
def dewpoint_temperature_from_relative_humidity(p, T, RH):
    """
    Computes dewpoint temperature from pressure, temperature, and relative
    humidity.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        RH: relatie humidity (Pa)

    Returns:
        Td: dewpoint temperature (K)

    """
    q = specific_humidity_from_relative_humidity(p, T, RH)
    Td = dewpoint_temperature(p, T, q)
    
    return Td
