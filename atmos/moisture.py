from atmos.constant import eps
from atmos.thermo import (mixing_ratio, 
                          vapour_pressure,
                          relative_humidity,
                          saturation_vapour_pressure,
                          saturation_specific_humidity,
                          saturation_mixing_ratio, 
                          dewpoint_temperature,
                          frost_point_temperature,
                          saturation_point_temperature)


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


def specific_humidity_from_relative_humidity(p, T, RH, phase='liquid', 
                                             icefrac=None):
    """
    Computes specific humidity from pressure, temperature, and relative 
    humidity.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        RH: relative humidity with respect to specified phase (fraction)
        phase (optional): condensed water phase ('liquid', 'ice', or 'mixed')
        icefrac (optional, required for mixed phase): ice fraction at saturation (fraction)
        
    Returns:
        q: specific humidity (kg/kg)

    """
    es = saturation_vapour_pressure(T, phase=phase, icefrac=icefrac)
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
    q = saturation_specific_humidity(p, Td, phase='liquid')
    
    return q


def specific_humidity_from_frost_point_temperature(p, Tf):
    """
    Computes specific humidity from pressure and frost-point temperature.

    Args:
        p: pressure (Pa)
        Tf: frost-point temperature (K)

    Returns:
        q: specific humidity (kg/kg)

    """    
    q = saturation_specific_humidity(p, Tf, phase='ice')
    
    return q


def specific_humidity_from_saturation_point_temperature(p, Ts, icefrac):
    """
    Computes specific humidity from pressure and saturation-point temperature.

    Args:
        p: pressure (Pa)
        Ts: saturation-point temperature (K)
        icefrac: ice fraction at saturation

    Returns:
        q: specific humidity (kg/kg)

    """    
    q = saturation_specific_humidity(p, Ts, phase='mixed', icefrac=icefrac)
    
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


def mixing_ratio_from_relative_humidity(p, T, RH, phase='liquid', 
                                        icefrac=None):
    """
    Computes mixing ratio from pressure, temperature, and relative humidity.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        RH: relative humidity with respect to specified phase (fraction)
        phase (optional): condensed water phase ('liquid', 'ice', or 'mixed')
        icefrac (optional, required for mixed phase): ice fraction at saturation (required for phase='mixed')

    Returns:
        r: mixing ratio (kg/kg)

    """
    es = saturation_vapour_pressure(T, phase=phase, icefrac=icefrac)
    e = RH * es
    r = mixing_ratio_from_vapour_pressure(p, e)
    
    return r


def mixing_ratio_from_dewpoint_temperature(p, Td):
    """
    Computes mixing ratio from pressure and dewpoint temperature.

    Args:
        p: pressure (Pa)
        Td: dewpoint temperature (K)

    Returns:
        r: mixing ratio (kg/kg)

    """    
    r = saturation_mixing_ratio(p, Td, phase='liquid')
    
    return r


def mixing_ratio_from_frost_point_temperature(p, Tf):
    """
    Computes mixing ratio from pressure and frost-point temperature.

    Args:
        p: pressure (Pa)
        Tf: frost-point temperature (K)

    Returns:
        r: mixing ratio (kg/kg)

    """    
    r = saturation_mixing_ratio(p, Tf, phase='ice')
    
    return r


def mixing_ratio_from_saturation_point_temperature(p, Ts, icefrac):
    """
    Computes mixing ratio from pressure and saturation-point temperature.

    Args:
        p: pressure (Pa)
        Ts: saturation-point temperature (K)
        icefrac: ice fraction at saturation

    Returns:
        r: mixing ratio (kg/kg)

    """    
    r = saturation_mixing_ratio(p, Ts, phase='mixed', icefrac=icefrac)
    
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


def vapour_pressure_from_relative_humidity(T, RH, phase='liquid', 
                                           icefrac=None):
    """
    Computes vapour pressure from temperature and relative humidity.

    Args:
        T: temperature (K)
        RH: relative humidity with respect to specified phase (fraction)
        phase (optional): condensed water phase ('liquid', 'ice', or 'mixed')
        icefrac (optional, required for mixed phase): ice fraction at saturation (fraction)

    Returns:
        e: vapour pressure (Pa)

    """
    es = saturation_vapour_pressure(T, phase=phase, icefrac=icefrac)
    e = RH * es

    return e


def vapour_pressure_from_dewpoint_temperature(Td):
    """
    Computes vapour pressure from dewpoint temperature.

    Args:
        Td: dewpoint temperature (K)

    Returns:
        e: vapour pressure (Pa)

    """
    e = saturation_vapour_pressure(Td, phase='liquid')

    return e


def vapour_pressure_from_frost_point_temperature(Tf):
    """
    Computes vapour pressure from frost-point temperature.

    Args:
        Tf: frost-point temperature (K)

    Returns:
        e: vapour pressure (Pa)

    """
    e = saturation_vapour_pressure(Tf, phase='ice')

    return e


def vapour_pressure_from_saturation_point_temperature(Ts, icefrac):
    """
    Computes vapour pressure from saturation-point temperature.

    Args:
        Ts: saturation-point temperature (K)
        icefrac: ice fraction at saturation

    Returns:
        e: vapour pressure (Pa)

    """
    e = saturation_vapour_pressure(Ts, phase='mixed', icefrac=icefrac)

    return e


def relative_humidity_from_specific_humidity(p, T, q, phase='liquid', 
                                             icefrac=None):
    """
    Computes relative humidity from pressure, temperature, and specific humidity.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)
        phase (optional): condensed water phase ('liquid', 'ice', or 'mixed')
        icefrac (optional, required for mixed phase): ice fraction at saturation (fraction)

    Returns:
        RH: relative humidity with respect to specified phase (fraction)

    """
    RH = relative_humidity(p, T, q, phase=phase, icefrac=icefrac)

    return RH
    
    
def relative_humidity_from_mixing_ratio(p, T, r, phase='liquid', 
                                        icefrac=None):
    """
    Computes relative humidity from pressure, temperature, and mixing ratio.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        r: mixing ratio (kg/kg)
        phase (optional): condensed water phase ('liquid', 'ice', or 'mixed')
        icefrac (optional, required for mixed phase): ice fraction at saturation (fraction)

    Returns:
        RH: relative humidity with respect to specified phase (fraction)

    """
    q = specific_humidity_from_mixing_ratio(r)
    RH = relative_humidity(p, T, q, phase=phase, icefrac=icefrac)

    return RH
    
    
def relative_humidity_from_vapour_pressure(T, e, phase='liquid', 
                                           icefrac=None):
    """
    Computes relative humidity from temperature and vapour pressure.

    Args:
        T: temperature (K)
        e: vapour pressure (Pa)
        phase (optional): condensed water phase ('liquid', 'ice', or 'mixed')
        icefrac (optional, required for mixed phase): ice fraction at saturation (fraction)

    Returns:
        RH: relative humidity with respect to specified phase (fraction)

    """
    es = saturation_vapour_pressure(T, phase=phase, icefrac=icefrac)
    RH = e / es

    return RH
    
    
def relative_humidity_from_dewpoint_temperature(T, Td):
    """
    Computes relative humidity from temperature and dewpoint temperature.

    Args:
        T: temperature (K)
        Td: dewpoint temperature (K)

    Returns:
        RH: relative humidity with respect to liquid water (fraction)

    """
    e = saturation_vapour_pressure(Td, phase='liquid')
    es = saturation_vapour_pressure(T, phase='liquid')
    RH = e / es

    return RH
    
    
def relative_humidity_from_frost_point_temperature(T, Tf):
    """
    Computes relative humidity from temperature and frost-point temperature.

    Args:
        T: temperature (K)
        Tf: frost-point temperature (K)

    Returns:
        RH: relative humidity with respect to ice (fraction)

    """
    e = saturation_vapour_pressure(Tf, phase='ice')
    es = saturation_vapour_pressure(T, phase='ice')
    RH = e / es

    return RH


def relative_humidity_from_saturation_point_temperature(T, Ts, icefrac):
    """
    Computes relative humidity from temperature and saturation-point 
    temperature.

    Args:
        T: temperature (K)
        Ts: saturation-point temperature (K)
        icefrac: ice fraction at saturation

    Returns:
        RH: mixed-phase relative humidity (fraction)

    """
    e = saturation_vapour_pressure(Ts, phase='mixed', icefrac=icefrac)
    es = saturation_vapour_pressure(T, phase='mixed', icefrac=icefrac)
    RH = e / es

    return RH
    

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
        RH: relative humidity with respect to liquid water (Pa)

    Returns:
        Td: dewpoint temperature (K)

    """
    q = specific_humidity_from_relative_humidity(p, T, RH, phase='liquid')
    Td = dewpoint_temperature(p, T, q)
    
    return Td


def frost_point_temperature_from_specific_humidity(p, T, q):
    """
    Computes frost-point temperature from pressure, temperature, and specific
    humidity.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)

    Returns:
        Tf: frost-point temperature (K)

    """
    Tf = frost_point_temperature(p, T, q)
    
    return Tf


def frost_point_temperature_from_mixing_ratio(p, T, r):
    """
    Computes frost-point temperature from pressure, temperature, and mixing
    ratio.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        r: mixing ratio (kg/kg)

    Returns:
        Tf: frost-point temperature (K)

    """
    q = specific_humidity_from_mixing_ratio(r)
    Tf = frost_point_temperature(p, T, q)
    
    return Tf


def frost_point_temperature_from_vapour_pressure(p, T, e):
    """
    Computes frost-point temperature from pressure, temperature, and vapour
    pressure.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        e: vapour pressure (Pa)

    Returns:
        Tf: frost-point temperature (K)

    """
    q = specific_humidity_from_vapour_pressure(p, e)
    Tf = frost_point_temperature(p, T, q)
    
    return Tf
    
    
def frost_point_temperature_from_relative_humidity(p, T, RH):
    """
    Computes frost-point temperature from pressure, temperature, and relative
    humidity.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        RH: relative humidity with respect to ice (Pa)

    Returns:
        Tf: frost-point temperature (K)

    """
    q = specific_humidity_from_relative_humidity(p, T, RH, phase='ice')
    Tf = frost_point_temperature(p, T, q)
    
    return Tf


def saturation_point_temperature_from_specific_humidity(p, T, q, icefrac):
    """
    Computes saturation-point temperature from pressure, temperature, and 
    specific humidity.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        q: specific humidity (kg/kg)
        icefrac: ice fraction at saturation

    Returns:
        Ts: saturation-point temperature (K)

    """
    Ts = saturation_point_temperature(p, T, q, icefrac)
    
    return Ts


def saturation_point_temperature_from_mixing_ratio(p, T, r, icefrac):
    """
    Computes saturation-point temperature from pressure, temperature, and 
    mixing ratio.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        r: mixing ratio (kg/kg)
        icefrac: ice fraction at saturation

    Returns:
        Ts: saturation-point temperature (K)

    """
    q = specific_humidity_from_mixing_ratio(r)
    Ts = saturation_point_temperature(p, T, q, icefrac)
    
    return Ts


def saturation_point_temperature_from_vapour_pressure(p, T, e, icefrac):
    """
    Computes saturation-point temperature from pressure, temperature, and 
    vapour pressure.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        e: vapour pressure (Pa)
        icefrac: ice fraction at saturation

    Returns:
        Ts: saturation-point temperature (K)

    """
    q = specific_humidity_from_vapour_pressure(p, e)
    Ts = saturation_point_temperature(p, T, q, icefrac)
    
    return Ts
    
    
def saturation_point_temperature_from_relative_humidity(p, T, RH, icefrac):
    """
    Computes saturation-point temperature from pressure, temperature, and
    relative humidity.

    Args:
        p: pressure (Pa)
        T: temperature (K)
        RH: mixed-phase relative humidity (fraction)
        icefrac: ice fraction at saturation
    
    Returns:
        Ts: saturation-point temperature (K)

    """
    q = specific_humidity_from_relative_humidity(p, T, RH, phase='mixed', 
                                                 icefrac=icefrac)
    Ts = saturation_point_temperature(p, T, q, icefrac)
    
    return Ts


def convert_relative_humidity(T, RH_in, phase_in='liquid', phase_out='ice', 
                              icefrac=None):
    """
    Converts relative humidity with respect to one phase to relative humidity
    with respect to another phase.

    Args:
        T: temperature (K)
        RH_in: relative humidity with respect to phase_in (fraction)
        phase_in (optional): input condensed water phase ('liquid', 'ice', or 'mixed')
        phase_out (optional): output condensed water phase ('liquid', 'ice', or 'mixed')
        icefrac (optional, required for mixed phase): ice fraction at saturation (fraction)
    
    Returns:
        RH_out: relative humidity with respect to phase_out (K)

    """
    es_in = saturation_vapour_pressure(T, phase=phase_in, icefrac=icefrac)
    es_out = saturation_vapour_pressure(T, phase=phase_out, icefrac=icefrac)

    RH_out = RH_in * es_in / es_out

    return RH_out
